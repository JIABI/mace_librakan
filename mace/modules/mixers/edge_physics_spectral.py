from __future__ import annotations

from typing import Dict, Optional
import math
import torch
from torch import nn

from mace.modules.mixers.PhysicsSpectralKernel import PhysicsSpectralKernel  # adjust import path if needed


def _get_activation(name: str):
    """
    Return an activation class based on a string name.
    """
    name = name.lower()
    if name == "gelu":
        return nn.GELU
    if name in ("silu", "swish"):
        return nn.SiLU
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unknown activation: {name}")


class BesselRadialBasis(nn.Module):
    """
    Spherical-Bessel-inspired radial basis with polynomial envelope.

    Maps distances r ∈ (0, cutoff] to num_basis smooth radial basis functions.
    Used as the low-frequency geometric dictionary.
    """

    def __init__(
        self,
        num_basis: int,
        cutoff: float,
        envelope_exponent: int = 5,
        eps: float = 1e-8,
        internal_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if num_basis <= 0:
            raise ValueError("num_basis must be positive.")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")

        self.num_basis = int(num_basis)
        self.cutoff = float(cutoff)
        self.envelope_exponent = int(envelope_exponent)
        self.eps = float(eps)
        self.internal_dtype = internal_dtype

        # Frequencies π n / cutoff
        n = torch.arange(1, self.num_basis + 1, dtype=internal_dtype)
        self.register_buffer("freq", math.pi * n / self.cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: [..., 1] or [...] distances.

        Returns:
            [..., num_basis] radial basis values.
        """
        if r.dim() == 0:
            r = r.view(1, 1)
        if r.size(-1) == 1:
            r = r[..., 0]

        r = r.to(self.freq.dtype)

        x = r.unsqueeze(-1) * self.freq
        j0 = torch.sin(x) / (x + self.eps)

        rc = torch.clamp(1.0 - r / self.cutoff, min=0.0)
        envelope = rc.pow(self.envelope_exponent).unsqueeze(-1)

        return envelope * j0


class EdgePhysicsSpectralMixer(nn.Module):
    """
    Physics-guided spectral radial mixer for edge features.

    Final architecture:
      1. Low-frequency branch: Bessel radial basis (smooth geometric prior).
      2. High-frequency branch: PhysicsSpectralKernel (Fourier-like + shrinkage).
      3. Spectral gating conditioned on edge features.
      4. Optional low-rank residual correction in spectral coordinates.

    This module is designed to replace the standard edge MLP in MACE.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_basis: int = 12,
        cutoff: float = 7.0,
        envelope_exponent: int = 5,
        shared_dim: Optional[int] = None,
        use_residual: bool = True,
        residual_rank: Optional[int] = None,
        residual_hidden: int = 32,
        use_spectral_gating: bool = True,
        gate_hidden: Optional[int] = 64,
        base_activation: str = "gelu",
        pre_layernorm: bool = True,
        dropout: float = 0.0,
        residual_gate_init_bias: float = -1.0,
        # High-frequency physics branch
        use_phys_kernel: bool = True,
        phys_num_freq: int = 4,
        phys_omega_max: float = math.pi,
        phys_shrinkage_lambda: float = 1e-2,
        phys_shrinkage_p: float = 0.7,
        phys_trainable_lambda: bool = False,
    ):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive.")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_basis = int(num_basis)
        self.cutoff = float(cutoff)
        self.envelope_exponent = int(envelope_exponent)
        self.use_residual = bool(use_residual)
        self.use_spectral_gating = bool(use_spectral_gating)
        self.use_phys_kernel = bool(use_phys_kernel)

        self.internal_dtype = torch.float32
        Act = _get_activation(base_activation)

        # 1) Low-frequency branch (Bessel radial basis)
        self.radial_basis = BesselRadialBasis(
            num_basis=self.num_basis,
            cutoff=self.cutoff,
            envelope_exponent=self.envelope_exponent,
            internal_dtype=self.internal_dtype,
        )

        # 2) High-frequency physics branch
        if self.use_phys_kernel:
            self.phys_kernel = PhysicsSpectralKernel(
                num_freq=phys_num_freq,
                cutoff=self.cutoff,
                omega_max=phys_omega_max,
                shrinkage_lambda=phys_shrinkage_lambda,
                shrinkage_p=phys_shrinkage_p,
                trainable_lambda=phys_trainable_lambda,
                internal_dtype=self.internal_dtype,
            )
            self.num_phys = int(phys_num_freq)
        else:
            self.phys_kernel = None
            self.num_phys = 0

        # Optional LayerNorm on edge features
        self.pre_ln = nn.LayerNorm(self.in_dim) if pre_layernorm else None

        # Combined radial dimension (low + high)
        total_radial_dim = self.num_basis + self.num_phys

        # Shared spectral projection
        shared_dim = min(64, max(self.in_dim, total_radial_dim))
        shared_dim_raw = shared_dim or max(self.in_dim, total_radial_dim)
        shared_cap = 128
        self.shared_dim = int(min(shared_dim_raw, shared_cap))

        self.shared_proj = nn.Linear(
            total_radial_dim,
            self.shared_dim,
            bias=False,
        )

        # 3) Spectral gating network
        if self.use_spectral_gating:
            gate_hidden_raw = gate_hidden or self.in_dim
            gate_hidden_cap = 128
            gate_hidden_eff = int(min(gate_hidden_raw, gate_hidden_cap))

            self.gate_net = nn.Sequential(
                nn.Linear(self.in_dim, gate_hidden_eff, bias=False),
                Act(),
                nn.Linear(gate_hidden_eff, total_radial_dim),
            )
        else:
            self.gate_net = None

        # Physics head: shared spectral embedding -> edge output
        self.phys_head = nn.Linear(self.shared_dim, self.out_dim, bias=False)

        # 4) Optional low-rank residual branch
        self.residual_rank = None
        if self.use_residual:
            if residual_rank is None:
                residual_rank_raw = max(4, self.out_dim // 2)
            else:
                residual_rank_raw = int(residual_rank)
            rank_cap = 16
            self.residual_rank = int(min(residual_rank_raw, rank_cap))

            residual_hidden_cap = 64
            residual_hidden_eff = int(min(residual_hidden, residual_hidden_cap))

            res_in_dim = self.shared_dim + self.in_dim
            self.res_mlp = nn.Sequential(
                nn.Linear(res_in_dim, residual_hidden_eff, bias=False),
                Act(),
                nn.Dropout(dropout) if dropout > 1e-8 else nn.Identity(),
                nn.Linear(residual_hidden_eff, self.residual_rank, bias=False),
            )
            self.res_proj = nn.Linear(self.residual_rank, self.out_dim, bias=False)
            self.res_gate = nn.Linear(self.in_dim, 1)

            # Initialize residual gate to be initially closed
            nn.init.constant_(self.res_gate.bias, float(residual_gate_init_bias))

            last_linear = self.res_mlp[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
        else:
            self.res_mlp = None
            self.res_proj = None
            self.res_gate = None

        self.out_dropout = nn.Dropout(dropout) if dropout > 1e-8 else None

        # Ensure internal parameters use the chosen internal dtype
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                m.to(self.internal_dtype)

    # ----------------- helpers ----------------- #

    def _build_r(
        self,
        edge_feats: torch.Tensor,
        edge_geom: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Build a radius tensor for the radial bases.

        Prefer explicit distances r_ij from geometry.
        If unavailable, fall back to ||edge_feats|| as a proxy.
        """
        if edge_geom is not None:
            r_ij = edge_geom.get("r_ij", None)
            if r_ij is not None:
                return r_ij.to(self.internal_dtype)

        r = edge_feats.norm(dim=-1, keepdim=True)
        return r.to(self.internal_dtype)

    # ----------------- forward ----------------- #

    def forward(
        self,
        edge_feats: torch.Tensor,
        edge_geom: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            edge_feats: [n_edges, in_dim] scalar edge features.
            edge_geom: optional dict with key 'r_ij': [n_edges, 1] distances.

        Returns:
            Mixed spectral edge features [n_edges, out_dim].
        """
        if edge_feats.dim() != 2:
            raise ValueError(
                f"edge_feats must have shape [n_edges, in_dim], got {edge_feats.shape}."
            )

        in_dtype = edge_feats.dtype
        x = edge_feats.to(self.internal_dtype)

        # Optional LayerNorm
        if self.pre_ln is not None:
            x_norm = self.pre_ln(x.to(self.pre_ln.weight.dtype)).to(self.internal_dtype)
        else:
            x_norm = x

        # 1) Low- and high-frequency radial basis
        r = self._build_r(x_norm, edge_geom)        # [E, 1]
        phi_low = self.radial_basis(r)              # [E, num_basis]

        if self.use_phys_kernel and self.phys_kernel is not None:
            phi_high = self.phys_kernel(r)          # [E, num_phys]
            phi_all = torch.cat([phi_low, phi_high], dim=-1)  # [E, total_radial_dim]
        else:
            phi_all = phi_low

        # 2) Spectral gating
        if self.use_spectral_gating and self.gate_net is not None:
            gates = torch.sigmoid(self.gate_net(x_norm))      # [E, total_radial_dim]
            phi_gated = phi_all * gates
        else:
            phi_gated = phi_all

        # 3) Shared spectral embedding
        h_shared = self.shared_proj(phi_gated)      # [E, shared_dim]
        w_phys = self.phys_head(h_shared)           # [E, out_dim]

        # 4) Optional low-rank residual in spectral coordinates
        if self.use_residual and self.res_mlp is not None:
            res_input = torch.cat([h_shared, x_norm], dim=-1)  # [E, shared_dim+in_dim]
            z = self.res_mlp(res_input)                       # [E, R]
            delta = self.res_proj(z)                          # [E, out_dim]
            lam = torch.sigmoid(self.res_gate(x_norm))        # [E, 1]
            out = w_phys + lam * delta                        # [E, out_dim]
        else:
            out = w_phys

        if self.out_dropout is not None:
            out = self.out_dropout(out)

        # Cast back to original dtype for compatibility with the rest of MACE
        return out.to(in_dtype)

    # ----------------- regularization hook ----------------- #

    def regularization_loss(
        self,
        l2_alpha: float = 0.0,
        decor_beta: float = 0.0,
        phys_coeff: float = 0.0,
        delta_coeff: float = 0.0,
    ) -> torch.Tensor:
        """
        Optional hook for adding radial / spectral regularization.

        Currently implemented as a no-op placeholder.
        """
        return torch.zeros((), device=next(self.parameters()).device)
