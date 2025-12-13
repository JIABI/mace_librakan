from __future__ import annotations
import math
import torch
from torch import nn
from typing import Optional, List, Tuple
from .nufft_es import NUFFTES
from .shrinkage import GeneralizedShrinkage
from e3nn import o3

def _ensure_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map float32→complex64, float64→complex128; otherwise keep as is."""
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64 if "32" in str(dtype) else torch.complex128


class GeneralLibraKAN(nn.Module):
    """
    GeneralLibraKAN: 
        f(x) = W2[(1 - alpha) * phi_l(x) + alpha * T^H S_{lambda,p}( eps ⊙ T(phi_g_in(x)) )]
    where
        - phi_l is a local branch (MLP or single activation) over hidden dim H
        - T / T^H are NUFFT-ES forward / adjoint
        - S_{lambda,p} is generalized shrinkage
        - eps are learnable per-frequency gains
        - alpha is a convex mixing weight between local and spectral paths
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Optional[int] = None,  # H; default = max(in_dim, out_dim)
        # shrinkage
        p: float = 1.0,
        lam: float = 1e-2,
        trainable_lambda: bool = True,
        # local branch
        activation: str = "gelu",
        local_kind: str = "mlp",          # "mlp" | "act"
        local_layers: Optional[List[int]] = None,  # e.g. [H] or [H, H]
        dropout: float = 0.0,
        use_layernorm: bool = False,
        # spectral dictionary
        F: Optional[int] = None,          # number of spectral samples (K)
        spectral_scale: float = 0.7,
        es_fmax: Optional[float] = None,
        omega_max: Optional[float] = None,  # legacy alias
        learn_omega: bool = True,
        # input sharing
        share_input_proj: bool = False,   # if True, W_l = W_g
        # alpha control
        alpha_min: float = 0.0,
        alpha_tau: float = 1.0,
        # metrics / active set
        active_threshold: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        H = hidden if (hidden is not None and hidden > 0) else max(in_dim, out_dim)
        self.in_dim, self.out_dim, self.H = int(in_dim), int(out_dim), int(H)

        # ---------------- Local branch ----------------
        if hasattr(torch.nn.functional, activation):
            self._act_fn = getattr(torch.nn.functional, activation)
        else:
            self._act_fn = torch.nn.functional.gelu

        local_layers = [] if local_layers is None else list(local_layers)
        layers: List[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(self.H))

        if local_kind == "mlp":
            if len(local_layers) == 0:
                local_layers = [self.H]
            dims = [self.H] + [
                int(v) if not isinstance(v, str) else self.H for v in local_layers
            ] + [self.H]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
                    if dropout and dropout > 0:
                        layers.append(nn.Dropout(dropout))
        else:
            # "act": no extra MLP, only a nonlinearity after the input projection
            pass

        self.local_stack = nn.Sequential(*layers) if len(layers) > 0 else None

        # Input projections for local / spectral branches
        self.lin_local = nn.Linear(self.in_dim, self.H, bias=True)
        if share_input_proj:
            self.lin_spec = self.lin_local  # share weights with local branch
        else:
            self.lin_spec = nn.Linear(self.in_dim, self.H, bias=True)

        # Output projection
        self.lin2 = nn.Linear(self.H, self.out_dim, bias=True)

        # ---------------- Spectral branch ----------------
        K = int(F) if (F is not None and F > 0) else self.H
        if es_fmax is not None:
            wmax = float(es_fmax)
        elif omega_max is not None:
            wmax = float(omega_max)
        else:
            wmax = float(spectral_scale) * math.pi

        omega = torch.linspace(-wmax, wmax, steps=max(1, K))
        if learn_omega:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)

        # Per-frequency gain eps (real-valued, applied to complex y)
        self.freq_gain = nn.Parameter(torch.ones(K))

        self.nufft = NUFFTES()
        self.shrink = GeneralizedShrinkage(
            lam=lam, p=p, trainable_lambda=trainable_lambda
        )

        # ---------------- Alpha controls ----------------
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.alpha_min = float(alpha_min)
        self.alpha_tau = float(alpha_tau)

        # ---------------- Metrics / active set ----------------
        self.register_buffer(
            "_last_active_freq", torch.tensor(0.0), persistent=False
        )
        self.active_threshold = float(active_threshold)

    @property
    def active_freq(self) -> float:
        """Mean number of active frequencies (|y_k|>tau) in last forward."""
        return float(self._last_active_freq.item())

    def _local_branch(self, h: torch.Tensor) -> torch.Tensor:
        """Apply local MLP or a single activation on hidden features."""
        if self.local_stack is not None:
            return self.local_stack(h)
        return self._act_fn(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_dim] (float32/float64)
        return: [..., out_dim]
        """
        # -------- Local path --------
        h_local_in = self.lin_local(x)              # [..., H]
        local = self._local_branch(h_local_in)      # [..., H]

        # -------- Spectral path --------
        h_spec_in = self.lin_spec(x)                # [..., H]
        complex_dtype = _ensure_complex_dtype(h_spec_in.dtype)

        # NUFFT forward: T(h_spec_in) -> y in C^K
        y = self.nufft(h_spec_in, self.omega.to(h_spec_in.device))
        if y.dtype != complex_dtype:
            y = y.to(complex_dtype)

        # Per-frequency amplitude modulation
        y = y * self.freq_gain.view(1, -1).to(y.real.dtype)

        # Generalized shrinkage
        y = self.shrink(y)

        # Active set mask + statistics
        with torch.no_grad():
            mag = torch.abs(y)
            active_mask = (mag > self.active_threshold).to(h_spec_in.dtype)
            self._last_active_freq.copy_(active_mask.sum(dim=-1).float().mean())
        if self.active_threshold > 0.0:
            y = y * (active_mask.to(y.dtype))

        # NUFFT adjoint: T^H(y) -> z_spec in R^H
        z_spec = self.nufft.adjoint(
            y, self.omega.to(h_spec_in.device), self.H
        ).real

        # -------- Fuse local + spectral --------
        tau = self.alpha_tau if self.alpha_tau > 0 else 1.0
        alpha = torch.sigmoid(self.alpha_logit / tau)
        if self.alpha_min > 0:
            alpha = torch.clamp(alpha, min=self.alpha_min)

        fused = (1.0 - alpha) * local + alpha * z_spec  # [..., H]

        # -------- Output projection --------
        out = self.lin2(fused)  # [..., out_dim]
        return out

class ReadoutScalarLibraKAN(nn.Module):
    """
    Scalar-only LibraKAN for MACE readout.
    This module assumes the input x has already been mapped to a flat
    hidden representation with dimension H = hidden_irreps.dim.
    It:
      1) extracts all l=0 channels according to hidden_irreps,
      2) applies a GeneralLibraKAN only on these scalar channels,
      3) adds a gated residual back to the scalar slots,
      4) leaves non-scalar channels unchanged.
    This keeps O(3)-equivariant structure intact while giving a
    spectral Libra correction on invariants only.
    """
    def __init__(
            self,
            hidden_irreps: o3.Irreps,
            hidden: Optional[int] = None,  # internal hidden dim for LibraKAN
            # GeneralLibraKAN hyper-params (subset)
            p: float = 1.0,
            lam: float = 1e-2,
            trainable_lambda: bool = True,
            activation: str = "gelu",
            local_kind: str = "act",  # "act" or "mlp"
            local_layers: Optional[List[int]] = None,
            dropout: float = 0.0,
            use_layernorm: bool = False,
            F: Optional[int] = None,
            spectral_scale: float = 0.7,
            es_fmax: Optional[float] = None,
            omega_max: Optional[float] = None,
            learn_omega: bool = True,
            share_input_proj: bool = False,
            alpha_min: float = 0.0,
            alpha_tau: float = 1.0,
            active_threshold: float = 1e-3,
    ):
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.H = self.hidden_irreps.dim
        # ---- build a boolean mask for l=0 slots in the flattened layout ----
        scalar_mask: List[bool] = []
        for mul, ir in self.hidden_irreps:
            block_dim = mul * ir.dim
            is_scalar = (ir.l == 0)
            scalar_mask.extend([is_scalar] * block_dim)
        if len(scalar_mask) != self.H:
            raise ValueError(
                f"Scalar mask length {len(scalar_mask)} != hidden dim {self.H}"
            )
        scalar_mask_tensor = torch.tensor(scalar_mask, dtype=torch.bool)
        self.register_buffer("scalar_mask", scalar_mask_tensor, persistent=False)
        scalar_dim = int(self.scalar_mask.sum().item())
        self.scalar_dim = scalar_dim
        if scalar_dim == 0:
            # no l=0 channels -> this module will act as identity
            self.libra = None
            self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.gate_weight = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.gate_bias = nn.Parameter(torch.zeros(1), requires_grad=False)
            return
        if hidden is None or hidden <= 0:
            hidden = scalar_dim

        # ---- core LibraKAN ONLY on scalar channels ----

        self.libra = GeneralLibraKAN(
            in_dim=scalar_dim,
            out_dim=scalar_dim,
            hidden=hidden,
            p=p,
            lam=lam,
            trainable_lambda=trainable_lambda,
            activation=activation,
            local_kind=local_kind,
            local_layers=local_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            F=F,
            spectral_scale=spectral_scale,
            es_fmax=es_fmax,
            omega_max=omega_max,
            learn_omega=learn_omega,
            share_input_proj=share_input_proj,
            alpha_min=alpha_min,
            alpha_tau=alpha_tau,
            active_threshold=active_threshold,
        )
        # ---- very lightweight gated residual ----
        # beta controls the overall residual scale
        self.beta = nn.Parameter(torch.tensor(0.1))
        # single scalar gate per sample: g = sigmoid(w * mean(x_scalar) + b)
        self.gate_weight = nn.Parameter(torch.zeros(1))
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        x: [..., H] where H = hidden_irreps.dim

        returns: same shape, scalar slots modified by LibraKAN residual.

        """
        if self.libra is None or self.scalar_dim == 0:
            # no scalar channels -> identity
            return x
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.H)  # [B, H]
        scalar_mask = self.scalar_mask  # [H]
        x_scalar = x_flat[:, scalar_mask]  # [B, Ds]
        x_rest = x_flat[:, ~scalar_mask]  # [B, H-Ds]
        if x_scalar.numel() == 0:
            return x
        # Libra core
        y = self.libra(x_scalar)  # [B, Ds]
        # simple scalar gate per sample
        mean_act = x_scalar.mean(dim=-1, keepdim=True)  # [B, 1]
        gate = torch.sigmoid(self.gate_weight * mean_act + self.gate_bias)  # [B, 1]
        # gated residual on scalar channels
        x_scalar_out = x_scalar + self.beta * gate * y  # [B, Ds]
        # reassemble
        out_flat = torch.empty_like(x_flat)
        out_flat[:, scalar_mask] = x_scalar_out
        out_flat[:, ~scalar_mask] = x_rest
        out = out_flat.reshape(orig_shape)
        return out


