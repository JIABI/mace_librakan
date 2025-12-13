from __future__ import annotations
import torch
from torch import nn
from typing import Optional, List

try:
    from e3nn import o3  # for parsing hidden_irreps when provided
except Exception:
    o3 = None

from .librakan import GeneralLibraKAN


def _scalar_mask_from_irreps_ir_mul(hidden_irreps: str) -> List[bool]:
    """
    Build a boolean mask for l=0 (scalar) slots under the 'ir_mul' layout:
    the feature vector is a concatenation over irreps; each irrep contributes
    (mul * dim) contiguous positions. For l=0, dim=1, so we add 'mul' True's.
    For l>0, dim=2l+1, so we add mul * dim False's.
    """
    if o3 is None:
        raise RuntimeError("e3nn.o3 is required to parse hidden_irreps.")
    irs = o3.Irreps(hidden_irreps)
    mask: List[bool] = []
    for mul, (l, _p) in ((ir.mul, (ir.ir.l, ir.ir.p)) for ir in irs):
        dim = 2 * l + 1
        if l == 0:
            mask.extend([True] * mul)  # each scalar contributes 1 slot
        else:
            mask.extend([False] * (mul * dim))  # all vector/tensor slots
    return mask


class NodeScalarLibraKAN(nn.Module):
    """
    Equivariance-friendly node mixer:
    - Extract only scalar (l=0) channels from the node feature vector (ir_mul layout).
    - Apply a lightweight GeneralLibraKAN on the scalar sub-vector.
    - Write back to the same scalar slots; non-scalars are untouched.

    Important design choices vs. the previous version:
    - Residual is gated: out_scalar = x_scalar + beta * gate * Libra(x_scalar).
    - gate is a learnable scalar in (0, 1), initialized small (almost identity).
    - Optional LayerNorm on the scalar slice (pre+post) to stabilize scale.
    - Libra core uses stronger shrinkage and smaller F by default, so it behaves
      like a gentle regularizer instead of a second strong readout.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            *,
            hidden_irreps: Optional[str] = None,  # pass args.hidden_irreps
            scalar_mask: Optional[torch.Tensor] = None,  # optional override: [S] bool

            # --- Node-LibraKAN defaults: conservative, regularizer-style ---
            p: float = 1.0,
            lam: float = 1e-2,  # stronger shrinkage than readout
            trainable_lambda: bool = True,
            activation: str = "gelu",
            F: Optional[int] = None,  # if None -> small value inferred below
            spectral_scale: float = 0.6,
            es_fmax: Optional[float] = None,
            alpha_min: float = 0.05,  # allow spectral branch to switch off
            alpha_tau: float = 1.2,
            learn_omega: bool = True,  # fixed Ω for stability
            use_layernorm: bool = False,  # internal LN in GeneralLibraKAN
            dropout: float = 0.0,
            local_kind: str = "act",  # tiny local branch
            local_layers: Optional[list] = None,

            # --- Residual & gating ---
            residual: str = "add",  # "add" or "replace"
            beta: float = 0.05,  # overall residual strength
            use_scalar_layernorm: bool = True,
            gate_init: float = -2.0,  # sigmoid(-2) ≈ 0.12, i.e. small at start
            enabled : bool = True,
        ):
        super().__init__()
        assert in_dim == out_dim, "NodeScalarLibraKAN expects square (in_dim == out_dim)."
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        self.beta = float(beta)
        self.use_scalar_layernorm = use_scalar_layernorm
        self.enabled = bool(enabled)

        # Build scalar mask
        if scalar_mask is not None:
            mask = scalar_mask.bool().view(-1)
        elif hidden_irreps is not None:
            mask_list = _scalar_mask_from_irreps_ir_mul(hidden_irreps)
            mask = torch.tensor(mask_list, dtype=torch.bool)
        else:
            raise ValueError("Provide either scalar_mask or hidden_irreps to build a scalar-only mask.")

        scalar_dim = int(mask.sum().item())
        if scalar_dim <= 0:
            raise ValueError("No l=0 scalar slots found in hidden_irreps; cannot build scalar-only mixer.")

        self.register_buffer("scalar_mask", mask, persistent=False)
        self.scalar_dim = scalar_dim

        # Optional LN just on scalar slice
        if self.use_scalar_layernorm:
            self.ln_pre = nn.LayerNorm(scalar_dim)
            self.ln_post = nn.LayerNorm(scalar_dim)
        else:
            self.ln_pre = None
            self.ln_post = None

        # If F not given, keep it small relative to scalar_dim
        if F is None:
            F = min(64, max(16, 2 * scalar_dim))

        # Lightweight LibraKAN operating only on the scalar slice
        self.core = GeneralLibraKAN(
            in_dim=scalar_dim,
            out_dim=scalar_dim,
            hidden=None,  # keep narrow
            p=p,
            lam=lam,
            trainable_lambda=trainable_lambda,
            activation=activation,
            F=F,
            spectral_scale=spectral_scale,
            es_fmax=es_fmax,
            alpha_min=alpha_min,
            alpha_tau=alpha_tau,
            learn_omega=learn_omega,
            use_layernorm=use_layernorm,
            dropout=dropout,
            local_kind=local_kind,
            local_layers=local_layers,
        )

        # Learnable gate to softly turn on the Libra correction
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def set_enabled(self, flag: bool) -> None:
        self.enabled = bool(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., S] where S == in_dim == out_dim. Assumes ir_mul layout ordering.
        """
        if not self.enabled or self.beta==0.0:
            return x
        S = x.shape[-1]
        if S != self.in_dim:
            raise RuntimeError(f"Last dim {S} != expected {self.in_dim}.")
        mask = self.scalar_mask

        # Extract scalar part
        xs = x[..., mask]  # [..., scalar_dim]
        ys = self.core(xs)

        out = x.clone()
        if self.residual == "replace":
            out[..., mask] = ys
        else:
            delta = ys - xs
            out[..., mask] = xs + self.beta * delta

        return out
