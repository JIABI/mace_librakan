# mace/modules/mixers/kaf.py
# KAF readout block adapted for MACE: scalar-safe, AMP-friendly, DDP-safe
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Random Fourier Features ----------------
class RandomFourierFeatures(nn.Module):
    """
    RFF: x -> [cos(xW+b), sin(xW+b)] -> Linear -> R^D (D = input_dim by default)
    - Weight variance follows activation_expectation for stable scaling.
    - Combination is Xavier-initialized.
    """

    def __init__(
            self,
            input_dim: int,
            num_grids: int,
            dropout: float = 0.0,
            activation_expectation: float = 1.64,
            out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_grids = int(num_grids)
        self.dropout = nn.Dropout(float(dropout))
        out_dim = int(out_dim) if out_dim is not None else self.input_dim

        # var_w consistent with your reference
        var_w = 1.0 / (self.input_dim * float(activation_expectation))
        self.weight = nn.Parameter(torch.randn(self.input_dim, self.num_grids) * math.sqrt(var_w))
        self.bias = nn.Parameter(torch.empty(self.num_grids))
        nn.init.uniform_(self.bias, 0.0, 2.0 * math.pi)

        self.combination = nn.Linear(2 * self.num_grids, out_dim)
        nn.init.xavier_uniform_(self.combination.weight)
        if self.combination.bias is not None:
            fan_in = self.combination.in_features
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.combination.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        # keep dtype and device consistent with x (no .to() in forward to be DDP-safe)
        proj = x.matmul(self.weight) + self.bias  # (..., G)
        # amp-friendly trig
        c = torch.cos(proj)
        s = torch.sin(proj)
        ff = torch.cat([c, s], dim=-1)  # (..., 2G)
        ff = self.dropout(ff)
        return self.combination(ff)  # (..., D_out)


# ---------------- RFFActivation (residual spectral activation) ----------------
class RFFActivation(nn.Module):
    """
    y = base_scale * base_activation(x) + spline_scale * RFF(LN(x))
    - num_grids: Fourier grids for RFF
    - use_layernorm: LN over last dim (disabled if feature dim == 1)
    - base_activation: callable (e.g., F.gelu / F.silu)
    - activation_expectation: controls RFF weight variance
    """

    def __init__(
            self,
            num_grids: int = 9,
            dropout: float = 0.0,
            activation_expectation: float = 1.64,
            use_layernorm: bool = False,
            base_activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
            out_dim: Optional[int] = None,
    ):
        super().__init__()
        self._initialized = False
        self.num_grids = int(num_grids)
        self.dropout = float(dropout)
        self.activation_expectation = float(activation_expectation)
        self.use_layernorm = bool(use_layernorm)
        self.base_activation = base_activation
        self.out_dim = out_dim  # if None, defaults to input dim at init-time

        # to be created lazily when input dim is known
        self.layernorm: Optional[nn.LayerNorm] = None
        self.rff: Optional[RandomFourierFeatures] = None

        # learnable scales (AMP/FP16-safe, small init on spectral)
        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.spline_scale = nn.Parameter(torch.tensor(1e-2))

    def _lazy_build(self, input_dim: int, device: torch.device, dtype: torch.dtype):
        if self._initialized:
            return
        dim = int(input_dim)
        if self.use_layernorm and dim > 1:
            self.layernorm = nn.LayerNorm(dim, device=device)
        # RFF out_dim: keep same dim so downstream Linear sees consistent size
        rff_out = dim if self.out_dim is None else int(self.out_dim)
        self.rff = RandomFourierFeatures(
            input_dim=dim,
            num_grids=self.num_grids,
            dropout=self.dropout,
            activation_expectation=self.activation_expectation,
            out_dim=rff_out,
        ).to(device=device, dtype=dtype)
        # ensure scales live on same device/dtype as input
        self.base_scale.data = self.base_scale.data.to(device=device, dtype=dtype)
        self.spline_scale.data = self.spline_scale.data.to(device=device, dtype=dtype)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # build on first use with current device/dtype (DDP/AMP friendly)
        self._lazy_build(x.size(-1), x.device, x.dtype)
        x_norm = self.layernorm(x) if self.layernorm is not None else x
        y_base = self.base_activation(x)
        y_spec = self.rff(x_norm) if self.rff is not None else 0.0
        return self.base_scale * y_base + self.spline_scale * y_spec


# ---------------- KAFBlock for readout ----------------
class KAFBlock(nn.Module):
    """
    KAF readout block for MACE:
      in_dim --(in_proj)--> H --(RFFActivation)--> H --(out_proj)--> out_dim

    Notes:
    - Designed for readout (scalar channels). For in_dim==1, LN is auto-disabled inside RFFActivation.
    - AMP-friendly; no device transfers in forward.
    - Hidden defaults to max(in_dim, out_dim) for stable capacity on scalar inputs.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            *,
            F: int = 128,  # == num_grids
            dropout: float = 0.0,
            use_layernorm: bool = False,
            base_activation: str = "gelu",  # "gelu" | "silu" | "relu"
            activation_expectation: float = 1.64,
            hidden: Optional[int] = None,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden) if hidden is not None else max(self.in_dim, self.out_dim)

        act = base_activation.lower()
        act_fn = torch.nn.functional.gelu
        if act == "silu":
            act_fn = torch.nn.functional.silu
        elif act == "relu":
            act_fn = torch.nn.functional.relu

        # projection to hidden
        self.in_proj = nn.Identity() if self.hidden == self.in_dim else nn.Linear(self.in_dim, self.hidden)

        # spectral activation (residual) at hidden width
        self.rff_act = RFFActivation(
            num_grids=F,
            dropout=dropout,
            activation_expectation=activation_expectation,
            use_layernorm=use_layernorm,
            base_activation=act_fn,
            out_dim=self.hidden,
        )

        # final projection to out_dim (usually 1)
        self.out_proj = nn.Linear(self.hidden, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.rff_act(h)
        y = self.out_proj(h)
        return y