from __future__ import annotations

import math

import torch

from torch import nn

from typing import Optional

from .nufft_es import NUFFTES

from .shrinkage import GeneralizedShrinkage


class GeneralLibraKAN(nn.Module):
    """
    GeneralLibraKAN: f(x) = W2[ (1-alpha)*phi(W1 x) + alpha * T^H S_{lambda,p}( T (W1 x) ) ]
    - Local branch: W1 -> activation -> mid
    - Spectral branch: NUFFT-ES (T), generalized shrinkage, exact adjoint (T^H)
    - Fusion via convex weight alpha in (0,1)
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden: Optional[int] = None,
            p: float = 1.0,
            lam: float = 1e-2,
            trainable_lambda: bool = True,
            activation: str = "gelu",
            F: Optional[int] = None,  # nominal dictionary width
            spectral_scale: float = 1.0,  # scales the omega range
            es_beta: float = 6.0,  # kept for API compatibility (unused here)
            es_fmax: Optional[float] = None,  # override for max frequency
            omega_max: Optional[float] = None,  # deprecated; prefer es_fmax or spectral_scale
            **kwargs,  # ignore unknown extras to remain robust
    ):
        super().__init__()
        # Hidden width fallback: typical MLP choice
        h = hidden or max(in_dim, out_dim)
        # Linear projections
        self.lin1 = nn.Linear(in_dim, h, bias=True)
        self.lin2 = nn.Linear(h, out_dim, bias=True)
        # Activation resolution (fallback to GELU)
        if hasattr(torch.nn.functional, activation):
            self.act = getattr(torch.nn.functional, activation)
        else:
            self.act = torch.nn.functional.gelu
        # NUFFT-ES and shrinkage
        self.nufft = NUFFTES()
        self.shrink = GeneralizedShrinkage(lam=lam, p=p, trainable_lambda=trainable_lambda)
        # Alpha in (0,1)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        # --------- Frequency dictionary Ω ---------
        # Determine number of spectral samples (K) and range.
        K = int(F) if F is not None else h
        # Choose max frequency: explicit es_fmax wins; else scale * pi; else omega_max legacy.
        if es_fmax is not None:
            wmax = float(es_fmax)
        elif omega_max is not None:
            wmax = float(omega_max)
        else:
            wmax = float(spectral_scale) * math.pi
        # Non-uniform option can be added later; linspace is a safe default here.
        omega = torch.linspace(-wmax, wmax, steps=max(1, K))
        # Register as learnable parameter to realize a "learnable non-uniform dictionary"
        self.omega = nn.Parameter(omega)  # learnable non-uniform frequency dictionary Ω
        # Keep attributes for introspection / future kernels
        self._spectral_cfg = dict(F=K, spectral_scale=spectral_scale, es_beta=es_beta, es_fmax=es_fmax)
        # NOTE: All steps are differentiable; gradients flow to omega via NUFFT.
        # This matches the paper's "learned non-uniform frequency dictionary Ω and shrinkage S_{lambda,p}" (Sec. 3.2).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local path
        z = self.lin1(x)  # [..., H]
        local = self.act(z)  # local branch
        # Spectral path: NUFFT -> shrink -> adjoint (all differentiable)
        y = self.nufft(z, self.omega)  # [..., K] complex
        y = self.shrink(y)  # [..., K] complex
        z_spec = self.nufft.adjoint(y, self.omega, z.shape[-1]).real  # [..., H] real
        # Fuse
        alpha = torch.sigmoid(self.alpha_logit)  # scalar in (0,1)
        fused = (1.0 - alpha) * local + alpha * z_spec
        out = self.lin2(fused)  # [..., out_dim]
        return out