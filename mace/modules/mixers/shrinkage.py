# shrinkage.py
# --------------------------------------------------------------------------------------
# ADDED: Generalized shrinkage S_{lambda,p} for complex/real tensors.
# For p=1 it reduces to soft-thresholding. For p in (0,1), we use a smooth p-shrinkage
# that preserves differentiability almost everywhere and is widely used in practice.
# --------------------------------------------------------------------------------------

from __future__ import annotations
import torch
from torch import nn
from typing import Optional

class GeneralizedShrinkage(nn.Module):
    """
    Elementwise generalized shrinkage:
        S_{lambda,p}(z) = z * max(0, 1 - (lambda / (|z| + eps))^{1-p})   for z != 0
                          0                                             for z == 0
    This equals soft-threshold when p=1 and becomes stronger for smaller p in (0,1).
    For complex z, the shrinkage is applied to magnitude and phase is preserved.
    """
    def __init__(self, lam: float = 1e-2, p: float = 1.0, eps: float = 1e-8, trainable_lambda: bool = False):
        super().__init__()
        if p <= 0.0 or p > 1.0:
            raise ValueError("p must be in (0,1].")
        self.eps = eps
        self.p = float(p)
        lam_t = torch.tensor(float(lam), dtype=torch.float32)
        if trainable_lambda:
            self.lam = nn.Parameter(lam_t)
        else:
            self.register_buffer("lam", lam_t)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        is_complex = torch.is_complex(z)
        mag = torch.abs(z)
        # factor in [0,1]
        if self.p == 1.0:
            # soft-threshold: max(|z|-lam, 0) * sign(z)
            scale = torch.clamp(mag - self.lam, min=0.0) / (mag + self.eps)
        else:
            one_minus = 1.0 - torch.pow(self.lam.clamp(min=0.0) / (mag + self.eps), 1.0 - self.p)
            scale = torch.clamp(one_minus, min=0.0, max=1.0)
        out = scale * z
        return out
