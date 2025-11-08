# edge_librakan.py
# --------------------------------------------------------------------------------------
# ADDED: EdgeLibraKAN — drop-in replacement for edge MLPs inside interaction blocks.
# Shape: (..., in_dim) -> (..., out_dim)
# --------------------------------------------------------------------------------------

from __future__ import annotations
import torch
from torch import nn
from typing import Optional

from .librakan import GeneralLibraKAN

class EdgeLibraKAN(nn.Module):
    """
    EdgeLibraKAN wraps GeneralLibraKAN with sensible defaults for edge updates.
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
    ):
        super().__init__()
        self.core = GeneralLibraKAN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            p=p,
            lam=lam,
            trainable_lambda=trainable_lambda,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)
