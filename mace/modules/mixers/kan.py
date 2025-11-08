# pylint: disable=all

"""

MultKAN-based readout head for replacing the MLP head in MACE.

This module wraps `pykan.MultKAN` as a plug-and-play readout:

- Accepts node embeddings with shapes:

    (a) [B, N, C]  (batched graphs, per-node features)

    (b) [num_nodes, C] with `batch` vector for graph indices

- Pooling: "sum" | "mean" | "none"

- Falls back to a simple MLP if pykan is unavailable.

Typical usage:

    from .multkan import MultKAN  # acts as a readout head

    head = MultKAN(in_dim, out_dim, hidden=(256,), pooling="sum")

Notes:

- Symbolic branch is disabled; `speed()` mode is used for stable/fast inference.

- No autosave / checkpointing during training from this wrapper.

"""

import math
from typing import Iterable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to use official pykan if available
try:
    from kan import MultKAN as PyMultKAN  # pip install pykan
    _HAS_PYKAN = True
except Exception:
    PyMultKAN = None
    _HAS_PYKAN = False

# Try torch_scatter for fast segment ops (optional)

try:
    from torch_scatter import scatter_sum, scatter_mean  # type: ignore
    _HAS_SCATTER = True
except Exception:
    _HAS_SCATTER = False


def _infer_tuple(x: Optional[Union[int, Iterable[int]]]) -> Tuple[int, ...]:
    if x is None:
        return tuple()
    if isinstance(x, int):
        return (x,)
    return tuple(x)


def _pool_nodes(
        x: torch.Tensor,
        pooling: str,
        batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pool node features to graph features.
    Args:
        x: Node features, shape either [B, N, C] or [num_nodes, C].
        pooling: "sum" | "mean" | "none".
        batch: Optional [num_nodes] graph index vector (only used when x is [num_nodes, C]).
    Returns:
        If pooling != "none": graph-level tensor [B, C] or [num_graphs, C].
        If pooling == "none": return node-level tensor unchanged.
    """

    if pooling == "none":
        return x

    if x.dim() == 3:
        # [B, N, C]
        if pooling == "sum":
            return x.sum(dim=1)
        elif pooling == "mean":
            return x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    if x.dim() == 2:
        # [num_nodes, C] with batch vector
        if batch is None:
            raise ValueError("batch indices are required when x is [num_nodes, C].")
        if _HAS_SCATTER:
            if pooling == "sum":
                return scatter_sum(x, batch, dim=0)
            elif pooling == "mean":
                return scatter_mean(x, batch, dim=0)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
        else:
            # Fallback without torch_scatter
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            out = x.new_zeros((num_graphs, x.size(-1)))
            if pooling == "sum":
                for g in range(num_graphs):
                    mask = (batch == g)
                    if torch.any(mask):
                        out[g] = x[mask].sum(dim=0)
            elif pooling == "mean":
                for g in range(num_graphs):
                    mask = (batch == g)
                    if torch.any(mask):
                        out[g] = x[mask].mean(dim=0)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            return out
    raise ValueError(f"Unsupported tensor rank for pooling: x.dim()={x.dim()}")

class _MLPFallback(nn.Module):
    """
    Lightweight MLP fallback when `pykan` is unavailable.
    Drop-in compatible with the MultKAN readout interface.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden: Tuple[int, ...],
            base_activation: str = "gelu",
            dropout: float = 0.0,
    ):
        super().__init__()
        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "identity": nn.Identity,
        }.get(base_activation.lower(), nn.GELU)
        dims = (in_dim,) + hidden + (out_dim,)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_layer())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        # Kaiming initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultKAN(nn.Module):
    """
    KAN readout head that wraps `pykan.MultKAN` (or MLP fallback).
    Args:
        in_dim: Input feature dimension C.
        out_dim: Output dimension.
        hidden: Optional hidden sizes for KAN mixer path, e.g., (256,) or (256, 256).
        pooling: "sum" | "mean" | "none".
        grid: Number of grid intervals (pykan spline parameter).
        k: Spline order (pykan).
        base_fun: Base activation inside KAN ("silu", "identity", "zero").
        dropout: Dropout applied only in MLP fallback; ignored for pykan.
        use_symbolic: Keep False for speed/stability (disables symbolic branch).
        speed_mode: If True, call `.speed()` to disable caches & symbolic.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden: Optional[Iterable[int]] = (256,),
            pooling: str = "sum",
            grid: int = 5,
            k: int = 3,
            base_fun: str = "silu",
            dropout: float = 0.0,
            use_symbolic: bool = False,
            speed_mode: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = _infer_tuple(hidden)
        self.pooling = pooling.lower()
        if _HAS_PYKAN:
            width = [self.in_dim] + list(self.hidden) + [self.out_dim]
            # Create a single MultKAN with width covering the whole readout MLP
            self.kan = (PyMultKAN(width=width,
                grid=grid,
                k=k,
                base_fun=base_fun,
                symbolic_enabled=bool(use_symbolic),
                affine_trainable=False,
                grid_eps=0.02,
                grid_range=[-1, 1],
                sp_trainable=True,
                sb_trainable=True,
                seed=1,
                save_act=False,  # no mid-cache during training
                sparse_init=False,
                auto_save=False,  # disable autosave inside readout
                first_init=True,
                ckpt_path="./_kan_readout_ckpt",
                state_id=0,
                round=0,
                device="cpu",
            ))
            if speed_mode:
                # disable symbolic/caches; rely on numeric branch only
                self.kan.speed(compile=False)
            self.impl = "pykan"
        else:
            # Fallback to a standard MLP with similar interface
            self.kan = _MLPFallback(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                hidden=self.hidden,
                base_activation=base_fun,
                dropout=dropout,
            )
            self.impl = "mlp_fallback"
    def forward(
            self,
            x: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward.
        Inputs:
            x:
              - [B, N, C] (batched, per-node features), or
              - [num_nodes, C] with `batch` indices (graph IDs).
            batch:
              - Optional [num_nodes] vector of graph indices (only when x is [num_nodes, C]).
        Returns:
            If pooling != "none": [B, out_dim] (or [num_graphs, out_dim]).
            If pooling == "none":
                - For [B, N, C] -> [B, N, out_dim]
                - For [num_nodes, C] -> [num_nodes, out_dim]
        """
        if x.dim() == 3:
            B, N, C = x.shape
            assert C == self.in_dim, f"Expected last dim {self.in_dim}, got {C}"
            x_flat = x.reshape(B * N, C)
            out_flat = self.kan(x_flat)
            out = out_flat.reshape(B, N, self.out_dim)
            return _pool_nodes(out, pooling=self.pooling, batch=None)
        if x.dim() == 2:
            num_nodes, C = x.shape
            assert C == self.in_dim, f"Expected last dim {self.in_dim}, got {C}"
            out = self.kan(x)  # [num_nodes, out_dim]
            return _pool_nodes(out, pooling=self.pooling, batch=batch)
        raise ValueError(
            f"Unsupported input rank {x.dim()}; expected [B, N, C] or [num_nodes, C]."
        )

    # Expose a small helper for clarity

    @property
    def uses_pykan(self) -> bool:
        return self.impl == "pykan"


