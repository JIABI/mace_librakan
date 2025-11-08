import torch.nn as nn
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from .kan import MultKAN
from .kaf import KAFBlock
from .edge_librakan import EdgeLibraKAN
from .node_librakan import NodeLibraKAN
from .librakan import GeneralLibraKAN

__all__ = [
    "make_mixer",
    "MultKAN",
    "KAFBlock",
    "GeneralLibraKAN",
    "EdgeLibraKAN",
    "NodeLibraKAN",
]


# ----------------------- helpers ----------------------- #

def _normalize_site(site: str) -> str:
    site_l = site.lower()
    if site_l not in {"readout", "edge", "node"}:
        raise ValueError(f"Unknown site={site!r}, expected one of {{'readout','edge','node'}}.")
    return site_l


def _get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name in {"gelu"}:
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name in {"relu"}:
        return nn.ReLU(inplace=False)
    if name in {"tanh"}:
        return nn.Tanh()
    if name in {"identity", "none"}:
        return nn.Identity()
    # 默认用 GELU
    return nn.GELU()


def _as_hidden(value: Optional[Union[int, Sequence[int]]], fallback: Optional[int] = None) -> List[int]:
    if value is None:
        return [] if fallback is None else [int(fallback)]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _maybe_layernorm(use_ln: bool, dim: int) -> nn.Module:
    return nn.LayerNorm(dim, eps=1e-6) if use_ln else nn.Identity()


def _build_mlp(
        in_dim: int,
        out_dim: int,
        *,
        hidden: Optional[Union[int, Sequence[int]]] = None,
        act: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = False,
) -> nn.Module:
    """Generic MLP fallback used for any site."""
    hs = _as_hidden(hidden)
    layers: List[nn.Module] = []
    last = in_dim
    act_mod = _get_activation(act)
    do = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()

    if len(hs) == 0:
        # linear-only, but保持接口一致
        layers.append(nn.Linear(last, out_dim, bias=True))
        return nn.Sequential(*layers)

    for h in hs:
        layers.extend([
            nn.Linear(last, h, bias=True),
            _maybe_layernorm(use_layernorm, h),
            act_mod,
            do,
        ])
        last = h
    layers.append(nn.Linear(last, out_dim, bias=True))
    return nn.Sequential(*layers)


# ----------------------- factory ----------------------- #

def make_mixer(kind: str, in_dim: int, out_dim: int, site: str, args) -> nn.Module:
    """
    Factory for feature mixers used inside MACE.

    site ∈ {"readout","edge","node"}
      - readout: mlp | libra | kan | kaf
      - edge/node: mlp | libra

    Args (from `args`) supported keys (all optional):
      * MLP (any site): {mlp_hidden, mlp_act, mlp_dropout, mlp_use_layernorm}
      * Libra (readout): see GeneralLibraKAN in librakan.py (expects the whole args)
      * Libra (edge/node): see EdgeLibraKAN / NodeLibraKAN
      * KAN (readout): {kan_hidden, kan_pooling, kan_grid, kan_k, kan_base_activation,
                        kan_dropout, kan_use_symbolic, kan_speed_mode}
      * KAF (readout): {kaf_F, kaf_dropout, kaf_use_layernorm, kaf_base_activation,
                        kaf_activation_expectation, kaf_hidden}
    """
    site_l = _normalize_site(site)
    kind_l = (kind or "").lower().strip()

    # ---------- MLP (all sites) ----------
    if kind_l in {"mlp", ""}:
        return _build_mlp(
            in_dim,
            out_dim,
            hidden=getattr(args, "mlp_hidden", None),
            act=getattr(args, "mlp_act", "gelu"),
            dropout=getattr(args, "mlp_dropout", 0.0),
            use_layernorm=getattr(args, "mlp_use_layernorm", False),
        )

    # ---------- Libra family ----------
    if kind_l == "libra":
        if site_l == "readout":
            # General (graph-head readout)
            return GeneralLibraKAN(in_dim, out_dim, args)
        elif site_l == "edge":
            return EdgeLibraKAN(in_dim, out_dim, args)
        elif site_l == "node":
            return NodeLibraKAN(in_dim, out_dim, args)
        else:
            # 理论上不会到这里（_normalize_site 已校验）
            raise ValueError(f"Unknown site for libra: {site}")

    # ---------- KAF (readout only) ----------
    if kind_l == "kaf":
        if site_l != "readout":
            raise ValueError(f"KAF is readout-only, got site={site_l}.")
        return KAFBlock(
            in_dim,
            out_dim,
            F=getattr(args, "kaf_F", 128),
            dropout=getattr(args, "kaf_dropout", 0.0),
            use_layernorm=getattr(args, "kaf_use_layernorm", False),
            base_activation=getattr(args, "kaf_base_activation", "gelu"),
            activation_expectation=getattr(args, "kaf_activation_expectation", 1.64),
            hidden=getattr(args, "kaf_hidden", None),
        )

    # ---------- KAN (readout only) ----------
    if kind_l == "kan":
        if site_l != "readout":
            raise ValueError(f"KAN is readout-only, got site={site_l}.")
        return MultKAN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=getattr(args, "kan_hidden", (getattr(args, "width", 256),)),
            pooling=getattr(args, "kan_pooling", "sum"),
            grid=getattr(args, "kan_grid", 5),
            k=getattr(args, "kan_k", 3),
            base_fun=getattr(args, "kan_base_activation", "silu"),
            dropout=getattr(args, "kan_dropout", 0.0),  # only used by any internal MLP fallback
            use_symbolic=getattr(args, "kan_use_symbolic", False),
            speed_mode=getattr(args, "kan_speed_mode", True),
        )

    raise ValueError(
        f"Unknown mixer kind={kind!r} for site={site_l!r}. "
        f"Allowed: readout -> {{'mlp','libra','kan','kaf'}}, edge/node -> {{'mlp','libra'}}."
    )