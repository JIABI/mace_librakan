# =========================

# Adapter: KAFMixer

# =========================

# This thin adapter makes KAFBlock compatible with the readout mixer factory.

# It accepts (in_dim, out_dim, **kwargs), filters kwargs, and falls back to

# a minimal constructor to avoid TypeErrors if signatures diverge.

import inspect

import torch

from torch import nn

try:

    KAFBlock  # noqa: F401

except NameError:

    from .kaf import KAFBlock  # just in case


def _filter_kwargs_for_ctor(ctor, kwargs: dict) -> dict:
    """Keep only kwargs that appear in ctor's signature; ignore the rest."""

    sig = inspect.signature(ctor)

    allowed = set(sig.parameters.keys())

    return {k: v for k, v in kwargs.items() if k in allowed}


class KAFMixer(nn.Module):
    """

    Thin wrapper around KAFBlock.

    Required signature: KAFMixer(in_dim: int, out_dim: int, **kwargs)

    Forward: y = core(x), where x has shape [..., in_dim]

    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):

        super().__init__()

        ctor = KAFBlock

        filtered = _filter_kwargs_for_ctor(ctor, kwargs)

        try:

            self.core = ctor(in_dim=in_dim, out_dim=out_dim, **filtered)

        except TypeError:

            # Minimal fallback to guarantee construction succeeds

            self.core = ctor(in_dim=in_dim, out_dim=out_dim)

        self.in_dim = int(in_dim)

        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.core(x)


__all__ = list(set(globals().get("__all__", [])) | {"KAFMixer"})

