# =========================
# Adapter: KANMixer
# =========================
# This thin adapter makes MultKAN compatible with the readout mixer factory
# expected by NonLinearReadoutBlock and model_script_utils.
# It accepts (in_dim, out_dim, **kwargs), filters kwargs by the underlying
# constructor signature, and falls back to a minimal constructor if needed.

import inspect
import torch
from torch import nn

try:
    # If MultKAN is defined in this file, import is not needed; keep safe.
    MultKAN  # noqa: F401
except NameError:
    from .kan import MultKAN  # just in case someone relocates identifiers


def _filter_kwargs_for_ctor(ctor, kwargs: dict) -> dict:
    """Keep only kwargs that appear in ctor's signature; ignore the rest."""
    sig = inspect.signature(ctor)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


class KANMixer(nn.Module):
    """
    Thin wrapper around MultKAN.
    Required signature: KANMixer(in_dim: int, out_dim: int, **kwargs)
    Forward: y = core(x), where x has shape [..., in_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        # Try a "rich" construction first, but filter kwargs by ctor signature.
        ctor = MultKAN
        filtered = _filter_kwargs_for_ctor(ctor, kwargs)

        try:
            self.core = ctor(in_dim=in_dim, out_dim=out_dim, **filtered)
        except TypeError:
            # Fallback: minimal constructor to avoid crashing on unknown args
            self.core = ctor(in_dim=in_dim, out_dim=out_dim)

        # Optional: remember dims for debugging
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


# Export the adapter for external imports
__all__ = list(set(globals().get("__all__", [])) | {"KANMixer"})