# mace/modules/updates.py
import torch.nn as nn
from .mixers import make_mixer

class NodeUpdate(nn.Module):
    """Node update mixer. Supports 'mlp' | 'libra'."""
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        kind = getattr(args, "node_mixer", "mlp")
        self.mix = make_mixer(kind, in_dim, out_dim, site="node", args=args)
    def forward(self, x):
        return self.mix(x)    