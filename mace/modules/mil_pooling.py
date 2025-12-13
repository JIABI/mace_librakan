###########################################################################################
# PAN Pooling Implementation (Physics-Aware Neighbourhood Pooling)
# Implements the scalar gating mechanism described in Methods 4.2 of the paper.
###########################################################################################

import torch
import torch.nn as nn

class PANPooling(nn.Module):
    """
    Physics-Aware Neighbourhood (PAN) Pooling.
    
    Computes a geometry-conditioned weight for each edge to modulate scalar messages 
    before aggregation.
    
    Formula: a_ij = sigma( MLP(h_ij, r_ij) )
    
    Args:
        edge_feat_dim (int): Dimension of invariant edge features (radial features).
        hidden_dim (int): Hidden dimension for the gating MLP.
    """
    def __init__(self, edge_feat_dim: int, hidden_dim: int = 16):
        super().__init__()
        # Input: edge_feats (invariants) + edge_length (scalar distance)
        input_dim = edge_feat_dim + 1 
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Ensures weights are in (0, 1) acting as a gate
        )

    def forward(self, edge_feats: torch.Tensor, edge_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_feats: [n_edges, edge_feat_dim] (Invariant radial features)
            edge_lengths: [n_edges, 1] (Interatomic distances)
            
        Returns:
            weights: [n_edges, 1] Attention/Gating weights.
        """
        # Concatenate invariant features with explicit distance
        # Ensure shapes match for concatenation
        if edge_lengths.dim() == 1:
            edge_lengths = edge_lengths.unsqueeze(-1)
            
        x = torch.cat([edge_feats, edge_lengths], dim=-1)
        weights = self.net(x)
        return weights

