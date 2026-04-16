"""
Architecture:
  1. GraphSAGE convolutions aggregate neighborhood information at each node
  2. For each edge (u,v), we concatenate:
     - node embedding of u
     - node embedding of v
     - edge features (road length, type, traffic state, etc.)
  3. An MLP predicts travel_time from this concatenated vector

The model predicts travel_time per edge. At inference time, these predictions
become edge weights for Dijkstra shortest-path, which is directly comparable to
the traffic-aware routing baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class EdgeTravelTimeGNN(nn.Module):
    """
    GNN that predicts per-edge travel time.

    Forward pass:
      1. Node embeddings via stacked GraphSAGE layers
      2. Edge representation = [src_embed ‖ dst_embed ‖ edge_features]
      3. MLP → predicted travel_time (seconds)
    """

    def __init__(
        self,
        node_features: int = 5,
        edge_features: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_proj = nn.Linear(node_features, hidden_dim)

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))


        edge_input_dim = hidden_dim * 2 + edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: node features [num_nodes, node_features]
            edge_index: [2, num_edges]
            edge_attr: edge features [num_edges, edge_features]

        Returns:
            predicted travel times [num_edges]
        """
        h = self.node_proj(x)

        # graphsage message passing
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([
            h[src],      
            h[dst],       
            edge_attr,    
        ], dim=1)

        # predict travel time
        pred = self.edge_mlp(edge_repr).squeeze(-1)

        pred = F.softplus(pred)

        return pred
