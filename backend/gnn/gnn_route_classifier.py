"""

Given (source, destination), classifies every edge in a subgraph as
"on the optimal route" (1) or "not on the route" (0).

Key architectural choices:
  - GCN (SAGEConv) for spatial neighbor aggregation
  - LSTM-style gates between message passing steps 
  - Edge classifier MLP on concatenated node embeddings + edge features
  - Dijkstra on probability-weighted edges for route reconstruction
"""

import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class LSTMGraphBlock(nn.Module):
    """
    One LSTM-style message passing step

    Instead of simple h = GCN(h), this uses:
      forget_gate = sigmoid(W_f * [h_prev, x])
      input_gate  = sigmoid(W_i * [h_prev, x])
      candidate   = tanh(W_c * [h_prev, x])
      cell        = forget_gate * cell_prev + input_gate * candidate
      output_gate = sigmoid(W_o * [h_prev, x])
      h           = output_gate * tanh(cell)

    Where the GCN conv is applied within the gates for graph-aware gating.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GCN conv for message passing within each gate
        self.conv_f = SAGEConv(hidden_dim, hidden_dim)
        self.conv_i = SAGEConv(hidden_dim, hidden_dim)
        self.conv_c = SAGEConv(hidden_dim, hidden_dim)
        self.conv_o = SAGEConv(hidden_dim, hidden_dim)

        # Gate projections: [h_prev || x] → hidden
        self.gate_f = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_i = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_o = nn.Linear(hidden_dim * 2, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, cell, x, edge_index):
        """
        Args:
            h: hidden state [num_nodes, hidden_dim]
            cell: cell state [num_nodes, hidden_dim]
            x: input (encoded features) [num_nodes, hidden_dim]
            edge_index: [2, num_edges]

        Returns:
            h_new, cell_new
        """
        # Concatenate hidden state and input
        combined = torch.cat([h, x], dim=-1)

        # Forget gate: which info to discard from cell
        f = torch.sigmoid(self.gate_f(combined) + self.conv_f(h, edge_index))

        # Input gate: which new info to store
        i = torch.sigmoid(self.gate_i(combined) + self.conv_i(h, edge_index))

        # Candidate cell state
        c_tilde = torch.tanh(self.gate_c(combined) + self.conv_c(h, edge_index))

        # New cell state
        cell_new = f * cell + i * c_tilde

        # Output gate
        o = torch.sigmoid(self.gate_o(combined) + self.conv_o(h, edge_index))

        # New hidden state (with graph-aware message passing applied)
        h_new = o * torch.tanh(cell_new)
        h_new = self.norm(h_new)

        return h_new, cell_new


class RouteClassifierGNN(nn.Module):
    """
    Route classifier with LSTM-GN gated message passing.

    Architecture:
      1. Node encoder (MLP): raw features → hidden_dim
      2. M steps of LSTM-style graph message passing
      3. Edge classifier (MLP): [src_embed || dst_embed || edge_features] → logit
    """

    def __init__(
        self,
        node_features: int = 9,
        edge_features: int = 12,
        hidden_dim: int = 64,
        num_steps: int = 8,   
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # LSTM-style message passing blocks
        # Share weights across steps (recurrent) to reduce parameters
        self.lstm_block = LSTMGraphBlock(hidden_dim)

        # Decoder / Edge classifier (Paper 1: GN-decoder + classification)
        edge_input_dim = hidden_dim * 2 + edge_features
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: [num_nodes, node_features]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_features]
        Returns:
            edge_logits: [num_edges]
        """
        encoded = self.node_encoder(x)

        h = encoded
        cell = torch.zeros_like(h)

        for step in range(self.num_steps):
            h, cell = self.lstm_block(h, cell, encoded, edge_index)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=1)
        logits = self.edge_classifier(edge_repr).squeeze(-1)
        return logits


def reconstruct_route(
    edge_index: torch.Tensor,
    edge_probs: torch.Tensor,
    edge_attr: torch.Tensor,
    source_idx: int,
    dest_idx: int,
    max_steps: int = 500,
    traffic_blend: float = 0.6,
) -> list[int]:
    """
    Reconstruct route using Dijkstra on blended weights.

    Weight = (1 - blend) * (-log(prob)) + blend * travel_time_estimate

    This combines GNN confidence with actual traffic conditions:
    - High probability + fast road = very low weight (preferred)
    - High probability + slow road = moderate weight
    - Low probability + fast road = moderate weight
    - Low probability + slow road = very high weight (avoided)
    
    edge_attr columns: [length_log, hw_class, speed_limit, lanes, oneway,
                        current_speed, free_flow, jam, cong_ratio, closure, bridge, tunnel]
    """
    adj: dict[int, list[tuple[int, float]]] = {}
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()
    probs = edge_probs.tolist()
    eps = 1e-6

    for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
        if s not in adj:
            adj[s] = []

        # Probability-based weight (lower = more likely on route)
        prob_weight = -torch.log(torch.tensor(max(probs[i], eps))).item()

        # Traffic-based weight from edge features
        closure = edge_attr[i][9].item()
        jam = edge_attr[i][7].item() * 10.0

        if closure > 0.5 or jam > 9.5:
            travel_weight_norm = 50.0 
        else:
            # Use pre-computed travel time (index 17, already normalized)
            travel_weight_norm = edge_attr[i][17].item() if edge_attr.shape[1] > 17 else 0.5
            
            # Penalize low-quality roads (index 16: road_quality, higher=better)
            if edge_attr.shape[1] > 16:
                road_quality = edge_attr[i][16].item()
                # Invert: low quality roads get penalty
                quality_penalty = max(0, (0.5 - road_quality)) * 0.5 
                travel_weight_norm += quality_penalty

        # Blend
        w = (1 - traffic_blend) * prob_weight + traffic_blend * travel_weight_norm
        adj[s].append((d, w))

    # Dijkstra
    dist = {source_idx: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, source_idx)]

    while heap:
        d, u = heapq.heappop(heap)
        if u == dest_idx:
            break
        if d > dist.get(u, float('inf')):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dest_idx not in prev and source_idx != dest_idx:
        return [source_idx]

    route = []
    current = dest_idx
    while current != source_idx:
        route.append(current)
        current = prev.get(current, source_idx)
        if len(route) > max_steps:
            break
    route.append(source_idx)
    route.reverse()
    return route