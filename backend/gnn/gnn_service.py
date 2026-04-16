"""
Uses data.edge_keys stored by data_pipeline to map predictions
back to NetworkX edges without re-iterating the graph.
"""

import pickle
from pathlib import Path
from typing import Optional

import torch
import networkx as nx

from data_pipeline import graph_to_pyg_data
from gnn_model import EdgeTravelTimeGNN


_model_cache: dict[str, tuple[EdgeTravelTimeGNN, dict]] = {}


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str | Path) -> tuple[EdgeTravelTimeGNN, dict]:
    """Load trained model from checkpoint. Cached after first load."""
    model_path = str(model_path)
    if model_path in _model_cache:
        return _model_cache[model_path]

    device = _get_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    config = ckpt["model_config"]

    model = EdgeTravelTimeGNN(
        node_features=config["node_features"],
        edge_features=config["edge_features"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model_cache[model_path] = (model, config)
    print(f"GNN model loaded from {model_path} (val_loss={ckpt.get('val_loss', '?'):.4f})")
    return model, config


@torch.no_grad()
def predict_travel_times(
    model: EdgeTravelTimeGNN,
    graph_path: str | Path,
    snapshot: Optional[dict] = None,
) -> dict[tuple[int, int, int], float]:
    """
    Run GNN inference on the graph with current traffic state.
    Returns {(u, v, key): predicted_travel_time_seconds} for every edge.
    Uses edge_keys stored during data pipeline construction for reliable mapping.
    """
    device = _get_device()

    # Build PyG data — this also stores edge_keys as a Python list
    data = graph_to_pyg_data(graph_path, snapshot_dict=snapshot)

    edge_keys = data.edge_keys

    # Move tensors to device for inference
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)

    #  inference
    pred = model(x, edge_index, edge_attr)
    pred = pred.cpu().numpy()

    # Map predictions to NetworkX edge keys
    assert len(pred) == len(edge_keys), (
        f"Prediction count {len(pred)} != edge_keys count {len(edge_keys)}"
    )

    weights = {}
    for i, (u, v, key) in enumerate(edge_keys):
        weights[(u, v, key)] = max(float(pred[i]), 0.1)  # min 0.1s

    return weights


def gnn_route_weights(
    G: nx.MultiDiGraph,
    graph_path: str | Path,
    model_path: str | Path,
    snapshot: Optional[dict] = None,
) -> dict[tuple[int, int, int], float]:
    """
    High-level function for routes.py integration.
    Loads model, runs inference, returns edge weights.
    """
    model, config = load_model(model_path)
    weights = predict_travel_times(model, graph_path, snapshot)
    return weights