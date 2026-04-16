"""
Given (source, destination), predicts the route by:
  1. Extracting subgraph around src/dst
  2. Running the GNN to classify edges as on-path/off-path
  3. Reconstructing the route by following highest-probability edges
  4. Converting back to graph coordinates for the API

"""

import json
import math
import pickle
import sys
from pathlib import Path
from functools import lru_cache

import networkx as nx
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gnn"))

from gnn_route_classifier import RouteClassifierGNN, reconstruct_route

# Reuse constants from training
HIGHWAY_CLASSES = {
    "motorway": 0, "motorway_link": 1, "trunk": 2, "trunk_link": 3,
    "primary": 4, "primary_link": 5, "secondary": 6, "secondary_link": 7,
    "tertiary": 8, "tertiary_link": 9, "residential": 10, "living_street": 11,
    "service": 12, "unclassified": 13,
}
HIGHWAY_SPEEDS = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 70, "primary_link": 40, "secondary": 60, "secondary_link": 30,
    "tertiary": 50, "tertiary_link": 25, "residential": 30, "living_street": 10,
    "service": 20, "unclassified": 40,
}


def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


_model_cache = {}

def load_classifier_model(model_path: Path):
    """Load trained route classifier model (cached)."""
    key = str(model_path)
    if key in _model_cache:
        return _model_cache[key]

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = RouteClassifierGNN(
        node_features=config["node_features"],
        edge_features=config["edge_features"],
        hidden_dim=config["hidden_dim"],
        num_steps=config.get("num_steps", config.get("num_layers", 8)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded route classifier: F1={checkpoint.get('best_f1', '?'):.3f}, "
          f"trained {checkpoint.get('trained_at', '?')}")

    _model_cache[key] = model
    return model



def extract_subgraph(G, source, dest, max_nodes=1500):
    """
    Extract subgraph using elliptical corridor (matches training).
    Node included if dist(node, src) + dist(node, dst) <= threshold.
    """
    src_lat = G.nodes[source].get("y", 0)
    src_lng = G.nodes[source].get("x", 0)
    dst_lat = G.nodes[dest].get("y", 0)
    dst_lng = G.nodes[dest].get("x", 0)
    
    direct_dist = haversine_m(src_lat, src_lng, dst_lat, dst_lng)
    
    # Adaptive scale factor matching training
    if direct_dist < 2000:
        scale = 2.5
    elif direct_dist < 6000:
        scale = 2.0
    else:
        scale = 1.7
    
    threshold = max(direct_dist * scale, 2000)
    
    subgraph_nodes = set()
    for n in G.nodes():
        lat = G.nodes[n].get("y", 0)
        lng = G.nodes[n].get("x", 0)
        d_src = haversine_m(lat, lng, src_lat, src_lng)
        d_dst = haversine_m(lat, lng, dst_lat, dst_lng)
        if d_src + d_dst <= threshold:
            subgraph_nodes.add(n)
    
    subgraph_nodes.add(source)
    subgraph_nodes.add(dest)
    
    try:
        sp = nx.shortest_path(G, source, dest, weight="length")
        subgraph_nodes.update(sp)
    except nx.NetworkXNoPath:
        pass
    
    if len(subgraph_nodes) > max_nodes:
        mid_lat = (src_lat + dst_lat) / 2
        mid_lng = (src_lng + dst_lng) / 2
        scored = sorted(
            (((G.nodes[n].get("y", 0) - mid_lat)**2 + (G.nodes[n].get("x", 0) - mid_lng)**2), n)
            for n in subgraph_nodes
        )
        subgraph_nodes = {n for _, n in scored[:max_nodes]}
        subgraph_nodes.add(source)
        subgraph_nodes.add(dest)
        try:
            sp = nx.shortest_path(G, source, dest, weight="length")
            subgraph_nodes.update(sp)
        except nx.NetworkXNoPath:
            pass
    
    return G.subgraph(subgraph_nodes).copy()


def get_edge_traffic(G, u, v, data, by_osmid, by_uv):
    osmid = data.get("osmid")
    if osmid is not None:
        ids = osmid if isinstance(osmid, list) else [osmid]
        for oid in ids:
            if int(oid) in by_osmid:
                return by_osmid[int(oid)]
    return by_uv.get((u, v))


def build_inference_data(subgraph, source, dest, by_osmid, by_uv):
    """Build PyG Data for inference (no labels needed)."""
    G = subgraph
    nodes = sorted(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}

    if source not in node_id_map or dest not in node_id_map:
        return None, None, None

    src_lat = G.nodes[source].get("y", 0)
    src_lng = G.nodes[source].get("x", 0)
    dst_lat = G.nodes[dest].get("y", 0)
    dst_lng = G.nodes[dest].get("x", 0)

    # Node features
    node_feats = []
    for n in nodes:
        lat = G.nodes[n].get("y", 0)
        lng = G.nodes[n].get("x", 0)
        degree = G.degree(n)
        dist_src = haversine_m(lat, lng, src_lat, src_lng) / 10000.0
        dist_dst = haversine_m(lat, lng, dst_lat, dst_lng) / 10000.0
        is_src = 1.0 if n == source else 0.0
        is_dst = 1.0 if n == dest else 0.0

        road_classes = []
        for _, _, d in G.edges(n, data=True):
            hw = d.get("highway", "unclassified")
            if isinstance(hw, list):
                hw = hw[0]
            road_classes.append(HIGHWAY_CLASSES.get(hw, 13))
        avg_class = sum(road_classes) / max(len(road_classes), 1) / 13.0
        has_signal = 1.0 if G.nodes[n].get("highway") == "traffic_signals" else 0.0

        node_feats.append([
            min(degree / 6.0, 1.0), lat / 90.0, lng / 180.0,
            dist_src, dist_dst, is_src, is_dst, avg_class, has_signal,
        ])

    # Edge features
    edge_src_list, edge_dst_list, edge_feats = [], [], []
    edge_keys = []  # (u, v, key) for mapping back

    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in node_id_map or v not in node_id_map:
            continue

        edge_src_list.append(node_id_map[u])
        edge_dst_list.append(node_id_map[v])
        edge_keys.append((u, v, key))

        length = data.get("length", 50)
        length_log = math.log1p(length) / 10.0
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        hw_class = HIGHWAY_CLASSES.get(hw, 13) / 13.0

        maxspeed = data.get("maxspeed")
        speed_limit = 50
        if maxspeed:
            try:
                speed_limit = float(str(maxspeed).split()[0])
            except (ValueError, TypeError):
                pass
        speed_limit_norm = min(speed_limit / 130.0, 1.0)

        lanes = data.get("lanes", 1)
        if isinstance(lanes, list):
            try: lanes = int(lanes[0])
            except: lanes = 1
        try: lanes = int(lanes)
        except: lanes = 1
        lanes_norm = min(lanes / 4.0, 1.0)
        oneway = 1.0 if data.get("oneway") else 0.0

        tp = get_edge_traffic(G, u, v, data, by_osmid, by_uv)
        if tp:
            current_speed = (tp.get("current_speed_kph") or 0) / 130.0
            free_flow = (tp.get("free_flow_speed_kph") or 0) / 130.0
            jam = (tp.get("jam_factor") or 0) / 10.0
            cong_ratio = tp.get("congestion_ratio") or 0
            closure = 1.0 if tp.get("road_closure") else 0.0
        else:
            current_speed = speed_limit_norm
            free_flow = speed_limit_norm
            jam = 0.0
            cong_ratio = 1.0
            closure = 0.0

        bridge = 1.0 if data.get("bridge") else 0.0
        tunnel = 1.0 if data.get("tunnel") else 0.0

        # Road hierarchy (must match training)
        is_highway = 1.0 if hw in ("motorway", "motorway_link", "trunk", "trunk_link") else 0.0
        is_arterial = 1.0 if hw in ("primary", "primary_link", "secondary", "secondary_link") else 0.0
        is_collector = 1.0 if hw in ("tertiary", "tertiary_link") else 0.0
        is_local = 1.0 if hw in ("residential", "living_street", "service", "unclassified") else 0.0

        road_quality = 0.0
        if hw in ("motorway", "trunk"): road_quality = 1.0
        elif hw in ("motorway_link", "trunk_link"): road_quality = 0.85
        elif hw in ("primary",): road_quality = 0.7
        elif hw in ("primary_link", "secondary"): road_quality = 0.55
        elif hw in ("secondary_link", "tertiary"): road_quality = 0.4
        elif hw in ("tertiary_link",): road_quality = 0.3
        elif hw in ("unclassified",): road_quality = 0.2
        elif hw in ("residential",): road_quality = 0.1
        else: road_quality = 0.05

        actual_speed = current_speed * 130.0
        if actual_speed > 0:
            travel_time_norm = (length / max(actual_speed / 3.6, 0.5)) / 120.0
        else:
            travel_time_norm = (length / max(speed_limit / 3.6, 0.5)) / 120.0

        edge_feats.append([
            length_log, hw_class, speed_limit_norm, lanes_norm, oneway,
            current_speed, free_flow, jam, cong_ratio, closure, bridge, tunnel,
            is_highway, is_arterial, is_collector, is_local,
            road_quality, min(travel_time_norm, 2.0),
        ])

    if not edge_src_list:
        return None, None, None

    import torch
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor([edge_src_list, edge_dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.source_idx = node_id_map[source]
    data.dest_idx = node_id_map[dest]

    return data, node_id_map, edge_keys


# ── Main Inference Function ───────────────────────────────────────────────────

def classifier_route(G, source_node, dest_node, model_path, snapshot=None):
    """
    Run the route classifier GNN to predict a route.

    Args:
        G: NetworkX graph
        source_node: source node ID in G
        dest_node: destination node ID in G
        model_path: Path to route_classifier.pt
        snapshot: traffic snapshot dict (optional, for edge features)

    Returns:
        route_nodes: list of node IDs forming the route
        edge_probs: dict {(u,v): probability} for all subgraph edges
    """
    model = load_classifier_model(model_path)

    # Build traffic lookup
    by_osmid, by_uv = {}, {}
    if snapshot:
        for feat in snapshot.get("features", []):
            props = feat.get("properties", {})
            osmid = props.get("osmid")
            if osmid is not None:
                ids = osmid if isinstance(osmid, list) else [osmid]
                for oid in ids:
                    by_osmid[int(oid)] = props
            u, v = props.get("u"), props.get("v")
            if u is not None and v is not None:
                by_uv[(int(u), int(v))] = props

    # Extract subgraph
    subgraph = extract_subgraph(G, source_node, dest_node)

    # Build PyG data
    data, node_id_map, edge_keys = build_inference_data(
        subgraph, source_node, dest_node, by_osmid, by_uv
    )
    if data is None:
        return None, None

    # Run model
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits)

    # Reconstruct route with traffic-blended weights
    route_indices = reconstruct_route(
        data.edge_index, probs, data.edge_attr, data.source_idx, data.dest_idx
    )

    # Map back to original node IDs
    idx_to_node = {i: n for n, i in node_id_map.items()}
    route_nodes = [idx_to_node[i] for i in route_indices if i in idx_to_node]

    # Edge probabilities for debugging/visualization
    edge_probs = {}
    for i, (u, v, key) in enumerate(edge_keys):
        edge_probs[(u, v)] = probs[i].item()

    return route_nodes, edge_probs