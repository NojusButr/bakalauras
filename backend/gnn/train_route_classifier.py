"""
Generates training data from traffic snapshots using multiple routing strategies:
  - Traffic-weighted Dijkstra (primary)
  - Distance-weighted Dijkstra (teaches structure awareness)
  - Simulated congestion routes (teaches crisis adaptation)

Architecture: LSTM-GN gated message passing  + GCN backbone
Training: Dynamic weighted BCE loss with F1-score tracking
"""

import json
import math
import pickle
import random
import sys
from pathlib import Path
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gnn"))

from gnn_route_classifier import RouteClassifierGNN



CITY = "vilnius"
GRAPH_PATH = PROJECT_ROOT / "cities" / CITY / "graph.pkl"
SNAPSHOTS_DIR = PROJECT_ROOT / "cities" / CITY / "snapshots"
MODEL_SAVE_DIR = PROJECT_ROOT / "cities" / CITY / "models"

ROUTES_PER_SNAPSHOT = 100
MAX_SUBGRAPH_NODES = 1500
MIN_PATH_EDGES = 5

EPOCHS = 150
BATCH_SIZE = 16
LR = 0.001
HIDDEN_DIM = 64
NUM_STEPS = 8
INITIAL_POS_WEIGHT = 20.0
FINAL_POS_WEIGHT = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIGHWAY_SPEEDS = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 70, "primary_link": 40, "secondary": 60, "secondary_link": 30,
    "tertiary": 50, "tertiary_link": 25, "residential": 30, "living_street": 10,
    "service": 20, "unclassified": 40,
}
HIGHWAY_CLASSES = {
    "motorway": 0, "motorway_link": 1, "trunk": 2, "trunk_link": 3,
    "primary": 4, "primary_link": 5, "secondary": 6, "secondary_link": 7,
    "tertiary": 8, "tertiary_link": 9, "residential": 10, "living_street": 11,
    "service": 12, "unclassified": 13,
}

DEGRADED_COPIES = 3
DEGRADE_LEVELS = [0.25, 0.5, 0.75]

# ── Loading ───────────────────────────────────────────────────────────────────

def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_snapshots():
    snapshots = []
    for f in sorted(SNAPSHOTS_DIR.glob("*.json")):
        if "degraded" in f.stem or "simulation" in f.stem:
            continue
        try:
            with open(f) as fh:
                snap = json.load(fh)
            if snap.get("features"):
                snapshots.append((f.stem, snap))
        except Exception:
            continue
    print(f"Loaded {len(snapshots)} snapshots")
    return snapshots


def build_traffic_lookup(snapshot):
    by_osmid, by_uv = {}, {}
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
    return by_osmid, by_uv


def get_edge_traffic(G, u, v, data, by_osmid, by_uv):
    osmid = data.get("osmid")
    if osmid is not None:
        ids = osmid if isinstance(osmid, list) else [osmid]
        for oid in ids:
            if int(oid) in by_osmid:
                return by_osmid[int(oid)]
    return by_uv.get((u, v))


# ── Subgraph Extraction ──────────────────────────────────────────────────────

def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_subgraph(G, source, dest, max_nodes=MAX_SUBGRAPH_NODES):
    """
    Extract subgraph using an elliptical corridor between source and destination.
    
    The ellipse has foci at source and destination. A node is included if:
        dist(node, src) + dist(node, dst) <= threshold
    
    Threshold = direct_distance * scale_factor, where scale_factor adapts to route length:
        - Short routes (<2km): factor=2.5 (wide exploration relative to distance)
        - Medium routes (2-6km): factor=2.0
        - Long routes (6-15km): factor=1.7 (includes arterials and highways)

    """
    src_lat = G.nodes[source].get("y", 0)
    src_lng = G.nodes[source].get("x", 0)
    dst_lat = G.nodes[dest].get("y", 0)
    dst_lng = G.nodes[dest].get("x", 0)
    
    direct_dist = haversine_m(src_lat, src_lng, dst_lat, dst_lng)
    

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
    
    try:
        tp = nx.shortest_path(G, source, dest, weight="_tw")
        subgraph_nodes.update(tp)
    except (nx.NetworkXNoPath, nx.NetworkXError):
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
        # Re-add critical paths
        try:
            sp = nx.shortest_path(G, source, dest, weight="length")
            subgraph_nodes.update(sp)
        except nx.NetworkXNoPath:
            pass
    
    return G.subgraph(subgraph_nodes).copy()


# ── Feature Engineering ───────────────────────────────────────────────────────

def degrade_traffic(by_osmid, by_uv, degrade_pct):
    """
    Randomly removes traffic info from a percentage of edges.
    Returns degraded copies of lookup dicts.
    """
    if degrade_pct <= 0:
        return by_osmid, by_uv

    by_osmid_deg = dict(by_osmid)
    by_uv_deg = dict(by_uv)

    osmids = list(by_osmid_deg.keys())
    n_remove = int(len(osmids) * degrade_pct)
    remove_ids = set(random.sample(osmids, n_remove))

    for oid in remove_ids:
        by_osmid_deg.pop(oid, None)

    uv_keys = list(by_uv_deg.keys())
    n_remove_uv = int(len(uv_keys) * degrade_pct)
    remove_uv = set(random.sample(uv_keys, n_remove_uv))

    for k in remove_uv:
        by_uv_deg.pop(k, None)

    return by_osmid_deg, by_uv_deg


def build_pyg_data(subgraph, source, dest, route_edges, by_osmid, by_uv):
    G = subgraph
    nodes = sorted(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}
    if source not in node_id_map or dest not in node_id_map:
        return None

    src_lat = G.nodes[source].get("y", 0)
    src_lng = G.nodes[source].get("x", 0)
    dst_lat = G.nodes[dest].get("y", 0)
    dst_lng = G.nodes[dest].get("x", 0)

    # Node features: [degree, lat, lng, dist_src, dist_dst, is_src, is_dst, road_class, signal]
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
            if isinstance(hw, list): hw = hw[0]
            road_classes.append(HIGHWAY_CLASSES.get(hw, 13))
        avg_class = sum(road_classes) / max(len(road_classes), 1) / 13.0
        has_signal = 1.0 if G.nodes[n].get("highway") == "traffic_signals" else 0.0

        node_feats.append([
            min(degree / 6.0, 1.0), lat / 90.0, lng / 180.0,
            dist_src, dist_dst, is_src, is_dst, avg_class, has_signal,
        ])

    edge_src, edge_dst, edge_feats, edge_labels = [], [], [], []

    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in node_id_map or v not in node_id_map:
            continue
        edge_src.append(node_id_map[u])
        edge_dst.append(node_id_map[v])

        length = data.get("length", 50)
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list): hw = hw[0]

        maxspeed = data.get("maxspeed")
        speed_limit = 50
        if maxspeed:
            try: speed_limit = float(str(maxspeed).split()[0])
            except: pass

        lanes = data.get("lanes", 1)
        if isinstance(lanes, list):
            try: lanes = int(lanes[0])
            except: lanes = 1
        try: lanes = int(lanes)
        except: lanes = 1

        tp = get_edge_traffic(G, u, v, data, by_osmid, by_uv)
        if tp:
            cur_speed = (tp.get("current_speed_kph") or 0) / 130.0
            free_flow = (tp.get("free_flow_speed_kph") or 0) / 130.0
            jam = (tp.get("jam_factor") or 0) / 10.0
            cong = tp.get("congestion_ratio") or 0
            closure = 1.0 if tp.get("road_closure") else 0.0
        else:
            cur_speed = min(speed_limit / 130.0, 1.0)
            free_flow = cur_speed
            jam = 0.0
            cong = 1.0
            closure = 0.0

        # Explicit road hierarchy 
        is_highway = 1.0 if hw in ("motorway", "motorway_link", "trunk", "trunk_link") else 0.0
        is_arterial = 1.0 if hw in ("primary", "primary_link", "secondary", "secondary_link") else 0.0
        is_collector = 1.0 if hw in ("tertiary", "tertiary_link") else 0.0
        is_local = 1.0 if hw in ("residential", "living_street", "service", "unclassified") else 0.0

        # Road quality score: higher = faster road (strong signal for the model)
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

        # Travel time estimate (normalized to ~minutes)
        actual_speed = cur_speed * 130.0  # denormalize
        if actual_speed > 0:
            travel_time_norm = (length / max(actual_speed / 3.6, 0.5)) / 120.0  # normalize: 2min = 1.0
        else:
            travel_time_norm = (length / max(speed_limit / 3.6, 0.5)) / 120.0

        edge_feats.append([
            math.log1p(length) / 10.0,        # 0: length
            HIGHWAY_CLASSES.get(hw, 13) / 13.0, # 1: highway class
            min(speed_limit / 130.0, 1.0),      # 2: speed limit
            min(lanes / 4.0, 1.0),              # 3: lanes
            1.0 if data.get("oneway") else 0.0, # 4: oneway
            cur_speed,                           # 5: current speed
            free_flow,                           # 6: free flow
            jam,                                 # 7: jam factor
            cong,                                # 8: congestion ratio
            closure,                             # 9: road closure
            1.0 if data.get("bridge") else 0.0,  # 10: bridge
            1.0 if data.get("tunnel") else 0.0,  # 11: tunnel
            is_highway,                          # 12: highway flag
            is_arterial,                         # 13: arterial flag
            is_collector,                        # 14: collector flag
            is_local,                            # 15: local road flag
            road_quality,                        # 16: road quality score
            min(travel_time_norm, 2.0),          # 17: travel time estimate
        ])

        is_on = 1.0 if (u, v) in route_edges or (u, v, key) in route_edges else 0.0
        edge_labels.append(is_on)

    if not edge_src:
        return None

    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    y = torch.tensor(edge_labels, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.source_idx = node_id_map[source]
    data.dest_idx = node_id_map[dest]
    data.num_positive = int(y.sum().item())
    data.num_edges_total = len(edge_labels)
    return data


# ── Training Data Generation ──────────────────────────────────────────────────

def generate_training_data(G, snapshots, routes_per_snap=ROUTES_PER_SNAPSHOT):
    all_data = []
    nodes = list(G.nodes())
    bridge_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("bridge")]
    
    node_positions = {}
    for n in nodes:
        node_positions[n] = (G.nodes[n].get("y", 0), G.nodes[n].get("x", 0))

    DISTANCE_BUCKETS = [
        (500, 2000, "short"),
        (2000, 5000, "medium"),
        (5000, 8000, "long"),
        (8000, 12000, "very_long"),
        (12000, 25000, "cross_city"),
    ]

    for snap_name, snapshot in snapshots:
        print(f"  Processing {snap_name}...")
        by_osmid, by_uv = build_traffic_lookup(snapshot)

        for u, v, key, data in G.edges(keys=True, data=True):
            length = data.get("length", 50)
            tp = get_edge_traffic(G, u, v, data, by_osmid, by_uv)
            if tp:
                speed = tp.get("current_speed_kph")
                if speed and speed > 0:
                    G[u][v][key]["_tw"] = length / max(speed / 3.6, 0.5)
                    continue
            hw = data.get("highway", "unclassified")
            if isinstance(hw, list): hw = hw[0]
            speed = HIGHWAY_SPEEDS.get(hw, 30)
            G[u][v][key]["_tw"] = length / max(speed / 3.6, 0.5)

        routes_per_bucket = routes_per_snap // len(DISTANCE_BUCKETS)
        bucket_counts = {b[2]: 0 for b in DISTANCE_BUCKETS}

        generated = 0
        attempts = 0
        max_attempts = routes_per_snap * 15

        while generated < routes_per_snap and attempts < max_attempts:
            attempts += 1

            # Find which bucket needs samples most
            min_bucket = min(bucket_counts, key=bucket_counts.get)
            min_data = next(b for b in DISTANCE_BUCKETS if b[2] == min_bucket)

            if bucket_counts[min_bucket] < routes_per_bucket:
                src = random.choice(nodes)
                s_lat, s_lng = node_positions[src]
                candidates = []
                for _ in range(50):
                    cand = random.choice(nodes)
                    if cand == src:
                        continue
                    c_lat, c_lng = node_positions[cand]
                    d = haversine_m(s_lat, s_lng, c_lat, c_lng)
                    if min_data[0] <= d < min_data[1]:
                        candidates.append(cand)
                if not candidates:
                    continue
                dst = random.choice(candidates)
            else:
                src = random.choice(nodes)
                dst = random.choice(nodes)
                if src == dst:
                    continue

            s_lat, s_lng = node_positions[src]
            d_lat, d_lng = node_positions[dst]
            direct_dist = haversine_m(s_lat, s_lng, d_lat, d_lng)

            bucket_name = None
            for min_d, max_d, name in DISTANCE_BUCKETS:
                if min_d <= direct_dist < max_d:
                    bucket_name = name
                    break
            if bucket_name is None or bucket_counts.get(bucket_name, 0) >= routes_per_bucket + 5:
                continue

            r = random.random()
            try:
                if r < 0.40:
                    route_nodes = nx.shortest_path(G, src, dst, weight="_tw")

                elif r < 0.55:
                    # no road closures, just heavy traffic in an area
                    G_congested = G.copy()
                    center = random.choice(nodes)
                    c_lat, c_lng = node_positions[center]
                    congestion_radius = random.uniform(500, 2000)
                    speed_factor = random.uniform(0.1, 0.4)  # 10-40% of normal speed

                    for u, v, k, d in G_congested.edges(keys=True, data=True):
                        u_lat = G.nodes[u].get("y", 0)
                        u_lng = G.nodes[u].get("x", 0)
                        dist_to_center = haversine_m(c_lat, c_lng, u_lat, u_lng)
                        if dist_to_center <= congestion_radius:
                            # Slow down this edge
                            length = d.get("length", 50)
                            hw = d.get("highway", "unclassified")
                            if isinstance(hw, list): hw = hw[0]
                            base_speed = HIGHWAY_SPEEDS.get(hw, 30)
                            slow_speed = base_speed * speed_factor
                            G_congested[u][v][k]["_tw"] = length / max(slow_speed / 3.6, 0.5)
                            # Mark in traffic lookup for feature building
                            by_uv[(u, v)] = {
                                "current_speed_kph": slow_speed,
                                "free_flow_speed_kph": base_speed,
                                "jam_factor": round((1 - speed_factor) * 10, 1),
                                "congestion_ratio": speed_factor,
                                "road_closure": False,
                            }

                    route_nodes = nx.shortest_path(G_congested, src, dst, weight="_tw")

                elif r < 0.70:
                    # Mild crisis: close edges + propagate congestion to neighbors
                    G_crisis = G.copy()
                    n_remove = random.randint(1, 3)
                    center = random.choice(nodes)
                    c_lat, c_lng = node_positions[center]
                    nearby = [
                        (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                        if haversine_m(c_lat, c_lng, G.nodes[u].get("y", 0), G.nodes[u].get("x", 0)) < 400
                    ]
                    removed = []
                    if nearby:
                        for u, v, k in random.sample(nearby, min(n_remove, len(nearby))):
                            if G_crisis.has_edge(u, v, k):
                                G_crisis.remove_edge(u, v, k)
                                removed.append((u, v, k))

                    # Propagate congestion around closed roads (like simulation does)
                    for u, v, k in removed:
                        by_uv[(u, v)] = {
                            "current_speed_kph": 0, "free_flow_speed_kph": 50,
                            "jam_factor": 10.0, "congestion_ratio": 0.0,
                            "road_closure": True,
                        }
                    # Slow down roads within 300-800m of closures
                    for u_r, v_r, k_r in removed:
                        r_lat = G.nodes[u_r].get("y", 0)
                        r_lng = G.nodes[u_r].get("x", 0)
                        for u2, v2, k2, d2 in G_crisis.edges(keys=True, data=True):
                            u2_lat = G.nodes[u2].get("y", 0)
                            u2_lng = G.nodes[u2].get("x", 0)
                            dist = haversine_m(r_lat, r_lng, u2_lat, u2_lng)
                            if dist < 800 and (u2, v2) not in by_uv:
                                decay = max(0.2, 1.0 - dist / 800)
                                hw2 = d2.get("highway", "unclassified")
                                if isinstance(hw2, list): hw2 = hw2[0]
                                base_speed = HIGHWAY_SPEEDS.get(hw2, 30)
                                slow_speed = base_speed * (1 - decay * 0.7)
                                length = d2.get("length", 50)
                                G_crisis[u2][v2][k2]["_tw"] = length / max(slow_speed / 3.6, 0.5)
                                by_uv[(u2, v2)] = {
                                    "current_speed_kph": slow_speed,
                                    "free_flow_speed_kph": base_speed,
                                    "jam_factor": round(decay * 8, 1),
                                    "congestion_ratio": round(1 - decay * 0.7, 2),
                                    "road_closure": False,
                                }

                    route_nodes = nx.shortest_path(G_crisis, src, dst, weight="_tw")

                else:
                    # Extreme crisis: bridges + major roads + heavy surrounding congestion
                    G_crisis = G.copy()
                    removed = []
                    if bridge_edges:
                        n_bridges = random.randint(1, min(4, len(bridge_edges)))
                        for u, v, k in random.sample(bridge_edges, n_bridges):
                            if G_crisis.has_edge(u, v, k):
                                G_crisis.remove_edge(u, v, k)
                                removed.append((u, v, k))
                    major_types = {"primary", "secondary", "trunk", "motorway"}
                    major_edges = [
                        (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                        if (d.get("highway") if isinstance(d.get("highway"), str) else (d.get("highway") or [""])[0]) in major_types
                    ]
                    if major_edges:
                        n_major = random.randint(2, min(5, len(major_edges)))
                        for u, v, k in random.sample(major_edges, n_major):
                            if G_crisis.has_edge(u, v, k):
                                G_crisis.remove_edge(u, v, k)
                                removed.append((u, v, k))

                    # Mark closures and propagate congestion
                    for u, v, k in removed:
                        by_uv[(u, v)] = {
                            "current_speed_kph": 0, "free_flow_speed_kph": 50,
                            "jam_factor": 10.0, "congestion_ratio": 0.0,
                            "road_closure": True,
                        }
                    for u_r, v_r, k_r in removed:
                        r_lat = G.nodes[u_r].get("y", 0)
                        r_lng = G.nodes[u_r].get("x", 0)
                        for u2, v2, k2, d2 in G_crisis.edges(keys=True, data=True):
                            u2_lat = G.nodes[u2].get("y", 0)
                            u2_lng = G.nodes[u2].get("x", 0)
                            dist = haversine_m(r_lat, r_lng, u2_lat, u2_lng)
                            if dist < 1200 and (u2, v2) not in by_uv:
                                decay = max(0.15, 1.0 - dist / 1200)
                                hw2 = d2.get("highway", "unclassified")
                                if isinstance(hw2, list): hw2 = hw2[0]
                                base_speed = HIGHWAY_SPEEDS.get(hw2, 30)
                                slow_speed = base_speed * (1 - decay * 0.8)
                                length = d2.get("length", 50)
                                G_crisis[u2][v2][k2]["_tw"] = length / max(slow_speed / 3.6, 0.5)
                                by_uv[(u2, v2)] = {
                                    "current_speed_kph": slow_speed,
                                    "free_flow_speed_kph": base_speed,
                                    "jam_factor": round(decay * 9, 1),
                                    "congestion_ratio": round(1 - decay * 0.8, 2),
                                    "road_closure": False,
                                }

                    route_nodes = nx.shortest_path(G_crisis, src, dst, weight="_tw")
            except nx.NetworkXNoPath:
                continue

            if len(route_nodes) < MIN_PATH_EDGES:
                continue

            route_edges = set()
            for i in range(len(route_nodes) - 1):
                route_edges.add((route_nodes[i], route_nodes[i + 1]))

            subgraph = extract_subgraph(G, src, dst)

            # --- CLEAN SAMPLE (full traffic) ---
            pyg_data = build_pyg_data(subgraph, src, dst, route_edges, by_osmid, by_uv)
            if pyg_data is not None and pyg_data.num_positive >= MIN_PATH_EDGES:
                all_data.append(pyg_data)
                generated += 1
                if bucket_name:
                    bucket_counts[bucket_name] += 1

            # --- DEGRADED SAMPLES ---
            for level in DEGRADE_LEVELS[:DEGRADED_COPIES]:
                by_osmid_deg, by_uv_deg = degrade_traffic(by_osmid, by_uv, level)

                pyg_data_deg = build_pyg_data(
                    subgraph,
                    src,
                    dst,
                    route_edges,          
                    by_osmid_deg,        
                    by_uv_deg
                )

                if pyg_data_deg is not None and pyg_data_deg.num_positive >= MIN_PATH_EDGES:
                    all_data.append(pyg_data_deg)

        print(f"    Generated {generated} samples — buckets: {dict(bucket_counts)}")

    print(f"Total training samples: {len(all_data)}")
    return all_data


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(data_list, val_split=0.15):
    random.shuffle(data_list)
    split = int(len(data_list) * (1 - val_split))
    train_data = data_list[:split]
    val_data = data_list[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Device: {DEVICE}")

    sample = train_data[0]
    node_features = sample.x.shape[1]
    edge_features = sample.edge_attr.shape[1]
    print(f"Node features: {node_features}, Edge features: {edge_features}")

    model = RouteClassifierGNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=HIDDEN_DIM,
        num_steps=NUM_STEPS,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = 0.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        progress = epoch / max(EPOCHS - 1, 1)
        pos_weight = INITIAL_POS_WEIGHT - (INITIAL_POS_WEIGHT - FINAL_POS_WEIGHT) * progress

        model.train()
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            batch = Batch.from_data_list(train_data[i:i + BATCH_SIZE]).to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            weight = torch.where(batch.y == 1, torch.tensor(pos_weight, device=DEVICE), torch.tensor(1.0, device=DEVICE))
            loss = F.binary_cross_entropy_with_logits(logits, batch.y, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Validation
        model.eval()
        val_tp, val_fp, val_fn = 0, 0, 0
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                batch = Batch.from_data_list(val_data[i:i + BATCH_SIZE]).to(DEVICE)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                weight = torch.where(batch.y == 1, torch.tensor(pos_weight, device=DEVICE), torch.tensor(1.0, device=DEVICE))
                loss = F.binary_cross_entropy_with_logits(logits, batch.y, weight=weight)
                val_loss += loss.item()
                val_batches += 1
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_tp += ((preds == 1) & (batch.y == 1)).sum().item()
                val_fp += ((preds == 1) & (batch.y == 0)).sum().item()
                val_fn += ((preds == 0) & (batch.y == 1)).sum().item()

        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        avg_val = val_loss / max(val_batches, 1)

        history.append({
            "epoch": epoch, "train_loss": avg_loss, "val_loss": avg_val,
            "precision": precision, "recall": recall, "f1": f1,
            "pos_weight": pos_weight,
        })

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:3d} | loss {avg_loss:.4f} | val {avg_val:.4f} | "
                  f"P {precision:.3f} R {recall:.3f} F1 {f1:.3f} | "
                  f"pw {pos_weight:.1f} | best {best_val_f1:.3f}")

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_DIR / "route_classifier.pt"
    torch.save({
        "model_state": best_state,
        "config": {
            "node_features": node_features,
            "edge_features": edge_features,
            "hidden_dim": HIDDEN_DIM,
            "num_steps": NUM_STEPS,
        },
        "best_f1": best_val_f1,
        "history": history,
        "trained_at": datetime.now().isoformat(),
    }, model_path)
    print(f"\nSaved to {model_path} (best F1: {best_val_f1:.3f})")

    with open(MODEL_SAVE_DIR / "classifier_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def main():
    print("=" * 70)
    print("ROUTE CLASSIFIER — LSTM-GN Training")
    print("=" * 70)
    print(f"City: {CITY}")
    print(f"Architecture: {NUM_STEPS}-step LSTM-GN, hidden={HIDDEN_DIM}")
    print(f"Training: {ROUTES_PER_SNAPSHOT} routes/snapshot, {EPOCHS} epochs")
    print(f"Dynamic loss: pos_weight {INITIAL_POS_WEIGHT} → {FINAL_POS_WEIGHT}")
    print()

    G = load_graph()
    snapshots = load_snapshots()
    if not snapshots:
        print("No snapshots found!")
        return

    print("\n--- Generating training data ---")
    data_list = generate_training_data(G, snapshots)
    if len(data_list) < 20:
        print(f"Only {len(data_list)} samples — need more.")
        return

    total_edges = sum(d.num_edges_total for d in data_list)
    total_pos = sum(d.num_positive for d in data_list)
    print(f"\nClass balance: {total_pos}/{total_edges} positive ({total_pos/max(total_edges,1):.2%})")
    print(f"Avg subgraph: {np.mean([d.x.shape[0] for d in data_list]):.0f} nodes, "
          f"{np.mean([d.edge_index.shape[1] for d in data_list]):.0f} edges")

    print("\n--- Training ---")
    train_model(data_list)
    print("\nDone!")


if __name__ == "__main__":
    main()