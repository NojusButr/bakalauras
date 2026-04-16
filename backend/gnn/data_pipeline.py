"""
data_pipeline.py — Convert OSM graph + traffic snapshots into PyG Data objects.

For each snapshot, generates:
  - 1 clean sample (full traffic data → full travel times)
  - N degraded samples (partial traffic data → full travel times)
"""

import json
import math
import pickle
import random
import time
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data


def _haversine_m(lat1, lng1, lat2, lng2):
    """Distance in meters between two lat/lng points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

HIGHWAY_ENCODING = {
    "motorway": 0, "motorway_link": 1, "trunk": 2, "trunk_link": 3,
    "primary": 4, "primary_link": 5, "secondary": 6, "secondary_link": 7,
    "tertiary": 8, "tertiary_link": 9, "residential": 10, "living_street": 11,
    "service": 12, "unclassified": 13,
}
NUM_HIGHWAY_TYPES = len(HIGHWAY_ENCODING)

HIGHWAY_DEFAULT_SPEED_KPH = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 70, "primary_link": 40, "secondary": 60, "secondary_link": 30,
    "tertiary": 50, "tertiary_link": 25, "residential": 30, "living_street": 10,
    "service": 20, "unclassified": 40,
}
DEFAULT_SPEED_KPH = 30


def _encode_highway(highway) -> int:
    if isinstance(highway, list):
        highway = highway[0]
    return HIGHWAY_ENCODING.get(highway, NUM_HIGHWAY_TYPES - 1)


def _get_speed_limit(data: dict) -> float:
    maxspeed = data.get("maxspeed")
    if maxspeed:
        try:
            return float(str(maxspeed).split()[0])
        except (ValueError, TypeError):
            pass
    highway = data.get("highway")
    if isinstance(highway, list):
        highway = highway[0]
    return HIGHWAY_DEFAULT_SPEED_KPH.get(highway, DEFAULT_SPEED_KPH)


def graph_to_pyg_data(
    graph_path: str | Path,
    snapshot_path: Optional[str | Path] = None,
    snapshot_dict: Optional[dict] = None,
    degrade_pct: Optional[float] = None,
    risk_events: Optional[list[dict]] = None,
) -> Data:
    """
    Convert graph + snapshot into PyG Data.
    
    If degrade_pct is set (0-1), randomly removes that fraction of traffic data
    from INPUT features, but keeps FULL data for target travel times.
    
    If risk_events is set, adds crisis penalties to TARGET travel times for
    affected edges, but INPUT features remain normal.
    
    risk_events format: [{"center": [lat, lng], "radius_m": 500, "penalty": 5.0}, ...]
    penalty is a multiplier on travel time (5.0 = 5x slower)
    """
    graph_path = Path(graph_path)
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    traffic_by_osmid: dict[int, dict] = {}
    if snapshot_dict is not None:
        snapshot = snapshot_dict
    elif snapshot_path is not None:
        with open(snapshot_path, "r") as f:
            snapshot = json.load(f)
    else:
        snapshot = None

    if snapshot:
        for feat in snapshot.get("features", []):
            props = feat.get("properties", {})
            speed = props.get("current_speed_kph")
            if speed is None:
                continue
            osmid = props.get("osmid")
            if osmid is None:
                continue
            if isinstance(osmid, list):
                for oid in osmid:
                    traffic_by_osmid[int(oid)] = props
            else:
                traffic_by_osmid[int(osmid)] = props

    # determine which osmids to degrade
    degraded_osmids = set()
    if degrade_pct is not None and degrade_pct > 0 and traffic_by_osmid:
        all_osmids = sorted(traffic_by_osmid.keys()) 
        n_remove = int(len(all_osmids) * degrade_pct)
        rng = random.Random(int(degrade_pct * 1000) + len(all_osmids))
        degraded_osmids = set(rng.sample(all_osmids, n_remove))

    nodes = list(G.nodes())
    node_id_to_idx = {nid: i for i, nid in enumerate(nodes)}
    num_nodes = len(nodes)

    lats = np.array([G.nodes[n].get("y", 0) for n in nodes])
    lngs = np.array([G.nodes[n].get("x", 0) for n in nodes])
    lat_mean, lat_std = lats.mean(), max(lats.std(), 1e-6)
    lng_mean, lng_std = lngs.mean(), max(lngs.std(), 1e-6)

    node_features = np.zeros((num_nodes, 5), dtype=np.float32)
    for i, nid in enumerate(nodes):
        ndata = G.nodes[nid]
        node_features[i, 0] = G.degree(nid) / 10.0
        hw_values = []
        for _, _, edata in G.edges(nid, data=True):
            hw_values.append(_encode_highway(edata.get("highway", "unclassified")))
        node_features[i, 1] = (np.mean(hw_values) / NUM_HIGHWAY_TYPES) if hw_values else 0.5
        node_features[i, 2] = 1.0 if ndata.get("highway") == "traffic_signals" else 0.0
        node_features[i, 3] = (ndata.get("y", 0) - lat_mean) / lat_std
        node_features[i, 4] = (ndata.get("x", 0) - lng_mean) / lng_std

    edge_src = []
    edge_dst = []
    edge_features_list = []
    edge_targets = []
    edge_keys_list = []

    for u, v, key, edata in G.edges(keys=True, data=True):
        if u not in node_id_to_idx or v not in node_id_to_idx:
            continue

        length = edata.get("length", 1.0)
        hw_enc = _encode_highway(edata.get("highway", "unclassified"))
        speed_limit = _get_speed_limit(edata)

        lanes = edata.get("lanes")
        if lanes:
            try:
                lanes = float(str(lanes).split(";")[0]) if isinstance(lanes, str) else float(lanes)
            except (ValueError, TypeError):
                lanes = 1.0
        else:
            lanes = 1.0
        lanes = min(lanes, 6.0)

        oneway = 1.0 if edata.get("oneway", False) else 0.0
        bridge = edata.get("bridge")
        is_bridge = 1.0 if bridge and bridge != "no" else 0.0
        tunnel = edata.get("tunnel")
        is_tunnel = 1.0 if tunnel and tunnel != "no" else 0.0

        # get full traffic data for target
        osmid = edata.get("osmid")
        full_speed = 0.0
        full_ff = 0.0
        full_jam = 0.0
        full_cong = 1.0
        full_closure = 0.0
        has_traffic = False

        if osmid is not None:
            lookup_ids = osmid if isinstance(osmid, list) else [osmid]
            for oid in lookup_ids:
                if int(oid) in traffic_by_osmid:
                    tp = traffic_by_osmid[int(oid)]
                    full_speed = tp.get("current_speed_kph") or 0.0
                    full_ff = tp.get("free_flow_speed_kph") or 0.0
                    full_jam = tp.get("jam_factor") or 0.0
                    full_cong = tp.get("congestion_ratio") if tp.get("congestion_ratio") is not None else 1.0
                    full_closure = 1.0 if tp.get("road_closure") else 0.0
                    has_traffic = True
                    break

        # check if this edge should be degraded
        is_degraded = False
        if osmid is not None and degraded_osmids:
            lookup_ids = osmid if isinstance(osmid, list) else [osmid]
            is_degraded = any(int(oid) in degraded_osmids for oid in lookup_ids)

        # input features: degraded edges get zero traffic
        if is_degraded:
            in_speed, in_ff, in_jam, in_cong, in_closure = 0.0, 0.0, 0.0, 1.0, 0.0
        elif has_traffic:
            in_speed, in_ff, in_jam, in_cong, in_closure = full_speed, full_ff, full_jam, full_cong, full_closure
        else:
            in_speed, in_ff, in_jam, in_cong, in_closure = 0.0, 0.0, 0.0, 1.0, 0.0

        feat = [
            math.log1p(length) / 10.0,
            hw_enc / NUM_HIGHWAY_TYPES,
            speed_limit / 130.0,
            lanes / 6.0,
            oneway,
            in_speed / 130.0,
            in_ff / 130.0,
            in_jam / 10.0,
            in_cong,
            in_closure,
            is_bridge,
            is_tunnel,
        ]


        eff_speed = full_speed if (has_traffic and full_speed > 0) else speed_limit
        travel_time = length / max(eff_speed / 3.6, 0.5)


        if risk_events:
            u_lat = G.nodes[u].get("y", 0)
            u_lng = G.nodes[u].get("x", 0)
            for evt in risk_events:
                e_lat, e_lng = evt["center"]
                r = evt.get("radius_m", 500)
                dist = _haversine_m(u_lat, u_lng, e_lat, e_lng)
                if dist <= r:
                    # Inside event radius: apply full penalty
                    travel_time *= evt.get("penalty", 5.0)
                    break
                elif dist <= r * 2:
                    # Propagation zone: partial penalty (linear decay)
                    decay = 1.0 - (dist - r) / r
                    partial_penalty = 1.0 + (evt.get("penalty", 5.0) - 1.0) * decay * 0.5
                    travel_time *= partial_penalty

        edge_src.append(node_id_to_idx[u])
        edge_dst.append(node_id_to_idx[v])
        edge_features_list.append(feat)
        edge_targets.append(travel_time)
        edge_keys_list.append((u, v, key))

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)
    y = torch.tensor(edge_targets, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_node_features = 5
    data.num_edge_features = 12
    data.node_ids = nodes
    data.node_id_to_idx = node_id_to_idx
    data.edge_keys = edge_keys_list
    data.lat_mean = lat_mean
    data.lat_std = lat_std
    data.lng_mean = lng_mean
    data.lng_std = lng_std
    return data


def load_all_snapshots(city_dir: str | Path) -> list[Path]:
    snapshots_dir = Path(city_dir) / "snapshots"
    if not snapshots_dir.exists():
        return []
    return sorted(f for f in snapshots_dir.glob("*.json") if "degraded" not in f.stem)


def _cache_key(city_dir, snapshot_files, degraded_copies, degrade_levels, risk_copies):
    """Hash of snapshot filenames + modification times + config to detect changes."""
    import hashlib
    parts = []
    for sf in snapshot_files:
        parts.append(f"{sf.name}:{sf.stat().st_mtime:.0f}")
    parts.append(f"deg={degraded_copies},{degrade_levels}")
    parts.append(f"risk={risk_copies}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def build_dataset(
    graph_path: str | Path,
    city_dir: str | Path,
    max_snapshots: int = 50,
    degraded_copies: int = 3,
    degrade_levels: list[float] | None = None,
    risk_copies: int = 2,
    force_rebuild: bool = False,
) -> list[Data]:
    """
    Build dataset with degraded data AND risk-aware augmentation.
    
    Caches the result to disk. Subsequent calls load instantly unless:
      - force_rebuild=True
      - snapshots changed (new files or modified)
      - config changed (degraded_copies, risk_copies, etc.)
    
    For each snapshot:
      - 1 clean sample (full data)
      - degraded_copies samples (missing traffic data)
      - risk_copies samples (normal input, crisis-penalized targets)
    """
    if degrade_levels is None:
        degrade_levels = [0.25, 0.50, 0.75]

    snapshot_files = load_all_snapshots(city_dir)[:max_snapshots]
    
    # Check cache
    cache_dir = Path(city_dir) / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    if snapshot_files:
        key = _cache_key(city_dir, snapshot_files, degraded_copies, degrade_levels, risk_copies)
        cache_path = cache_dir / f"dataset_{key}.pt"
        
        if not force_rebuild and cache_path.exists():
            print(f"Loading cached dataset from {cache_path.name}...")
            t0 = time.time()
            dataset = torch.load(cache_path, weights_only=False)
            print(f"  Loaded {len(dataset)} samples in {time.time() - t0:.1f}s")
            return dataset
    else:
        cache_path = None

    if not snapshot_files:
        print("No snapshots found.")
        return [graph_to_pyg_data(graph_path)]

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Identify structurally vulnerable points for event placement
    bridge_nodes = []
    major_intersections = []
    for n in G.nodes():
        lat = G.nodes[n].get("y", 0)
        lng = G.nodes[n].get("x", 0)
        if not lat or not lng:
            continue
        for _, _, d in G.edges(n, data=True):
            if d.get("bridge") and d.get("bridge") != "no":
                bridge_nodes.append((lat, lng))
                break
        if G.degree(n) >= 4:
            major_intersections.append((lat, lng))

    all_nodes = [(G.nodes[n].get("y", 0), G.nodes[n].get("x", 0)) for n in G.nodes()
                 if G.nodes[n].get("y", 0) != 0]

    print(f"  Risk locations: {len(bridge_nodes)} bridges, {len(major_intersections)} major intersections")

    t0 = time.time()
    dataset = []
    for sf in snapshot_files:
        print(f"  Processing: {sf.name}")

        # 1. Clean sample
        dataset.append(graph_to_pyg_data(graph_path, snapshot_path=sf))

        # 2. Degraded samples
        for level in degrade_levels[:degraded_copies]:
            dataset.append(graph_to_pyg_data(graph_path, snapshot_path=sf, degrade_pct=level))

        # 3. Risk-augmented samples
        for r in range(risk_copies):
            events = _generate_random_crisis(bridge_nodes, major_intersections, all_nodes)
            dataset.append(graph_to_pyg_data(graph_path, snapshot_path=sf, risk_events=events))

    total_per_snap = 1 + min(degraded_copies, len(degrade_levels)) + risk_copies
    build_time = time.time() - t0
    print(f"Dataset: {len(dataset)} samples "
          f"({len(snapshot_files)} snapshots × {total_per_snap} variants) "
          f"built in {build_time:.1f}s")
    print(f"  Variants: 1 clean + {min(degraded_copies, len(degrade_levels))} degraded + {risk_copies} risk-augmented")

    if cache_path:
        print(f"  Saving cache to {cache_path.name}...")
        torch.save(dataset, cache_path)
        for old in cache_dir.glob("dataset_*.pt"):
            if old != cache_path:
                old.unlink()
                print(f"  Removed old cache: {old.name}")

    return dataset


def _generate_random_crisis(bridge_nodes, major_intersections, all_nodes):
    """Generate 1-4 random crisis events at structurally vulnerable locations."""
    events = []
    n_events = random.randint(1, 4)

    for _ in range(n_events):
        r = random.random()
        if r < 0.4 and bridge_nodes:
            # Bridge collapse
            lat, lng = random.choice(bridge_nodes)
            events.append({
                "center": [lat, lng],
                "radius_m": random.uniform(50, 200),
                "penalty": random.uniform(8.0, 50.0),
            })
        elif r < 0.7 and major_intersections:
            # Major intersection blocked
            lat, lng = random.choice(major_intersections)
            events.append({
                "center": [lat, lng],
                "radius_m": random.uniform(200, 600),
                "penalty": random.uniform(3.0, 8.0),
            })
        elif all_nodes:
            # Random area congestion
            lat, lng = random.choice(all_nodes)
            events.append({
                "center": [lat, lng],
                "radius_m": random.uniform(300, 1500),
                "penalty": random.uniform(2.0, 5.0),
            })

    return events


if __name__ == "__main__":
    import sys
    city_dir = sys.argv[1] if len(sys.argv) > 1 else "cities/vilnius"
    force = "--rebuild" in sys.argv
    graph_path = Path(city_dir) / "graph.pkl"
    if not graph_path.exists():
        print(f"Graph not found at {graph_path}")
        sys.exit(1)
    dataset = build_dataset(graph_path, city_dir, force_rebuild=force)
    d = dataset[0]
    print(f"\nGraph: {d.x.shape[0]} nodes, {d.edge_index.shape[1]} edges")
    print(f"Features: {d.x.shape[1]} node, {d.edge_attr.shape[1]} edge")
    print(f"Travel time: {d.y.min():.1f}s — {d.y.max():.1f}s (mean {d.y.mean():.1f}s)")