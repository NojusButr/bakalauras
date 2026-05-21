"""
Degraded-data experiment: GNN vs Traffic (Dijkstra) routing comparison.

For each degradation level x each scenario x each route pair:
  - Both methods receive the SAME degraded snapshot
  - Both are evaluated against FULL data (ground truth)
  - Deterministic seeds for reproducibility

Scenarios = all user-created crisis presets + "Normal traffic" baseline.
Degradation levels include random % levels and a "minor" mode that strips
all minor-road traffic data.

WIN COMPARISON: Traffic vs GNN only.
Shortest path and Classifier are recorded but excluded from win/loss tallying.

Output:
    experiments/degraded_full_<timestamp>.json
    experiments/degraded_full_<timestamp>.csv
    experiments/degraded_full_<timestamp>.txt
"""

import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gnn"))

import pickle
import networkx as nx
import osmnx as ox

# ── Config ────────────────────────────────────────────────────────────────────

CITY = "vilnius"
DATA_LEVELS = ["minor", 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 0]

HIGHWAY_SPEEDS = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 70, "primary_link": 40, "secondary": 60, "secondary_link": 30,
    "tertiary": 50, "tertiary_link": 25, "residential": 30, "living_street": 10,
    "service": 20, "unclassified": 40,
}

# Diverse route pairs across Vilnius
ROUTE_PAIRS = [
    # Short (1-3km)
    {"name": "Old Town loop",           "start": [54.6822, 25.2840], "end": [54.6814, 25.2906]},
    {"name": "Gediminas Ave",           "start": [54.6860, 25.2858], "end": [54.6871, 25.2748]},
    {"name": "Station to centre",       "start": [54.6712, 25.2834], "end": [54.6811, 25.2848]},
    {"name": "Uzupis internal",         "start": [54.6803, 25.2933], "end": [54.6844, 25.3037]},
    # Medium (3-7km)
    {"name": "Zirmunai to Naujamiestis","start": [54.7143, 25.3031], "end": [54.6799, 25.2646]},
    {"name": "Antakalnis to centre",    "start": [54.7053, 25.3218], "end": [54.6872, 25.2830]},
    {"name": "Seskine to Old Town",     "start": [54.7143, 25.2516], "end": [54.6837, 25.2845]},
    {"name": "Karoliniskes to Uzupis",  "start": [54.6926, 25.2128], "end": [54.6804, 25.2930]},
    {"name": "Justiniskes to centre",   "start": [54.7212, 25.2208], "end": [54.6875, 25.2797]},
    {"name": "Fabijoniskes to river",   "start": [54.7294, 25.2420], "end": [54.6899, 25.2876]},
    # Long (7+km)
    {"name": "West to East",            "start": [54.7050, 25.1700], "end": [54.6920, 25.3400]},
    {"name": "North to South",          "start": [54.7350, 25.2700], "end": [54.6500, 25.2850]},
    {"name": "Pilaite to Pasilaiciai",  "start": [54.7100, 25.1900], "end": [54.7250, 25.2750]},
    {"name": "Lazdynai to Antakalnis",  "start": [54.6750, 25.2100], "end": [54.6962, 25.3252]},
    {"name": "Pilaite to Fabijoniskes", "start": [54.7050, 25.1750], "end": [54.7300, 25.2850]},
    {"name": "Lazdynai to Jeruzale",    "start": [54.6700, 25.2050], "end": [54.7350, 25.2600]},
    {"name": "South to Northeast",      "start": [54.6480, 25.2700], "end": [54.7200, 25.3100]},
    {"name": "Ring road W to E",        "start": [54.7150, 25.1650], "end": [54.6550, 25.3350]},
    {"name": "Grigiskes to Verkiai",    "start": [54.6724, 25.0750], "end": [54.7400, 25.2900]},
    {"name": "Salininkai to Seskine",   "start": [54.6550, 25.2500], "end": [54.7150, 25.2500]},
]

# ── Crisis presets (loaded from user-created JSON scenarios) ──────────────────

PRESETS_DIR = PROJECT_ROOT / "cities" / CITY / "user_presets"


def load_crisis_presets():
    """Load all JSON crisis presets from the user_presets directory."""
    presets = []
    if not PRESETS_DIR.exists():
        print(f"WARNING: Presets directory not found: {PRESETS_DIR}")
        return presets

    for json_file in sorted(PRESETS_DIR.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            preset = {
                "name": data.get("name", json_file.stem),
                "events": data.get("events", []),
                "source_file": json_file.name,
            }
            if preset["events"]:
                presets.append(preset)
                print(f"  Loaded preset: {preset['name']} ({len(preset['events'])} events)")
            else:
                print(f"  Skipped (no events): {json_file.name}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Failed to load {json_file.name}: {e}")

    return presets


# ── Helpers ───────────────────────────────────────────────────────────────────

def level_label(level):
    """Human-readable label for a degradation level."""
    if level == "minor":
        return "minor"
    return f"{level}%"


def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def load_system():
    from app.config import CITIES_CONFIG, CITIES_DIR
    from app.services.graph_service import load_or_create_city_graph
    from app.services.traffic_service import load_latest_snapshot

    place_name, coords = CITIES_CONFIG[CITY]
    G = load_or_create_city_graph(CITY, place_name, coords)
    snapshot = load_latest_snapshot(CITY)
    assert snapshot, "No snapshot found!"

    model_path = CITIES_DIR / CITY / "models" / "best_model.pt"
    graph_path = CITIES_DIR / CITY / "graph.pkl"
    assert model_path.exists(), f"GNN model not found at {model_path}"

    n_traffic = sum(1 for f in snapshot.get("features", [])
                    if f.get("properties", {}).get("current_speed_kph") is not None)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Snapshot: {n_traffic} edges with traffic data")

    return G, snapshot, model_path, graph_path


def build_traffic_weights(G, snapshot):
    traffic_by_osmid = {}
    for feat in snapshot.get("features", []):
        props = feat.get("properties", {})
        speed = props.get("current_speed_kph")
        if speed is None:
            continue
        osmid = props.get("osmid")
        if osmid is None:
            continue
        ids = osmid if isinstance(osmid, list) else [osmid]
        for oid in ids:
            traffic_by_osmid[int(oid)] = props

    weights = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        length = data.get("length", 1)
        osmid = data.get("osmid")
        speed = None
        if osmid:
            ids = osmid if isinstance(osmid, list) else [osmid]
            for oid in ids:
                if int(oid) in traffic_by_osmid:
                    speed = traffic_by_osmid[int(oid)].get("current_speed_kph")
                    break
        if speed is None:
            hw = data.get("highway", "unclassified")
            if isinstance(hw, list): hw = hw[0]
            speed = HIGHWAY_SPEEDS.get(hw, 30)
        weights[(u, v, key)] = length / max(speed / 3.6, 0.5)
    return weights


def degrade_snapshot(snapshot, keep_pct, seed, mode="random", graph=None):
    """
    Strip traffic data with different realistic patterns.

    mode="random": keep_pct controls how much data remains.
    mode="minor":  removes ALL minor-road traffic data (keep_pct ignored).
    """
    degraded = deepcopy(snapshot)
    features = degraded.get("features", [])

    # Only consider features that have traffic data
    # Protect direct event impacts (propagation_hop=0)
    has_traffic = []
    for i, f in enumerate(features):
        props = f.get("properties", {})
        if props.get("current_speed_kph") is None:
            continue
        if props.get("simulated") and props.get("propagation_hop", 99) == 0:
            continue
        has_traffic.append(i)

    if not has_traffic:
        return degraded
    if mode == "random" and keep_pct >= 100:
        return degraded

    rng = random.Random(seed)
    to_remove = set()

    if mode == "minor" and graph is not None:
        MINOR_TYPES = {"residential", "living_street", "service", "unclassified",
                       "tertiary", "tertiary_link"}
        osmid_to_highway = {}
        for u, v, k, d in graph.edges(keys=True, data=True):
            osmid = d.get("osmid")
            hw = d.get("highway", "unclassified")
            if isinstance(hw, list):
                hw = hw[0]
            if osmid is not None:
                ids = osmid if isinstance(osmid, list) else [osmid]
                for oid in ids:
                    osmid_to_highway[int(oid)] = hw

        for i in has_traffic:
            props = features[i].get("properties", {})
            osmid = props.get("osmid")
            hw = "unclassified"
            if osmid is not None:
                ids = osmid if isinstance(osmid, list) else [osmid]
                for oid in ids:
                    if int(oid) in osmid_to_highway:
                        hw = osmid_to_highway[int(oid)]
                        break
            if hw in MINOR_TYPES:
                to_remove.add(i)
    else:
        # Default random degradation
        n_remove = int(len(has_traffic) * (1 - keep_pct / 100))
        to_remove = set(rng.sample(has_traffic, min(n_remove, len(has_traffic))))

    for i in to_remove:
        props = features[i]["properties"]
        props["current_speed_kph"] = None
        props["free_flow_speed_kph"] = None
        props["jam_factor"] = None
        props["congestion_ratio"] = None

    return degraded


def apply_crisis(G, snapshot, preset, rng):
    """Apply a crisis preset to the graph. Events already contain coordinates."""
    G_crisis = G.copy()
    events_applied = []

    for evt in preset["events"]:
        c_lat, c_lng = evt["center"]
        radius = evt.get("radius_m", 80)
        speed_pct = evt.get("speed_reduction_pct", 0)
        evt_type = evt.get("type", "damage")

        events_applied.append(evt)

        edges_to_remove = []
        edges_to_slow = []
        for u, v, k, d in G_crisis.edges(keys=True, data=True):
            u_lat = G_crisis.nodes[u].get("y", 0)
            u_lng = G_crisis.nodes[u].get("x", 0)
            dist = haversine_m(c_lat, c_lng, u_lat, u_lng)

            if dist <= radius:
                if evt_type == "damage" and speed_pct == 0:
                    edges_to_remove.append((u, v, k))
                elif speed_pct > 0:
                    edges_to_slow.append((u, v, k, speed_pct))

        for u, v, k in edges_to_remove:
            if G_crisis.has_edge(u, v, k):
                G_crisis.remove_edge(u, v, k)

        for u, v, k, pct in edges_to_slow:
            if G_crisis.has_edge(u, v, k):
                length = G_crisis[u][v][k].get("length", 1)
                hw = G_crisis[u][v][k].get("highway", "unclassified")
                if isinstance(hw, list): hw = hw[0]
                base_speed = HIGHWAY_SPEEDS.get(hw, 30)
                reduced_speed = base_speed * (1 - pct / 100)
                G_crisis[u][v][k]["_crisis_speed"] = max(reduced_speed, 1)

    return G_crisis, events_applied


def route_and_evaluate(G_routing, G_original, start, end, snapshot_routing, snapshot_full,
                       method, model_path=None, graph_path=None, classifier_path=None):
    """Route with given method on routing graph, evaluate on full data.

    Returns {
        "distance_m": route distance in meters,
        "real_time_s": travel time evaluated on full data (seconds),
        "compute_time_s": time to compute route (seconds),
    }
    """
    compute_start = time.time()

    try:
        start_node = ox.distance.nearest_nodes(G_routing, start[1], start[0])
        end_node = ox.distance.nearest_nodes(G_routing, end[1], end[0])

        if method == "shortest":
            # Distance-only shortest path
            route_nodes = nx.shortest_path(G_routing, start_node, end_node, weight="length")

        elif method == "traffic":
            weights = build_traffic_weights(G_routing, snapshot_routing)
            attr = "_exp_tt"
            for (u, v, key), t in weights.items():
                if G_routing.has_edge(u, v, key):
                    G_routing[u][v][key][attr] = t
            route_nodes = nx.shortest_path(G_routing, start_node, end_node, weight=attr)

        elif method == "gnn":
            from gnn_service import gnn_route_weights
            weights = gnn_route_weights(G_routing, graph_path, model_path, snapshot_routing)
            attr = "_exp_gnn"
            for (u, v, key), t in weights.items():
                if G_routing.has_edge(u, v, key):
                    G_routing[u][v][key][attr] = t
            route_nodes = nx.shortest_path(G_routing, start_node, end_node, weight=attr)

        elif method == "classifier":
            # LSTM-GNN route classifier
            from app.services.classifier_service import classifier_route
            result = classifier_route(G_routing, start_node, end_node, classifier_path, snapshot_routing)
            if result is None or result[0] is None:
                return None
            route_nodes = result[0]

        else:
            return None

        compute_time = time.time() - compute_start

        # Distance on original graph
        dist = sum(G_original[u][v][0].get("length", 0)
                   for u, v in zip(route_nodes[:-1], route_nodes[1:])
                   if G_original.has_edge(u, v))

        # Evaluate with FULL data (ground truth travel time)
        full_weights = build_traffic_weights(G_original, snapshot_full)
        real_time = 0
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            w = full_weights.get((u, v, 0))
            if w is not None:
                real_time += w
            else:
                edge = G_original[u][v][0] if G_original.has_edge(u, v) else {}
                length = edge.get("length", 100)
                hw = edge.get("highway", "unclassified")
                if isinstance(hw, list): hw = hw[0]
                speed = HIGHWAY_SPEEDS.get(hw, 30)
                real_time += length / max(speed / 3.6, 0.5)

        return {
            "distance_m": round(dist),
            "real_time_s": round(real_time),
            "compute_time_s": round(compute_time, 3),
        }

    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, Exception):
        return None


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_summary_row(lbl, s):
    """Format one row of the summary table (Traffic vs GNN only)."""
    traffic_w = s['traffic_wins']
    gnn_w = s['gnn_wins']
    ties = s['ties']
    total = s['total']

    avg_traffic = s['avg_traffic_time_s']
    avg_gnn = s['avg_gnn_time_s']
    avg_time_diff = s['avg_time_diff_s']
    relative_imp = s['relative_improvement_pct']

    avg_compute_traffic = s['avg_compute_traffic_s']
    avg_compute_gnn = s['avg_compute_gnn_s']

    win_pct = round(gnn_w / max(total, 1) * 100, 1)

    return (f"  {lbl:>6} | "
            f"{gnn_w:>3}/{total:<3} | "
            f"{traffic_w:>3}/{total:<3} | "
            f"{ties:>3} | "
            f"{win_pct:>6.1f}% | "
            f"T:{avg_traffic:>6.1f}s G:{avg_gnn:>6.1f}s | "
            f"Δ:{avg_time_diff:>6.1f}s | "
            f"{relative_imp:>+7.1f}% | "
            f"T:{avg_compute_traffic:.3f}s G:{avg_compute_gnn:.3f}s")


HEADER = (f"  {'Level':>6} | {'GNN W':>7} | {'Traffic':>7} | "
          f"{'Ties':>3} | {'Win %':>6} | {'Avg Times':>14} | "
          f"{'Δ':>8} | {'Impr':>7} | Compute Times")
SEP = "  " + "-" * 120


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("DEGRADED DATA EXPERIMENT - ALL ROUTES x ALL SCENARIOS")
    print("Win comparison: Traffic vs GNN only (Shortest and Classifier recorded, not compared)")
    print(f"Degradation levels: {[level_label(l) for l in DATA_LEVELS]}")
    print(f"Route pairs: {len(ROUTE_PAIRS)}")
    print("=" * 80)

    G, snapshot, model_path, graph_path = load_system()

    # Setup classifier path (optional - will skip if not found)
    classifier_path = PROJECT_ROOT / "cities" / CITY / "models" / "route_classifier.pt"
    if classifier_path.exists():
        print(f"Route classifier found: {classifier_path}")
    else:
        print(f"Note: Route classifier not found at {classifier_path} (will skip classifier method)")

    print(f"\nLoading crisis presets from {PRESETS_DIR}...")
    crisis_presets = load_crisis_presets()
    if not crisis_presets:
        print("ERROR: No crisis presets found. Place JSON files in cities/vilnius/user_presets/")
        sys.exit(1)
    print(f"Loaded {len(crisis_presets)} crisis presets\n")

    # Build scenario list: normal baseline + all crisis presets
    scenarios = [{"name": "Normal traffic", "preset": None}]
    for p in crisis_presets:
        scenarios.append({"name": p["name"], "preset": p})

    n_total = len(DATA_LEVELS) * len(ROUTE_PAIRS) * len(scenarios)
    print(f"Total runs: {len(DATA_LEVELS)} levels x {len(ROUTE_PAIRS)} routes "
          f"x {len(scenarios)} scenarios = {n_total}")
    print("=" * 80)

    all_results = []
    summary_by_level = {}
    run_count = 0

    for level in DATA_LEVELS:
        is_minor = level == "minor"
        lbl = level_label(level)
        print(f"\n{'='*80}")
        print(f"  DEGRADATION: {lbl}")
        print(f"{'='*80}")

        # ── Win tracking: Traffic vs GNN only ────────────────────────────────
        traffic_wins = 0
        gnn_wins = 0
        ties = 0

        # Timing accumulators for all 4 methods (for reporting, not comparison)
        method_times = {"shortest": [], "traffic": [], "gnn": [], "classifier": []}
        method_compute_times = {"shortest": [], "traffic": [], "gnn": [], "classifier": []}
        route_durations = []
        time_differences = []  # |traffic_time - gnn_time| per route

        for scenario in scenarios:
            print(f"\n  Scenario: {scenario['name']}")
            print(f"  {'-'*70}")

            for route_pair in ROUTE_PAIRS:
                run_count += 1

                # Deterministic seed
                seed_val = 999 if is_minor else level
                seed = int(abs(route_pair["start"][0] * 1e6)
                           + abs(route_pair["end"][1] * 1e6)
                           + seed_val * 100
                           + hash(scenario["name"]) % 10000) % (2**31)
                rng = random.Random(seed)

                # Degrade snapshot
                if is_minor:
                    degraded = degrade_snapshot(snapshot, 100, seed,
                                                mode="minor", graph=G)
                else:
                    degraded = degrade_snapshot(snapshot, level, seed)

                # Apply crisis if this scenario has one
                if scenario["preset"] is not None:
                    G_routing, events = apply_crisis(
                        G.copy(), snapshot, scenario["preset"], rng)
                else:
                    G_routing = G.copy()
                    events = []

                # ── Route with all 4 methods ──────────────────────────────────
                results_by_method = {}

                for method in ["shortest", "traffic", "gnn", "classifier"]:
                    if method == "classifier" and not classifier_path.exists():
                        results_by_method[method] = None
                        continue

                    result = route_and_evaluate(
                        G_routing.copy(), G,
                        route_pair["start"], route_pair["end"],
                        degraded, snapshot, method,
                        model_path=model_path if method == "gnn" else None,
                        graph_path=graph_path if method == "gnn" else None,
                        classifier_path=classifier_path if method == "classifier" else None,
                    )
                    results_by_method[method] = result

                    if result is not None:
                        method_times[method].append(result["real_time_s"])
                        method_compute_times[method].append(result["compute_time_s"])

                shortest   = results_by_method.get("shortest")
                traffic    = results_by_method.get("traffic")
                gnn        = results_by_method.get("gnn")
                classifier = results_by_method.get("classifier")

                # ── Win comparison: Traffic vs GNN only ───────────────────────
                # Requires both traffic and GNN to have valid results.
                # Shortest and classifier results are stored but do not affect
                # traffic_wins / gnn_wins / ties.
                if traffic is None or gnn is None:
                    winner = "NO_DATA"
                    print(f"    {route_pair['name']:30s} | SKIPPED (traffic={traffic is not None}, gnn={gnn is not None})")
                    if traffic is None and shortest is None:
                        continue 
                else:
                    traffic_time = traffic["real_time_s"]
                    gnn_time     = gnn["real_time_s"]
                    diff_s       = traffic_time - gnn_time   # positive = GNN faster

                    time_differences.append(abs(diff_s))

                    if shortest is not None:
                        route_durations.append(
                            (shortest["real_time_s"] + traffic_time) / 2 / 60
                        )

                    if abs(diff_s) < 2:       # 2-second tie threshold
                        winner = "TIE"
                        ties += 1
                    elif diff_s > 0:
                        winner = "GNN"
                        gnn_wins += 1
                    else:
                        winner = "TRAFFIC"
                        traffic_wins += 1

                    # Progress display
                    marker = ("<<GNN" if winner == "GNN"
                              else "TRAF>>" if winner == "TRAFFIC"
                              else "  ==  ")

                    times_str = f"T:{traffic_time:4.0f}s G:{gnn_time:4.0f}s"
                    if shortest:
                        times_str = f"S:{shortest['real_time_s']:4.0f}s " + times_str
                    if classifier:
                        times_str += f" C:{classifier['real_time_s']:4.0f}s"

                    print(f"    {route_pair['name']:30s} | {times_str:50s} {marker}")

                # Record full result row
                result_rec = {
                    "data_level":                  level,
                    "route":                       route_pair["name"],
                    "scenario":                    scenario["name"],
                    "is_crisis":                   scenario["preset"] is not None,
                    "route_distance_km":           round(shortest["distance_m"] / 1000, 2) if shortest else None,
                    "avg_route_duration_minutes":  round((shortest["real_time_s"] + traffic["real_time_s"]) / 2 / 60, 2)
                                                   if (shortest and traffic) else None,
                    # Travel times (ground-truth evaluation)
                    "time_shortest_s":             shortest["real_time_s"]   if shortest    else None,
                    "time_traffic_s":              traffic["real_time_s"]    if traffic     else None,
                    "time_gnn_s":                  gnn["real_time_s"]        if gnn         else None,
                    "time_classifier_s":           classifier["real_time_s"] if classifier  else None,
                    # Compute times
                    "compute_time_shortest_s":     shortest["compute_time_s"]   if shortest    else None,
                    "compute_time_traffic_s":      traffic["compute_time_s"]    if traffic     else None,
                    "compute_time_gnn_s":          gnn["compute_time_s"]        if gnn         else None,
                    "compute_time_classifier_s":   classifier["compute_time_s"] if classifier  else None,
                    # Winner = Traffic vs GNN comparison only
                    "winner":                      winner,
                    "seed":                        seed,
                }
                all_results.append(result_rec)

                if run_count % 50 == 0:
                    print(f"\n    ... progress: {run_count}/{n_total} "
                          f"({run_count * 100 // n_total}%)\n")

        # ── Per-level summary statistics ──────────────────────────────────────
        total_valid = traffic_wins + gnn_wins + ties

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        avg_traffic_time    = _avg(method_times["traffic"])
        avg_gnn_time        = _avg(method_times["gnn"])
        avg_time_diff       = _avg(time_differences)
        avg_compute_traffic = _avg(method_compute_times["traffic"])
        avg_compute_gnn     = _avg(method_compute_times["gnn"])

        # Relative improvement: how much faster is the best vs worst of traffic/gnn
        times_pair = [t for t in [avg_traffic_time, avg_gnn_time] if t > 0]
        slowest = max(times_pair) if times_pair else 1
        fastest = min(times_pair) if times_pair else 1
        relative_improvement_pct = (slowest - fastest) / slowest * 100 if slowest > 0 else 0.0

        summary_by_level[str(level)] = {
            "traffic_wins":             traffic_wins,
            "gnn_wins":                 gnn_wins,
            "ties":                     ties,
            "total":                    total_valid,
            "avg_traffic_time_s":       round(avg_traffic_time, 1),
            "avg_gnn_time_s":           round(avg_gnn_time, 1),
            "avg_time_diff_s":          round(avg_time_diff, 1),
            "relative_improvement_pct": round(relative_improvement_pct, 1),
            "avg_compute_traffic_s":    round(avg_compute_traffic, 4),
            "avg_compute_gnn_s":        round(avg_compute_gnn, 4),
        }

        print(f"\n  Level {lbl} summary: GNN {gnn_wins}/{total_valid}, "
              f"Traffic {traffic_wins}, Ties {ties}")
        print(f"    Avg times - Traffic: {avg_traffic_time:6.1f}s, "
              f"GNN: {avg_gnn_time:6.1f}s  (diff: {abs(avg_traffic_time - avg_gnn_time):+.1f}s)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV export
    import csv
    csv_path = out_dir / f"degraded_full_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "data_level", "route", "scenario", "is_crisis", "route_distance_km",
            "avg_route_duration_minutes", "time_shortest_s", "time_traffic_s",
            "time_gnn_s", "time_classifier_s", "compute_time_shortest_s",
            "compute_time_traffic_s", "compute_time_gnn_s", "compute_time_classifier_s",
            "winner", "seed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # JSON
    full_output = {
        "timestamp": timestamp,
        "config": {
            "data_levels":    [str(l) for l in DATA_LEVELS],
            "n_route_pairs":  len(ROUTE_PAIRS),
            "scenarios":      [s["name"] for s in scenarios],
            "crisis_presets": [p["name"] for p in crisis_presets],
            "methods":        ["shortest", "traffic", "gnn", "classifier"],
            "win_comparison": "traffic vs gnn only",
        },
        "summary_by_level": summary_by_level,
        "results": all_results,
    }
    json_path = out_dir / f"degraded_full_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)

    # text report
    txt_path = out_dir / f"degraded_full_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 180 + "\n")
        f.write("DEGRADED DATA EXPERIMENT - FULL RESULTS\n")
        f.write("Win comparison: Traffic vs GNN only  "
                "(Shortest path and Classifier are recorded but NOT included in wins/losses)\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Routes: {len(ROUTE_PAIRS)}, "
                f"Scenarios: {len(scenarios)}, "
                f"Levels: {len(DATA_LEVELS)}\n")
        f.write(f"Runs per level: {len(ROUTE_PAIRS) * len(scenarios)}\n")
        f.write("=" * 180 + "\n\n")

        f.write("SUMMARY BY DEGRADATION LEVEL\n")
        f.write("(Wins = Traffic vs GNN comparison only; "
                "Ties = within 2-second threshold)\n\n")
        f.write(HEADER + "\n")
        f.write(SEP + "\n")
        for level in DATA_LEVELS:
            s = summary_by_level[str(level)]
            f.write(fmt_summary_row(level_label(level), s) + "\n")

        f.write("\n\nDETAILED RESULTS (per route)\n")
        detail_hdr = (f"  {'Level':>6} | {'Route':30s} | "
                      f"{'Scenario':35s} | "
                      f"{'Dist(km)':>8} | {'Winner':>9} | "
                      f"{'Short(s)':>8} {'Traf(s)':>8} {'GNN(s)':>8} {'Cls(s)':>8}")
        f.write(detail_hdr + "\n")
        f.write("  " + "-" * (len(detail_hdr) - 2) + "\n")
        for r in all_results:
            lbl = level_label(r["data_level"])
            short_str = f"{r['time_shortest_s']:8.0f}"   if r["time_shortest_s"]   is not None else "       -"
            traf_str  = f"{r['time_traffic_s']:8.0f}"    if r["time_traffic_s"]    is not None else "       -"
            gnn_str   = f"{r['time_gnn_s']:8.0f}"        if r["time_gnn_s"]        is not None else "       -"
            cls_str   = f"{r['time_classifier_s']:8.0f}" if r["time_classifier_s"] is not None else "       -"
            dist_str  = f"{r['route_distance_km']:8.2f}" if r["route_distance_km"] is not None else "       -"
            f.write(f"  {lbl:>6} | {r['route']:30s} | "
                    f"{r['scenario']:35s} | "
                    f"{dist_str} | {r['winner']:>9} | "
                    f"{short_str} {traf_str} {gnn_str} {cls_str}\n")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("OVERALL SUMMARY - TRAFFIC vs GNN  "
          "(Shortest and Classifier recorded only, not compared)")
    print(f"{'=' * 120}\n")
    print(HEADER)
    print(SEP)
    for level in DATA_LEVELS:
        s = summary_by_level[str(level)]
        print(fmt_summary_row(level_label(level), s))

    print(f"\nTotal runs: {len(all_results)}")
    print(f"\nResults saved to:\n  {csv_path}\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()