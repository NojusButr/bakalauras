"""
For each degradation level (90%, 85%, 80%, ... 5%):
  - Routes 20 diverse pairs
  - ~20% under normal traffic, ~80% under random crisis simulations
  - Both Dijkstra and GNN receive SAME degraded data
  - Both evaluated against FULL data (ground truth)
  - Deterministic seeds for reproducibility

Output:
    experiments/degraded_full_results.json
    experiments/degraded_full_summary.txt

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
DATA_LEVELS = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
ROUTES_PER_LEVEL = 20
CRISIS_RATIO = 0.80  # 80% of routes under crisis, 20% normal traffic

HIGHWAY_SPEEDS = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 70, "primary_link": 40, "secondary": 60, "secondary_link": 30,
    "tertiary": 50, "tertiary_link": 25, "residential": 30, "living_street": 10,
    "service": 20, "unclassified": 40,
}

# Diverse route pairs across Vilnius
ROUTE_PAIRS = [
    # Short (1-3km)
    {"name": "Old Town loop",         "start": [54.6823, 25.2878], "end": [54.6775, 25.2770]},
    {"name": "Gediminas Ave",         "start": [54.6873, 25.2600], "end": [54.6838, 25.2820]},
    {"name": "Station to centre",     "start": [54.6705, 25.2845], "end": [54.6830, 25.2785]},
    {"name": "Uzupis internal",       "start": [54.6795, 25.2950], "end": [54.6760, 25.3050]},
    # Medium (3-7km)
    {"name": "Zirmunai to Naujam",    "start": [54.7100, 25.2900], "end": [54.6770, 25.2580]},
    {"name": "Antakalnis to centre",  "start": [54.6950, 25.3200], "end": [54.6838, 25.2790]},
    {"name": "Seskine to Old Town",   "start": [54.7100, 25.2400], "end": [54.6800, 25.2850]},
    {"name": "Karoliniskes to Uzupis","start": [54.6940, 25.2280], "end": [54.6785, 25.2980]},
    {"name": "Justiniskes to centre", "start": [54.7050, 25.2200], "end": [54.6850, 25.2750]},
    {"name": "Fabijoniskes to river", "start": [54.7250, 25.2600], "end": [54.6950, 25.2700]},
    # Long (7-12km)
    {"name": "West to East",          "start": [54.7050, 25.1700], "end": [54.6920, 25.3400]},
    {"name": "North to South",        "start": [54.7350, 25.2700], "end": [54.6500, 25.2850]},
    {"name": "Pilaite to Pasilaiciai","start": [54.7100, 25.1900], "end": [54.7250, 25.2750]},
    {"name": "Lazdynai to Antakalnis","start": [54.6750, 25.2100], "end": [54.6980, 25.3250]},
    # Cross-city (12km+)
    {"name": "Pilaite to Fabijoniskes","start": [54.7050, 25.1750], "end": [54.7300, 25.2850]},
    {"name": "Lazdynai to Jeruzale",   "start": [54.6700, 25.2050], "end": [54.7350, 25.2600]},
    {"name": "South to Northeast",     "start": [54.6480, 25.2700], "end": [54.7200, 25.3100]},
    {"name": "Ring road W to E",       "start": [54.7150, 25.1650], "end": [54.6550, 25.3350]},
    {"name": "Grigiškės to Verkiai",   "start": [54.6650, 25.1850], "end": [54.7400, 25.2900]},
    {"name": "Trakų to Šeškinė",      "start": [54.6550, 25.2500], "end": [54.7150, 25.2500]},
]

# Crisis templates — randomly selected and placed
CRISIS_TEMPLATES = [
    {
        "name": "Bridge collapse",
        "events": [
            {"type": "damage", "radius_m": 100, "speed_reduction_pct": 0},
        ],
        "prop_depth": 3, "prop_decay": 50,
    },
    {
        "name": "Area congestion",
        "events": [
            {"type": "congestion", "radius_m": 600, "speed_reduction_pct": 8},
        ],
        "prop_depth": 4, "prop_decay": 45,
    },
    {
        "name": "Road damage + congestion",
        "events": [
            {"type": "damage", "radius_m": 120, "speed_reduction_pct": 0},
            {"type": "congestion", "radius_m": 400, "speed_reduction_pct": 10},
        ],
        "prop_depth": 3, "prop_decay": 55,
    },
    {
        "name": "Multi-point crisis",
        "events": [
            {"type": "damage", "radius_m": 80, "speed_reduction_pct": 0},
            {"type": "congestion", "radius_m": 500, "speed_reduction_pct": 8},
            {"type": "damage", "radius_m": 100, "speed_reduction_pct": 0},
        ],
        "prop_depth": 4, "prop_decay": 50,
    },
    {
        "name": "Highway blockage",
        "events": [
            {"type": "damage", "radius_m": 150, "speed_reduction_pct": 0},
            {"type": "congestion", "radius_m": 500, "speed_reduction_pct": 5},
        ],
        "prop_depth": 3, "prop_decay": 60,
    },
]

# Known vulnerable locations for crisis placement
CRISIS_LOCATIONS = [
    [54.6920, 25.2530],  # Zverynas bridge
    [54.6810, 25.2850],  # Old Town centre
    [54.7050, 25.2100],  # Western highway
    [54.6950, 25.2900],  # Zirmunai area
    [54.7100, 25.2400],  # Seskine junction
    [54.6750, 25.2300],  # Karoliniskes
    [54.7200, 25.2700],  # Fabijoniskes
    [54.6700, 25.2800],  # Naujamiestis
    [54.6850, 25.3100],  # Antakalnis entrance
    [54.7300, 25.2500],  # Northern ring
]


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


def degrade_snapshot(snapshot, keep_pct, seed):
    degraded = deepcopy(snapshot)
    features = degraded.get("features", [])
    has_traffic = [i for i, f in enumerate(features)
                   if f.get("properties", {}).get("current_speed_kph") is not None]
    if not has_traffic or keep_pct >= 100:
        return degraded
    rng = random.Random(seed)
    n_remove = int(len(has_traffic) * (1 - keep_pct / 100))
    to_remove = set(rng.sample(has_traffic, min(n_remove, len(has_traffic))))
    for i in to_remove:
        props = features[i]["properties"]
        props["current_speed_kph"] = None
        props["free_flow_speed_kph"] = None
        props["jam_factor"] = None
        props["congestion_ratio"] = None
    return degraded


def apply_crisis(G, snapshot, crisis_template, locations, rng):
    """Apply crisis events at random locations, return modified graph."""
    G_crisis = G.copy()
    events_applied = []

    for evt_template in crisis_template["events"]:
        loc = rng.choice(locations)
        evt = {**evt_template, "center": loc}
        events_applied.append(evt)

        c_lat, c_lng = loc
        radius = evt["radius_m"]
        speed_pct = evt["speed_reduction_pct"]

        edges_to_remove = []
        for u, v, k, d in G_crisis.edges(keys=True, data=True):
            u_lat = G_crisis.nodes[u].get("y", 0)
            u_lng = G_crisis.nodes[u].get("x", 0)
            dist = haversine_m(c_lat, c_lng, u_lat, u_lng)

            if dist <= radius:
                if evt["type"] == "damage" and speed_pct == 0:
                    edges_to_remove.append((u, v, k))

        for u, v, k in edges_to_remove:
            if G_crisis.has_edge(u, v, k):
                G_crisis.remove_edge(u, v, k)

    return G_crisis, events_applied


def route_and_evaluate(G_routing, G_original, start, end, snapshot_routing, snapshot_full,
                       method, model_path=None, graph_path=None):
    """Route with given method on routing graph, evaluate on full data."""
    try:
        start_node = ox.distance.nearest_nodes(G_routing, start[1], start[0])
        end_node = ox.distance.nearest_nodes(G_routing, end[1], end[0])

        if method == "traffic":
            weights = build_traffic_weights(G_routing, snapshot_routing)
            attr = "_exp_tt"
        elif method == "gnn":
            from gnn_service import gnn_route_weights
            weights = gnn_route_weights(G_routing, graph_path, model_path, snapshot_routing)
            attr = "_exp_gnn"
        else:
            return None

        for (u, v, key), t in weights.items():
            if G_routing.has_edge(u, v, key):
                G_routing[u][v][key][attr] = t

        route_nodes = nx.shortest_path(G_routing, start_node, end_node, weight=attr)

        # Distance
        dist = sum(G_original[u][v][0].get("length", 0)
                   for u, v in zip(route_nodes[:-1], route_nodes[1:])
                   if G_original.has_edge(u, v))

        # Evaluate with FULL data
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

        # GNN estimate
        gnn_est = None
        if method == "gnn":
            gnn_est = sum(G_routing[u][v][0].get(attr, 0)
                          for u, v in zip(route_nodes[:-1], route_nodes[1:])
                          if G_routing.has_edge(u, v))

        return {
            "distance_m": round(dist),
            "real_time_s": round(real_time),
            "real_time_min": round(real_time / 60, 1),
            "gnn_est_s": round(gnn_est) if gnn_est else None,
        }

    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
        return None


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("COMPREHENSIVE DEGRADED DATA EXPERIMENT")
    print(f"Levels: {DATA_LEVELS}")
    print(f"Routes per level: {ROUTES_PER_LEVEL} ({int(CRISIS_RATIO*100)}% crisis, {int((1-CRISIS_RATIO)*100)}% normal)")
    print("=" * 80)

    G, snapshot, model_path, graph_path = load_system()

    n_crisis = int(ROUTES_PER_LEVEL * CRISIS_RATIO)
    n_normal = ROUTES_PER_LEVEL - n_crisis

    all_results = []
    summary_by_level = {}

    for level in DATA_LEVELS:
        print(f"\n--- {level}% data ---")
        level_results = []
        gnn_wins = 0
        traffic_wins = 0
        ties = 0
        gnn_advantages = []

        for r_idx in range(ROUTES_PER_LEVEL):
            route_pair = ROUTE_PAIRS[r_idx % len(ROUTE_PAIRS)]
            is_crisis = r_idx < n_crisis

            # Deterministic seed from coords + level + index
            seed = int(abs(route_pair["start"][0] * 1e6) + abs(route_pair["end"][1] * 1e6)
                       + level * 100 + r_idx) % (2**31)
            rng = random.Random(seed)

            # Degrade snapshot
            degraded = degrade_snapshot(snapshot, level, seed)

            # Apply crisis if needed
            if is_crisis:
                crisis = rng.choice(CRISIS_TEMPLATES)
                G_routing = G.copy()
                G_routing, events = apply_crisis(G_routing, snapshot, crisis, CRISIS_LOCATIONS, rng)
                scenario = crisis["name"]
            else:
                G_routing = G.copy()
                events = []
                scenario = "Normal traffic"

            # Route with both methods
            G_traffic = G_routing.copy()
            G_gnn = G_routing.copy()

            traffic_result = route_and_evaluate(
                G_traffic, G, route_pair["start"], route_pair["end"],
                degraded, snapshot, "traffic"
            )
            gnn_result = route_and_evaluate(
                G_gnn, G, route_pair["start"], route_pair["end"],
                degraded, snapshot, "gnn", model_path, graph_path
            )

            if traffic_result and gnn_result:
                t_time = traffic_result["real_time_min"]
                g_time = gnn_result["real_time_min"]
                diff = t_time - g_time  # positive = GNN wins

                if abs(diff) < 0.3:
                    winner = "TIE"
                    ties += 1
                elif diff > 0:
                    winner = "GNN"
                    gnn_wins += 1
                else:
                    winner = "TRAFFIC"
                    traffic_wins += 1

                gnn_advantages.append(diff)

                result = {
                    "data_pct": level,
                    "route": route_pair["name"],
                    "scenario": scenario,
                    "is_crisis": is_crisis,
                    "traffic_dist_km": round(traffic_result["distance_m"] / 1000, 2),
                    "traffic_time_min": t_time,
                    "gnn_dist_km": round(gnn_result["distance_m"] / 1000, 2),
                    "gnn_time_min": g_time,
                    "gnn_est_min": round(gnn_result["gnn_est_s"] / 60, 1) if gnn_result["gnn_est_s"] else None,
                    "advantage_min": round(diff, 1),
                    "winner": winner,
                    "seed": seed,
                }
                level_results.append(result)
                all_results.append(result)

                marker = "<<GNN" if winner == "GNN" else "TRAF>>" if winner == "TRAFFIC" else "  ==  "
                print(f"  {route_pair['name']:30s} | T:{t_time:5.1f} G:{g_time:5.1f} {marker} | {scenario}")
            else:
                failed = "traffic" if not traffic_result else "gnn"
                print(f"  {route_pair['name']:30s} | FAILED ({failed}) | {scenario}")

        avg_adv = sum(gnn_advantages) / max(len(gnn_advantages), 1)
        total_valid = gnn_wins + traffic_wins + ties
        summary_by_level[level] = {
            "gnn_wins": gnn_wins,
            "traffic_wins": traffic_wins,
            "ties": ties,
            "total": total_valid,
            "gnn_win_rate": round(gnn_wins / max(total_valid, 1) * 100, 1),
            "avg_advantage_min": round(avg_adv, 2),
        }

        print(f"  Summary: GNN wins {gnn_wins}/{total_valid}, "
              f"Traffic wins {traffic_wins}, Ties {ties}, "
              f"Avg advantage: {avg_adv:+.1f} min")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    full_output = {
        "timestamp": timestamp,
        "config": {
            "data_levels": DATA_LEVELS,
            "routes_per_level": ROUTES_PER_LEVEL,
            "crisis_ratio": CRISIS_RATIO,
            "n_route_pairs": len(ROUTE_PAIRS),
        },
        "summary_by_level": summary_by_level,
        "results": all_results,
    }
    json_path = out_dir / f"degraded_full_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)

    # Human-readable summary
    txt_path = out_dir / f"degraded_full_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DEGRADED DATA EXPERIMENT — SUMMARY\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Routes per level: {ROUTES_PER_LEVEL} ({int(CRISIS_RATIO*100)}% crisis)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Data%':>6} | {'GNN wins':>9} | {'Traffic':>9} | {'Ties':>6} | {'Win rate':>9} | {'Avg adv':>9}\n")
        f.write("-" * 70 + "\n")
        for level in DATA_LEVELS:
            s = summary_by_level[level]
            bar = "█" * s["gnn_wins"] + "▒" * s["ties"] + "░" * s["traffic_wins"]
            f.write(f"  {level:3d}% | {s['gnn_wins']:4d}/{s['total']:2d}   | "
                    f"{s['traffic_wins']:4d}/{s['total']:2d}   | {s['ties']:4d}  | "
                    f"{s['gnn_win_rate']:6.1f}%  | {s['avg_advantage_min']:+6.2f} min  {bar}\n")

        f.write("\n\nDETAILED RESULTS:\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            f.write(f"  [{r['data_pct']:3d}%] {r['route']:30s} | "
                    f"T:{r['traffic_time_min']:5.1f} G:{r['gnn_time_min']:5.1f} "
                    f"({r['advantage_min']:+.1f}) {r['winner']:>7} | {r['scenario']}\n")

    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Data%':>6} | {'GNN wins':>9} | {'Traffic':>9} | {'Ties':>6} | {'Win rate':>9} | {'Avg advantage':>14}")
    print("-" * 75)
    for level in DATA_LEVELS:
        s = summary_by_level[level]
        bar = "█" * s["gnn_wins"] + "▒" * s["ties"] + "░" * s["traffic_wins"]
        print(f"  {level:3d}% | {s['gnn_wins']:4d}/{s['total']:2d}   | "
              f"{s['traffic_wins']:4d}/{s['total']:2d}   | {s['ties']:4d}  | "
              f"{s['gnn_win_rate']:6.1f}%  | {s['avg_advantage_min']:+6.2f} min  {bar}")

    print(f"\nResults saved to:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()