"""
experiment_degraded.py — Route with partial data, evaluate with full data.

Dijkstra gets degraded snapshot for path selection (simulating limited coverage).
GNN always uses its trained model (no degradation).
ALL routes evaluated against the FULL snapshot for fair ground-truth travel time.
"""

import json
import random
import requests
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone

API = "http://localhost:8000"

ROUTES = [
    ("West to East", [54.6903, 25.2150], [54.6897, 25.3361]),
    ("North to South", [54.7050, 25.2700], [54.6700, 25.2800]),
    ("Centre to Antakalnis", [54.6870, 25.2640], [54.6960, 25.3200]),
    ("Zirmunai to Naujamiestis", [54.7020, 25.2900], [54.6750, 25.2550]),
    ("Short urban route", [54.6880, 25.2750], [54.6830, 25.2900]),
]

DEGRADATION_LEVELS = [100, 75, 50, 25, 10]


def get_snapshot_list(city="vilnius"):
    r = requests.get(f"{API}/traffic/snapshot/{city}/list")
    return r.json().get("snapshots", []) if r.ok else []


def get_latest_snapshot(city="vilnius"):
    r = requests.get(f"{API}/traffic/snapshot/{city}/latest")
    return r.json() if r.ok else None


def degrade_snapshot(snapshot, keep_pct):
    degraded = deepcopy(snapshot)
    features = degraded.get("features", [])
    has_traffic = [i for i, f in enumerate(features) if f.get("properties", {}).get("current_speed_kph") is not None]
    if not has_traffic:
        return degraded
    n_remove = int(len(has_traffic) * (1 - keep_pct / 100))
    to_remove = set(random.sample(has_traffic, n_remove))
    for i in to_remove:
        props = features[i]["properties"]
        props["current_speed_kph"] = None
        props["free_flow_speed_kph"] = None
        props["jam_factor"] = None
        props["congestion_ratio"] = None
        props["congestion_level"] = "unknown"
        props["confidence"] = None
    return degraded


def save_temp_snapshot(city, snapshot, label):
    d = Path(f"cities/{city}/snapshots")
    d.mkdir(parents=True, exist_ok=True)
    filename = f"degraded_{label}pct_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    with open(d / f"{filename}.json", "w") as f:
        json.dump(snapshot, f)
    return filename


def cleanup(city):
    d = Path(f"cities/{city}/snapshots")
    for f in d.glob("degraded_*.json"):
        f.unlink()


def route_single(start, end, city, weight, snapshot=None, eval_snapshot=None):
    """Call a single route endpoint with optional eval_snapshot."""
    endpoint = {"length": "classic", "travel_time": "traffic", "gnn_travel_time": "gnn"}[weight]
    body = {"start": start, "end": end, "city": city.capitalize()}
    if snapshot:
        body["snapshot"] = snapshot
    if eval_snapshot:
        body["eval_snapshot"] = eval_snapshot
    try:
        r = requests.post(f"{API}/route/{endpoint}", json=body, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        feat = data.get("features", [{}])[0]
        props = feat.get("properties", {})
        return {
            "dist_km": round(props["total_length_m"] / 1000, 2) if props.get("total_length_m") else None,
            "time_min": round(props["estimated_time_s"] / 60, 1) if props.get("estimated_time_s") else None,
            "gnn_est": round(props["gnn_predicted_time_s"] / 60, 1) if props.get("gnn_predicted_time_s") else None,
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    city = "vilnius"
    random.seed(42)  # reproducible

    print("=" * 90)
    print("DEGRADED DATA EXPERIMENT (Fair Evaluation)")
    print("=" * 90)
    print()
    print("Method: Route with DEGRADED data, evaluate with FULL data.")
    print("Dijkstra sees partial traffic → picks path → we measure that path's REAL travel time.")
    print("GNN always uses its trained model → picks path → same REAL travel time evaluation.")
    print()

    # Get full snapshot name for evaluation
    snapshots = get_snapshot_list(city)
    if not snapshots:
        print("No snapshots. Create one first.")
        return
    full_snapshot_name = snapshots[-1]  # latest
    full_snapshot = get_latest_snapshot(city)
    if not full_snapshot:
        return

    has_traffic = sum(1 for f in full_snapshot["features"] if f.get("properties", {}).get("current_speed_kph") is not None)
    print(f"Full snapshot: {full_snapshot_name} ({has_traffic} edges with traffic data)")
    print(f"Evaluation: all routes measured against full snapshot")
    print()

    results = []

    for keep_pct in DEGRADATION_LEVELS:
        print(f"--- {keep_pct}% traffic data retained ---")

        if keep_pct == 100:
            routing_snapshot = full_snapshot_name
        else:
            degraded = degrade_snapshot(full_snapshot, keep_pct)
            routing_snapshot = save_temp_snapshot(city, degraded, keep_pct)
            kept = sum(1 for f in degraded["features"] if f.get("properties", {}).get("current_speed_kph") is not None)
            print(f"  Routing with {kept} / {has_traffic} edges having traffic data")

        for route_name, start, end in ROUTES:
            # Traffic-aware: route with degraded data, evaluate with full data
            traffic = route_single(start, end, city, "travel_time",
                                   snapshot=routing_snapshot, eval_snapshot=full_snapshot_name)

            # GNN: always routes with full model, evaluate with full data
            gnn = route_single(start, end, city, "gnn_travel_time",
                               snapshot=full_snapshot_name, eval_snapshot=full_snapshot_name)

            if not traffic or not traffic["time_min"]:
                print(f"  {route_name:40s} | FAILED")
                continue

            t_time = traffic["time_min"]
            t_dist = traffic["dist_km"]
            g_time = gnn["time_min"] if gnn else None
            g_dist = gnn["dist_km"] if gnn else None
            g_est = gnn["gnn_est"] if gnn else None

            winner = ""
            if g_time and t_time:
                if g_time < t_time - 0.05:
                    winner = " << GNN WINS"
                elif t_time < g_time - 0.05:
                    winner = " >> Traffic WINS"
                else:
                    winner = " == TIE"

            results.append({
                "data_pct": keep_pct, "route": route_name,
                "traffic_dist": t_dist, "traffic_time": t_time,
                "gnn_dist": g_dist, "gnn_time": g_time, "gnn_est": g_est,
            })

            g_str = f"{g_time:5.1f}" if g_time else "  N/A"
            g_est_str = f"{g_est}" if g_est else "N/A"
            print(f"  {route_name:40s} | Traffic: {t_dist:5.1f}km {t_time:5.1f}min | "
                  f"GNN: {g_dist or 0:5.1f}km {g_str}min (est {g_est_str}){winner}")

        print()

    cleanup(city)

    # Summary
    print("=" * 90)
    print("SUMMARY: Real travel time (min) — route with degraded data, evaluate with full")
    print("=" * 90)
    print(f"{'Data%':>5s} | ", end="")
    route_names = list(dict.fromkeys(r["route"] for r in results))
    for rn in route_names:
        print(f"{rn[:15]:>15s}      ", end="")
    print()
    print(f"{'':>5s} | ", end="")
    for _ in route_names:
        print(f"{'Traf':>7s} {'GNN':>7s}   ", end="")
    print()
    print("-" * (8 + len(route_names) * 18))

    for pct in DEGRADATION_LEVELS:
        print(f"{pct:4d}% | ", end="")
        for rn in route_names:
            rows = [r for r in results if r["route"] == rn and r["data_pct"] == pct]
            if rows:
                t = rows[0]["traffic_time"] or 0
                g = rows[0]["gnn_time"] or 0
                marker = "*" if g < t - 0.05 else " "
                print(f"{t:7.1f} {g:6.1f}{marker}  ", end="")
            else:
                print(f"{'—':>7s} {'—':>7s}   ", end="")
        print()

    print("\n* = GNN wins (lower real travel time)")

    # Win rate summary
    print()
    print("=" * 90)
    print("GNN WIN RATE by data availability")
    print("=" * 90)
    for pct in DEGRADATION_LEVELS:
        rows = [r for r in results if r["data_pct"] == pct and r["gnn_time"] and r["traffic_time"]]
        wins = sum(1 for r in rows if r["gnn_time"] < r["traffic_time"] - 0.05)
        losses = sum(1 for r in rows if r["traffic_time"] < r["gnn_time"] - 0.05)
        ties = len(rows) - wins - losses
        avg = sum(r["traffic_time"] - r["gnn_time"] for r in rows) / len(rows) if rows else 0
        bar = "█" * wins + "▒" * ties + "░" * losses
        print(f"  {pct:3d}% data: {bar} GNN wins {wins}/{len(rows)}, ties {ties}, avg advantage: {avg:+.1f} min")

    # Save
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"degraded_fair_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: experiments/degraded_fair_{ts}.json")


if __name__ == "__main__":
    main()