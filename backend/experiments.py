"""
experiments.py — Structured routing experiments for thesis.

Tests three routing methods under normal and crisis conditions.
Outputs comparison tables for thesis results chapter.

Run with FastAPI server active:
  python experiments.py
"""

import json
import time
import csv
import requests
from datetime import datetime
from pathlib import Path

API = "http://localhost:8000"

# Test routes across different parts of Vilnius
TEST_ROUTES = [
    {"name": "West to East (cross-city)", "start": [54.6903, 25.2263], "end": [54.6896, 25.3361]},
    {"name": "North to South (cross-river)", "start": [54.7100, 25.2800], "end": [54.6700, 25.2700]},
    {"name": "Centre to Antakalnis", "start": [54.6870, 25.2790], "end": [54.6960, 25.3300]},
    {"name": "Zirmunai to Naujamiestis", "start": [54.7020, 25.2950], "end": [54.6780, 25.2650]},
    {"name": "Short urban route", "start": [54.6830, 25.2870], "end": [54.6790, 25.2750]},
]

SCENARIOS = [
    {"name": "Normal traffic", "preset": None},
    {"name": "Bridge collapse", "preset": "bridge_collapse"},
    {"name": "All bridges down", "preset": "all_bridges_out"},
    {"name": "Old Town lockdown", "preset": "senamiestis_lockdown"},
    {"name": "Winter storm", "preset": "winter_storm"},
    {"name": "Rush hour cascade", "preset": "rush_hour_cascade"},
]


def run_preset(key, depth=4, decay=0.65):
    try:
        r = requests.post(f"{API}/simulation/preset/vilnius/{key}?propagation_depth={depth}&propagation_decay={decay}")
        if r.ok:
            return r.json().get("metadata", {}).get("saved_as")
    except Exception as e:
        print(f"  ERROR: {e}")
    return None


def compare_routes(start, end, simulation=None):
    body = {"start": start, "end": end, "city": "Vilnius"}
    if simulation:
        body["simulation"] = simulation
    try:
        r = requests.post(f"{API}/route/compare", json=body, timeout=120)
        if r.ok:
            return r.json()
    except Exception as e:
        print(f"  ERROR: {e}")
    return None


def extract(route_data):
    if not route_data:
        return {"km": None, "min": None, "gnn_est": None}
    feats = route_data.get("features", [])
    if not feats:
        return {"km": None, "min": None, "gnn_est": None}
    p = feats[0].get("properties", {})
    d = p.get("total_length_m")
    t = p.get("estimated_time_s")
    g = p.get("gnn_predicted_time_s")
    return {
        "km": round(d / 1000, 2) if d else None,
        "min": round(t / 60, 1) if t else None,
        "gnn_est": round(g / 60, 1) if g else None,
    }


def main():
    print("=" * 90)
    print(f"ROUTING EXPERIMENTS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario['name']} ---")

        sim = None
        if scenario["preset"]:
            sim = run_preset(scenario["preset"])
            if not sim:
                print(f"  FAILED to run preset")
                continue
            print(f"  Simulation: {sim}")

        for route in TEST_ROUTES:
            t0 = time.time()
            data = compare_routes(route["start"], route["end"], sim)
            elapsed = time.time() - t0

            if not data:
                print(f"  {route['name']}: FAILED")
                continue

            c = extract(data.get("classic"))
            t = extract(data.get("traffic"))
            g = extract(data.get("gnn"))

            differ = c["km"] != t["km"] or t["km"] != g["km"]

            print(f"  {route['name']:35s} | Short: {c['km']:>6} km {c['min']:>5} min | "
                  f"Traffic: {t['km']:>6} km {t['min']:>5} min | "
                  f"GNN: {g['km']:>6} km {g['min']:>5} min (est {g['gnn_est']:>5}) | "
                  f"{'DIFFER' if differ else 'same':>6} | {elapsed:.1f}s")

            results.append({
                "scenario": scenario["name"],
                "route": route["name"],
                "shortest_km": c["km"], "shortest_min": c["min"],
                "traffic_km": t["km"], "traffic_min": t["min"],
                "gnn_km": g["km"], "gnn_min": g["min"], "gnn_est_min": g["gnn_est"],
                "routes_differ": differ, "response_s": round(elapsed, 2),
            })

    # Save CSV
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out / f"results_{ts}.csv"

    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    # Summary tables
    print(f"\n{'=' * 90}")
    print(f"DISTANCE COMPARISON (km)")
    print(f"{'=' * 90}")
    print(f"{'Scenario':<22} {'Route':<35} {'Shortest':>8} {'Traffic':>8} {'GNN':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['scenario']:<22} {r['route']:<35} {r['shortest_km'] or '-':>8} {r['traffic_km'] or '-':>8} {r['gnn_km'] or '-':>8}")

    print(f"\n{'=' * 90}")
    print(f"TIME COMPARISON (min)")
    print(f"{'=' * 90}")
    print(f"{'Scenario':<22} {'Route':<35} {'Shortest':>8} {'Traffic':>8} {'GNN':>8} {'GNN est':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['scenario']:<22} {r['route']:<35} {r['shortest_min'] or '-':>8} {r['traffic_min'] or '-':>8} {r['gnn_min'] or '-':>8} {r['gnn_est_min'] or '-':>8}")

    # Key findings
    print(f"\n{'=' * 90}")
    print("KEY FINDINGS")
    print(f"{'=' * 90}")
    normal = [r for r in results if r["scenario"] == "Normal traffic"]
    crisis = [r for r in results if r["scenario"] != "Normal traffic"]
    normal_differ = sum(1 for r in normal if r["routes_differ"])
    crisis_differ = sum(1 for r in crisis if r["routes_differ"])
    print(f"  Normal conditions: {normal_differ}/{len(normal)} routes differ between methods")
    print(f"  Crisis conditions: {crisis_differ}/{len(crisis)} routes differ between methods")

    if normal:
        avg_gnn_err = []
        for r in normal:
            if r["gnn_min"] and r["traffic_min"]:
                avg_gnn_err.append(abs(r["gnn_min"] - r["traffic_min"]))
        if avg_gnn_err:
            print(f"  GNN vs Traffic-aware time difference (normal): avg {sum(avg_gnn_err)/len(avg_gnn_err):.1f} min")

    print(f"\n  Results saved to: {csv_path}")
    print(f"  Total experiments: {len(results)}")


if __name__ == "__main__":
    main()