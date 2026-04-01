"""
Simulation layer for crisis scenario generation with traffic propagation.

When a road is closed or damaged, traffic doesn't just disappear — it pushes
onto neighboring roads. This module models that using BFS-based graph diffusion:
congestion spreads outward hop by hop from the crisis zone, decreasing by
`propagation_decay` at each hop until it reaches `propagation_depth` hops
or drops below a minimum threshold.

Propagation parameters:
  propagation_depth  — max hops congestion spreads (1-6, default 3)
  propagation_decay  — factor per hop, 0-1 (default 0.55 — each hop gets 55% of previous)

Example: bridge destroyed (jam=10)
  Hop 0 (bridge):       jam=10.0, speed=0      — destroyed
  Hop 1 (adjacent):     jam=7.0,  speed ~30%   — severe backup
  Hop 2 (nearby):       jam=3.85, speed ~60%   — moderate congestion
  Hop 3 (surrounding):  jam=2.1,  speed ~75%   — light slowdown
"""

import json
import math
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Literal

from app.config import CITIES_DIR, CITIES_CONFIG
from app.services.traffic_service import load_latest_snapshot, load_snapshot


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _edge_midpoint(feature: dict) -> tuple[float, float] | None:
    geom  = feature.get("geometry", {})
    coords = geom.get("coordinates", [])
    gtype  = geom.get("type")
    if gtype == "LineString" and coords:
        mid = coords[len(coords) // 2]
        return mid[1], mid[0]
    elif gtype == "MultiLineString" and coords:
        flat = [c for part in coords for c in part]
        if flat:
            mid = flat[len(flat) // 2]
            return mid[1], mid[0]
    return None


def _edges_in_radius(features: list[dict], lat: float, lng: float, radius_m: float) -> list[int]:
    return [
        i for i, feat in enumerate(features)
        if (mp := _edge_midpoint(feat)) and _haversine_m(lat, lng, mp[0], mp[1]) <= radius_m
    ]


# ── Spatial adjacency index ───────────────────────────────────────────────────

_ADJACENCY_THRESHOLD_M = 80  # edges whose midpoints are within 80m are considered neighbors

def _build_adjacency(features: list[dict]) -> dict[int, list[int]]:
    """
    Build a neighbor index: for each edge, which other edges are close enough
    to be considered road-network neighbors (connected or parallel nearby road).
    Uses midpoint distance — fast enough for city-scale graphs.
    """
    midpoints = []
    for feat in features:
        mp = _edge_midpoint(feat)
        midpoints.append(mp)  # None if geometry missing

    adj: dict[int, list[int]] = defaultdict(list)

    # O(n²) but only runs once per simulation — ~6000 edges = ~36M comparisons.
    # Fast enough in practice (~1-2s) because haversine is cheap.
    for i in range(len(features)):
        if midpoints[i] is None:
            continue
        lat1, lng1 = midpoints[i]
        for j in range(i + 1, len(features)):
            if midpoints[j] is None:
                continue
            lat2, lng2 = midpoints[j]
            # Quick bbox reject before haversine
            if abs(lat1 - lat2) > 0.002 or abs(lng1 - lng2) > 0.002:
                continue
            dist = _haversine_m(lat1, lng1, lat2, lng2)
            if dist <= _ADJACENCY_THRESHOLD_M:
                adj[i].append(j)
                adj[j].append(i)

    return dict(adj)


# ── Congestion application helpers ────────────────────────────────────────────

def _jam_to_congestion_level(jam: float) -> str:
    if jam >= 7:   return "red"
    if jam >= 4:   return "yellow"
    if jam >= 1:   return "green"
    return "green"


def _apply_jam_to_edge(props: dict, jam_factor: float, hop: int, is_closure: bool = False):
    """
    Apply a jam_factor to an edge's properties, blending with existing data.
    Higher jam always wins — we don't reduce congestion through propagation.
    """
    existing_jam = props.get("jam_factor") or 0
    if jam_factor <= existing_jam:
        return  # don't reduce existing congestion

    free_flow = props.get("free_flow_speed_kph") or 50
    # Speed is proportional to (1 - jam/10) of free-flow
    speed_ratio   = max(0.0, 1.0 - (jam_factor / 10.0))
    new_speed     = round(free_flow * speed_ratio, 1)
    congestion_ratio = round(new_speed / free_flow, 3) if free_flow else 0

    props["jam_factor"]        = round(jam_factor, 2)
    props["current_speed_kph"] = new_speed
    props["congestion_ratio"]  = congestion_ratio
    props["congestion_level"]  = _jam_to_congestion_level(jam_factor)
    props["road_closure"]      = is_closure and hop == 0
    props["simulated"]         = True
    props["propagation_hop"]   = hop  # useful for debugging / GNN feature


# ── Crisis appliers (now return seed jam values for propagation) ──────────────

def _apply_road_closure(features, lat, lng, radius_m) -> tuple[list, dict[int, float]]:
    indices = _edges_in_radius(features, lat, lng, radius_m)
    features = deepcopy(features)
    seed_jams = {}
    for i in indices:
        p = features[i].setdefault("properties", {})
        p["jam_factor"]          = 10.0
        p["current_speed_kph"]   = 0
        p["congestion_ratio"]    = 0.0
        p["congestion_level"]    = "red"
        p["road_closure"]        = True
        p["infrastructure_damaged"] = False
        p["simulated"]           = True
        p["propagation_hop"]     = 0
        seed_jams[i] = 10.0
    return features, seed_jams


def _apply_infrastructure_damage(features, lat, lng, radius_m) -> tuple[list, dict[int, float]]:
    indices = _edges_in_radius(features, lat, lng, radius_m)
    features = deepcopy(features)
    seed_jams = {}
    for i in indices:
        p = features[i].setdefault("properties", {})
        p["jam_factor"]          = 10.0
        p["current_speed_kph"]   = 0
        p["congestion_ratio"]    = 0.0
        p["congestion_level"]    = "red"
        p["road_closure"]        = True
        p["infrastructure_damaged"] = True
        p["simulated"]           = True
        p["propagation_hop"]     = 0
        seed_jams[i] = 10.0
    return features, seed_jams


def _apply_district_congestion(features, lat, lng, radius_m, congestion_factor) -> tuple[list, dict[int, float]]:
    indices = _edges_in_radius(features, lat, lng, radius_m)
    features = deepcopy(features)
    factor = max(0.05, min(1.0, congestion_factor))
    seed_jams = {}
    for i in indices:
        p = features[i].setdefault("properties", {})
        free_flow  = p.get("free_flow_speed_kph") or p.get("current_speed_kph") or 30
        new_speed  = round(free_flow * factor, 1)
        cr         = round(new_speed / free_flow, 3) if free_flow else 0
        jam        = round((1 - cr) * 9, 1)
        p["current_speed_kph"] = new_speed
        p["congestion_ratio"]  = cr
        p["congestion_level"]  = _jam_to_congestion_level(jam)
        p["jam_factor"]        = jam
        p["road_closure"]      = False
        p["infrastructure_damaged"] = False
        p["simulated"]         = True
        p["propagation_hop"]   = 0
        seed_jams[i] = jam
    return features, seed_jams


# ── BFS propagation ────────────────────────────────────────────────────────────

MIN_JAM_TO_PROPAGATE = 0.5  # stop propagating below this jam value

def _propagate_congestion(
    features: list[dict],
    seed_jams: dict[int, float],
    adjacency: dict[int, list[int]],
    max_depth: int,
    decay: float,
) -> int:
    """
    BFS outward from seed edges, applying decaying congestion to neighbors.
    Returns count of additionally affected edges.
    """
    # visited[i] = minimum jam already applied to edge i
    visited: dict[int, float] = {i: jam for i, jam in seed_jams.items()}
    queue: deque[tuple[int, int, float]] = deque()  # (edge_idx, hop, jam)

    for idx, jam in seed_jams.items():
        queue.append((idx, 0, jam))

    extra_affected = 0

    while queue:
        idx, hop, jam = queue.popleft()

        if hop >= max_depth:
            continue

        next_jam = round(jam * decay, 3)
        if next_jam < MIN_JAM_TO_PROPAGATE:
            continue

        for neighbor in adjacency.get(idx, []):
            existing = visited.get(neighbor, 0)
            if next_jam <= existing:
                continue  # already has equal or worse congestion

            visited[neighbor] = next_jam
            p = features[neighbor].setdefault("properties", {})

            was_simulated = p.get("simulated", False)
            _apply_jam_to_edge(p, next_jam, hop + 1)

            if not was_simulated:
                extra_affected += 1

            queue.append((neighbor, hop + 1, next_jam))

    return extra_affected


# ── Main simulation builder ───────────────────────────────────────────────────

def apply_simulation(
    city_name: str,
    base_snapshot: dict,
    events: list[dict],
    scenario_name: str = "custom",
    scenario_description: str = "",
    propagation_depth: int = 3,
    propagation_decay: float = 0.55,
) -> dict:
    """
    Apply crisis events then propagate congestion outward through the graph.

    propagation_depth: how many hops congestion spreads (1-6)
    propagation_decay: congestion multiplier per hop (0.3-0.8)
                       0.3 = sharp dropoff, 0.7 = wide spread
    """
    features = base_snapshot.get("features", [])
    total_direct = 0
    total_propagated = 0
    applied_events = []
    all_seed_jams: dict[int, float] = {}

    # Step 1: Apply direct crisis effects
    for event in events:
        crisis_type = event.get("type")
        center      = event.get("center", [0, 0])
        radius_m    = event.get("radius_m", 500)
        lat, lng    = center[0], center[1]

        if crisis_type == "road_closure":
            features, seed_jams = _apply_road_closure(features, lat, lng, radius_m)
        elif crisis_type == "infrastructure_damage":
            features, seed_jams = _apply_infrastructure_damage(features, lat, lng, radius_m)
        elif crisis_type == "district_congestion":
            factor = event.get("congestion_factor", 0.2)
            features, seed_jams = _apply_district_congestion(features, lat, lng, radius_m, factor)
        else:
            continue

        n = len(seed_jams)
        total_direct += n
        all_seed_jams.update(seed_jams)
        applied_events.append({**event, "edges_affected": n})
        print(f"  [{crisis_type}] ({lat},{lng}) r={radius_m}m → {n} direct edges")

    # Step 2: Build adjacency and propagate
    if propagation_depth > 0 and all_seed_jams:
        print(f"  Building adjacency index for {len(features)} edges...")
        adjacency = _build_adjacency(features)
        print(f"  Propagating congestion (depth={propagation_depth}, decay={propagation_decay})...")
        total_propagated = _propagate_congestion(
            features, all_seed_jams, adjacency, propagation_depth, propagation_decay
        )
        print(f"  Propagation: {total_propagated} additional edges affected")

    base_meta = base_snapshot.get("metadata", {})

    return {
        "type": "FeatureCollection",
        "metadata": {
            "city":                  city_name.lower(),
            "timestamp":             datetime.utcnow().isoformat(),
            "is_simulation":         True,
            "scenario_name":         scenario_name,
            "scenario_description":  scenario_description,
            "base_snapshot_ts":      base_meta.get("timestamp"),
            "events":                applied_events,
            "total_edges_affected":  total_direct + total_propagated,
            "direct_edges":          total_direct,
            "propagated_edges":      total_propagated,
            "propagation_depth":     propagation_depth,
            "propagation_decay":     propagation_decay,
            "osm_edges_total":       len(features),
            "incidents":             base_meta.get("incidents", []),
            "here_flow_segments":    base_meta.get("here_flow_segments", 0),
        },
        "features": features,
    }


def run_preset(
    city_name: str,
    preset_key: str,
    base_snapshot: dict | None = None,
    propagation_depth: int = 3,
    propagation_decay: float = 0.55,
) -> dict:
    city_key     = city_name.lower()
    city_presets = PRESET_SCENARIOS.get(city_key, {})

    if preset_key not in city_presets:
        raise ValueError(f"Unknown preset '{preset_key}' for '{city_key}'. "
                         f"Available: {list(city_presets.keys())}")

    preset = city_presets[preset_key]

    if base_snapshot is None:
        base_snapshot = load_latest_snapshot(city_key)
        if base_snapshot is None:
            raise ValueError(f"No snapshot found for '{city_key}'. Collect one first.")

    print(f"[{city_key}] Running '{preset_key}': {preset['name']}")
    return apply_simulation(
        city_name            = city_key,
        base_snapshot        = base_snapshot,
        events               = preset["events"],
        scenario_name        = preset["name"],
        scenario_description = preset["description"],
        propagation_depth    = propagation_depth,
        propagation_decay    = propagation_decay,
    )


# ── Preset scenarios ──────────────────────────────────────────────────────────

PRESET_SCENARIOS = {
    "vilnius": {
        "bridge_collapse": {
            "name": "Žirmūnai Bridge Collapse",
            "description": "The main bridge connecting Žirmūnai to Antakalnis is destroyed, forcing all traffic onto remaining Neris crossings.",
            "events": [
                {
                    "type": "infrastructure_damage",
                    "center": [54.6946, 25.3018],
                    "radius_m": 200,
                    "description": "Žirmūnų tiltas destroyed"
                }
            ]
        },
        "city_centre_congestion": {
            "name": "Mass City Centre Congestion",
            "description": "A major public event causes extreme congestion across the entire city centre.",
            "events": [
                {
                    "type": "district_congestion",
                    "center": [54.6872, 25.2797],
                    "radius_m": 2500,
                    "congestion_factor": 0.1,
                    "description": "City centre gridlock"
                }
            ]
        },
        "multi_closure": {
            "name": "Multi-Road Crisis",
            "description": "Both main Neris bridges destroyed simultaneously, forcing all river crossings onto outer routes.",
            "events": [
                {
                    "type": "infrastructure_damage",
                    "center": [54.6946, 25.3018],
                    "radius_m": 200,
                    "description": "Žirmūnų tiltas destroyed"
                },
                {
                    "type": "infrastructure_damage",
                    "center": [54.6890, 25.2882],
                    "radius_m": 150,
                    "description": "Mindaugo tiltas destroyed"
                },
                {
                    "type": "district_congestion",
                    "center": [54.6872, 25.2797],
                    "radius_m": 2000,
                    "congestion_factor": 0.2,
                    "description": "Centre congestion from mass rerouting"
                }
            ]
        },
        "eastern_district_damage": {
            "name": "Eastern Infrastructure Damage",
            "description": "Critical road infrastructure in the eastern districts is destroyed.",
            "events": [
                {
                    "type": "infrastructure_damage",
                    "center": [54.6946, 25.3018],
                    "radius_m": 200,
                    "description": "Žirmūnų tiltas destroyed"
                },
                {
                    "type": "district_congestion",
                    "center": [54.6920, 25.3200],
                    "radius_m": 2000,
                    "congestion_factor": 0.25,
                    "description": "Eastern district congestion"
                }
            ]
        }
    }
}


# ── Simulation persistence ────────────────────────────────────────────────────

def save_simulation(city_name: str, simulation: dict) -> Path:
    city_key  = city_name.lower()
    sim_dir   = CITIES_DIR / city_key / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    scenario  = simulation["metadata"]["scenario_name"].replace(" ", "_").lower()
    timestamp = simulation["metadata"]["timestamp"].replace(":", "-")
    filepath  = sim_dir / f"{scenario}__{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(simulation, f, indent=2)

    print(f"[{city_key}] Simulation saved → {filepath}")
    return filepath


def list_simulations(city_name: str) -> list[dict]:
    sim_dir = CITIES_DIR / city_name.lower() / "simulations"
    if not sim_dir.exists():
        return []
    result = []
    for f in sorted(sim_dir.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            result.append({"filename": f.stem, **data.get("metadata", {})})
        except Exception:
            continue
    return result


def load_simulation(city_name: str, filename: str) -> dict | None:
    filepath = CITIES_DIR / city_name.lower() / "simulations" / f"{filename}.json"
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)