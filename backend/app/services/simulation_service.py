"""
Simulation service — crisis scenarios with distance-ring propagation.

Event types: construction, congestion, damage
Each has speed_reduction_pct (0=impassable, 100=normal).
Damage always forced to 0.
"""

import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from app.config import CITIES_DIR, CITIES_CONFIG
from app.services.traffic_service import load_latest_snapshot, load_snapshot
from app.services.simulation_presets import PRESET_SCENARIOS



def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _edge_midpoint(feature):
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [])
    gtype = geom.get("type")
    if gtype == "LineString" and coords:
        mid = coords[len(coords) // 2]
        return mid[1], mid[0]
    elif gtype == "MultiLineString" and coords:
        flat = [c for part in coords for c in part]
        if flat:
            mid = flat[len(flat) // 2]
            return mid[1], mid[0]
    return None


def _edges_in_radius(features, lat, lng, radius_m):
    return [
        i for i, feat in enumerate(features)
        if (mp := _edge_midpoint(feat)) and _haversine_m(lat, lng, mp[0], mp[1]) <= radius_m
    ]


def _jam_to_congestion_level(jam):
    if jam >= 7:   return "red"
    if jam >= 4:   return "yellow"
    if jam >= 1:   return "green"
    return "green"



def _apply_event(features, event):
    event_type = event.get("type", "congestion")
    center = event.get("center", [0, 0])
    radius_m = event.get("radius_m", 500)
    lat, lng = center[0], center[1]

    if event_type == "damage":
        speed_pct = 0
    else:
        speed_pct = max(0, min(100, event.get("speed_reduction_pct", 50)))

    indices = _edges_in_radius(features, lat, lng, radius_m)
    seed_jams = {}
    is_impassable = (speed_pct == 0)

    for i in indices:
        p = features[i].setdefault("properties", {})
        free_flow = p.get("free_flow_speed_kph") or p.get("current_speed_kph") or 50

        if is_impassable:
            p["jam_factor"] = 10.0
            p["current_speed_kph"] = 0
            p["congestion_ratio"] = 0.0
            p["congestion_level"] = "red"
            p["road_closure"] = True
            p["impassable"] = True
        else:
            factor = speed_pct / 100.0
            new_speed = round(free_flow * factor, 1)
            cr = round(factor, 3)
            jam = round((1 - cr) * 10, 2)
            jam = min(jam, 9.9)
            p["jam_factor"] = jam
            p["current_speed_kph"] = new_speed
            p["congestion_ratio"] = cr
            p["congestion_level"] = _jam_to_congestion_level(jam)
            p["road_closure"] = False
            p["impassable"] = False

        p["infrastructure_damaged"] = (event_type == "damage")
        p["construction"] = (event_type == "construction")
        p["event_type"] = event_type
        p["simulated"] = True
        p["propagation_hop"] = 0
        p["speed_reduction_pct"] = speed_pct
        seed_jams[i] = p["jam_factor"]

    return seed_jams



MIN_JAM_TO_PROPAGATE = 0.5

def _propagate_congestion(features, seed_jams, max_depth, decay):
    if not seed_jams:
        return 0

    max_seed_jam = max(seed_jams.values())
    seed_centers = []
    for idx in seed_jams:
        mp = _edge_midpoint(features[idx])
        if mp:
            seed_centers.append(mp)
    if not seed_centers:
        return 0

    avg_lat = sum(c[0] for c in seed_centers) / len(seed_centers)
    avg_lng = sum(c[1] for c in seed_centers) / len(seed_centers)

    max_seed_dist = 0
    for idx in seed_jams:
        mp = _edge_midpoint(features[idx])
        if mp:
            d = _haversine_m(avg_lat, avg_lng, mp[0], mp[1])
            if d > max_seed_dist:
                max_seed_dist = d

    base_ring = max(max_seed_dist, 100)
    ring_step = base_ring * 0.8

    seed_set = set(seed_jams.keys())
    affected_set = set(seed_set)
    extra_affected = 0

    for depth in range(1, max_depth + 1):
        ring_jam = round(max_seed_jam * (decay ** depth), 3)
        if ring_jam < MIN_JAM_TO_PROPAGATE:
            break

        outer_radius = base_ring + ring_step * depth

        for i, feat in enumerate(features):
            if i in affected_set:
                continue
            mp = _edge_midpoint(feat)
            if not mp:
                continue
            if _haversine_m(avg_lat, avg_lng, mp[0], mp[1]) > outer_radius:
                continue

            p = feat.setdefault("properties", {})
            existing_jam = p.get("jam_factor") or 0
            was_simulated = p.get("simulated", False)

            if ring_jam > existing_jam:
                free_flow = p.get("free_flow_speed_kph") or 50
                speed_ratio = max(0.0, 1.0 - (ring_jam / 10.0))
                new_speed = round(free_flow * speed_ratio, 1)
                congestion_ratio = round(new_speed / free_flow, 3) if free_flow else 0
                p["jam_factor"] = round(ring_jam, 2)
                p["current_speed_kph"] = new_speed
                p["congestion_ratio"] = congestion_ratio
                p["congestion_level"] = _jam_to_congestion_level(ring_jam)
                p["simulated"] = True
                p["propagation_hop"] = depth
                if not was_simulated:
                    extra_affected += 1
            elif not was_simulated:
                p["simulated"] = True
                p["propagation_hop"] = depth
                extra_affected += 1

            affected_set.add(i)

    return extra_affected



def apply_simulation(
    city_name, base_snapshot, events,
    scenario_name="custom", scenario_description="",
    propagation_depth=3, propagation_decay=0.55,
):
    features = deepcopy(base_snapshot.get("features", []))
    total_direct = 0
    total_propagated = 0
    applied_events = []
    propagation_seeds = {}

    for event in events:
        seed_jams = _apply_event(features, event)
        n = len(seed_jams)
        total_direct += n

        event_type = event.get("type", "congestion")
        speed_pct = event.get("speed_reduction_pct", 50)
        propagation_seeds.update(seed_jams)

        applied_events.append({**event, "edges_affected": n})

    if propagation_depth > 0 and propagation_seeds:
        total_propagated = _propagate_congestion(
            features, propagation_seeds, propagation_depth, propagation_decay
        )

    base_meta = base_snapshot.get("metadata", {})
    return {
        "type": "FeatureCollection",
        "metadata": {
            "city": city_name.lower(),
            "timestamp": datetime.utcnow().isoformat(),
            "is_simulation": True,
            "scenario_name": scenario_name,
            "scenario_description": scenario_description,
            "base_snapshot_ts": base_meta.get("timestamp"),
            "events": applied_events,
            "total_edges_affected": total_direct + total_propagated,
            "direct_edges": total_direct,
            "propagated_edges": total_propagated,
            "propagation_depth": propagation_depth,
            "propagation_decay": propagation_decay,
            "osm_edges_total": len(features),
            "incidents": base_meta.get("incidents", []),
            "here_flow_segments": base_meta.get("here_flow_segments", 0),
        },
        "features": features,
    }


def preview_simulation(city_name, base_snapshot, events, propagation_depth=3, propagation_decay=0.55):
    features = deepcopy(base_snapshot.get("features", []))
    propagation_seeds = {}
    total_direct = 0
    applied_events = []

    for event in events:
        seed_jams = _apply_event(features, event)
        total_direct += len(seed_jams)
        # Always propagate — _propagate_congestion handles the cutoff via MIN_JAM_TO_PROPAGATE
        propagation_seeds.update(seed_jams)
        applied_events.append({**event, "edges_affected": len(seed_jams)})

    total_propagated = 0
    if propagation_depth > 0 and propagation_seeds:
        total_propagated = _propagate_congestion(
            features, propagation_seeds, propagation_depth, propagation_decay
        )

    affected = [f for f in features if f.get("properties", {}).get("simulated")]
    return {
        "type": "FeatureCollection",
        "metadata": {
            "city": city_name.lower(),
            "timestamp": datetime.utcnow().isoformat(),
            "is_simulation": True, "is_preview": True,
            "events": applied_events,
            "direct_edges": total_direct,
            "propagated_edges": total_propagated,
            "total_edges_affected": total_direct + total_propagated,
            "propagation_depth": propagation_depth,
            "propagation_decay": propagation_decay,
        },
        "features": affected,
    }


def run_preset(city_name, preset_key, base_snapshot=None, propagation_depth=3, propagation_decay=0.55):
    city_key = city_name.lower()
    # Check built-in presets first, then user presets
    city_presets = PRESET_SCENARIOS.get(city_key, {})
    preset = city_presets.get(preset_key)
    if not preset:
        user_presets = load_user_presets(city_key)
        preset = user_presets.get(preset_key)
    if not preset:
        raise ValueError(f"Unknown preset '{preset_key}'")

    if base_snapshot is None:
        base_snapshot = load_latest_snapshot(city_key)
        if base_snapshot is None:
            raise ValueError(f"No snapshot for '{city_key}'")

    return apply_simulation(
        city_key, base_snapshot, preset["events"],
        preset["name"], preset.get("description", ""),
        propagation_depth, propagation_decay,
    )


# ── User presets (saved by users, per city) ───────────────────────────────────

def _user_presets_dir(city_name: str) -> Path:
    d = CITIES_DIR / city_name.lower() / "user_presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_user_presets(city_name: str) -> dict:
    """Load all user-created presets for a city. Returns {key: preset_dict}."""
    d = _user_presets_dir(city_name)
    presets = {}
    for f in sorted(d.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            presets[f.stem] = data
        except Exception:
            continue
    return presets


def save_user_preset(city_name: str, key: str, name: str, description: str, events: list) -> dict:
    """Save a user-created preset. Returns the saved preset data."""
    d = _user_presets_dir(city_name)
    safe_key = key.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    preset = {
        "name": name,
        "description": description,
        "events": events,
        "created": datetime.utcnow().isoformat(),
        "user_created": True,
    }
    filepath = d / f"{safe_key}.json"
    with open(filepath, "w") as f:
        json.dump(preset, f, indent=2)
    return preset


def delete_user_preset(city_name: str, key: str) -> bool:
    """Delete a user-created preset. Returns True if deleted."""
    d = _user_presets_dir(city_name)
    filepath = d / f"{key}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def get_all_presets(city_name: str) -> list[dict]:
    """Get merged list of built-in + user presets for a city."""
    city_key = city_name.lower()
    result = []

    # Built-in presets
    for k, v in PRESET_SCENARIOS.get(city_key, {}).items():
        result.append({
            "key": k,
            "name": v["name"],
            "description": v.get("description", ""),
            "event_count": len(v["events"]),
            "user_created": False,
        })

    # User presets
    for k, v in load_user_presets(city_key).items():
        result.append({
            "key": k,
            "name": v.get("name", k),
            "description": v.get("description", ""),
            "event_count": len(v.get("events", [])),
            "user_created": True,
        })

    return result


# ── Simulation persistence ────────────────────────────────────────────────────

def save_simulation(city_name, simulation):
    city_key = city_name.lower()
    sim_dir = CITIES_DIR / city_key / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    scenario = simulation["metadata"]["scenario_name"].replace(" ", "_").lower()
    timestamp = simulation["metadata"]["timestamp"].replace(":", "-")
    filepath = sim_dir / f"{scenario}__{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(simulation, f, indent=2)
    return filepath


def list_simulations(city_name):
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


def load_simulation(city_name, filename):
    filepath = CITIES_DIR / city_name.lower() / "simulations" / f"{filename}.json"
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)