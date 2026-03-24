import json
import time
import requests
from datetime import datetime
from pathlib import Path
from shapely.geometry import LineString, shape

from app.config import TOMTOM_API, CITIES_DIR


# ── Config ────────────────────────────────────────────────────────────────────

REQUEST_DELAY_S = 0.05        # 50ms between requests (~20 req/s, well under limits)
MATCH_TOLERANCE_DEG = 0.005   # ~500m — max distance to match a TomTom seg to OSM edge
EARLY_STOP_WINDOW = 150       # stop if no new unique segments found in this many requests

# Sample these highway types — ones TomTom is likely to monitor.
# Ordered from most → least important so we prioritize major roads first.
SAMPLED_HIGHWAY_TYPES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential",
    "unclassified",
    "living_street",
    "service",
}


# ── FRC → free-flow fallback (only used if TomTom doesn't return freeFlowSpeed) ──
# In practice the API always returns it, this is just a safety net
FRC_FREE_FLOW = {
    "FRC0": 120,  # Motorway
    "FRC1": 100,  # Major road
    "FRC2": 90,
    "FRC3": 70,
    "FRC4": 50,
    "FRC5": 40,
    "FRC6": 30,
    "FRC7": 20,
}


def _congestion_level(current: float, free_flow: float) -> str:
    if not free_flow or free_flow <= 0:
        return "unknown"
    ratio = current / free_flow
    if ratio >= 0.8:
        return "green"
    elif ratio >= 0.5:
        return "yellow"
    else:
        return "red"


# ── OSM edge midpoint sampler ─────────────────────────────────────────────────

# Priority order for highway types — major roads first so we find the most
# TomTom-monitored segments early and can benefit from early stopping sooner.
_HIGHWAY_PRIORITY = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "unclassified",
    "living_street", "service",
]

def _edge_midpoints(geojson: dict) -> list[tuple[float, float]]:
    """
    Extract the midpoint (lat, lng) of every OSM edge, ordered by road importance.
    Filters to SAMPLED_HIGHWAY_TYPES only — no point sampling footways/cycleways.
    Deduplicates midpoints that are suspiciously close (within ~50m) to avoid
    redundant requests on parallel edges.
    """
    # Group features by highway type so we can sort by priority
    by_type: dict[str, list] = {h: [] for h in _HIGHWAY_PRIORITY}
    other = []

    for feat in geojson["features"]:
        hw = feat.get("properties", {}).get("highway")
        # highway can be a list (OSM quirk) — take first value
        if isinstance(hw, list):
            hw = hw[0] if hw else None
        if hw in SAMPLED_HIGHWAY_TYPES:
            by_type.get(hw, other).append(feat)

    ordered_features = []
    for hw in _HIGHWAY_PRIORITY:
        ordered_features.extend(by_type[hw])

    # Extract midpoints, dedup within ~50m
    points: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()  # bucketed to ~50m grid

    for feat in ordered_features:
        geom = feat.get("geometry", {})
        try:
            line = shape(geom)
            mp = line.interpolate(0.5, normalized=True)
            lat, lng = mp.y, mp.x
        except Exception:
            continue

        # Bucket to ~0.0005 deg ≈ 50m to skip near-duplicate midpoints
        bucket = (round(lat, 4), round(lng, 4))
        if bucket in seen:
            continue
        seen.add(bucket)
        points.append((lat, lng))

    return points


# ── TomTom Flow Segment API ───────────────────────────────────────────────────

def _fetch_flow_segment(lat: float, lng: float) -> dict | None:
    """
    Hit TomTom flowSegmentData for a single point.
    Returns the flowSegmentData dict or None on failure.
    """
    url = (
        f"https://api.tomtom.com/traffic/services/4"
        f"/flowSegmentData/absolute/10/json"
    )
    params = {
        "key": TOMTOM_API,
        "point": f"{lat},{lng}",
        "unit": "kmph",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        print(f"Request error at ({lat},{lng}): {e}")
        return None

    if r.status_code == 404:
        return None  # No road near this point — normal
    if r.status_code == 403:
        raise RuntimeError(f"TomTom quota exhausted: {r.text[:100]}")
    if r.status_code != 200:
        print(f"TomTom error at ({lat},{lng}): {r.status_code} {r.text[:100]}")
        return None

    return r.json().get("flowSegmentData")


# ── Deduplication ─────────────────────────────────────────────────────────────

def _segment_key(flow: dict) -> str:
    """
    Stable dedup key from the segment's coordinate list.
    TomTom returns the same coordinates for the same physical road segment
    regardless of which grid point triggered it.
    """
    coords = flow.get("coordinates", {}).get("coordinate", [])
    if not coords:
        return ""
    # Use first + last coordinate as a lightweight key
    first = coords[0]
    last = coords[-1]
    return f"{first['latitude']:.5f},{first['longitude']:.5f}|{last['latitude']:.5f},{last['longitude']:.5f}"


def _flow_to_segment(flow: dict) -> dict:
    """
    Convert a TomTom flowSegmentData response into our internal segment format:
    {
      "geometry": shapely LineString (lng, lat),
      "props": { all traffic fields }
    }
    """
    coords = flow.get("coordinates", {}).get("coordinate", [])
    # TomTom returns (lat, lng) — convert to (lng, lat) for GeoJSON convention
    line_coords = [(c["longitude"], c["latitude"]) for c in coords]

    current_speed = flow.get("currentSpeed", 0)
    free_flow_speed = flow.get("freeFlowSpeed") or FRC_FREE_FLOW.get(flow.get("frc", ""), 50)
    confidence = flow.get("confidence")
    road_closure = flow.get("roadClosure", False)
    current_travel_time = flow.get("currentTravelTime")
    free_flow_travel_time = flow.get("freeFlowTravelTime")
    frc = flow.get("frc", "")

    congestion_ratio = round(current_speed / free_flow_speed, 3) if free_flow_speed else None
    congestion = _congestion_level(current_speed, free_flow_speed)

    # Travel time delay — useful GNN feature
    travel_time_delay = (
        round(current_travel_time - free_flow_travel_time, 1)
        if current_travel_time is not None and free_flow_travel_time is not None
        else None
    )

    return {
        "geometry": LineString(line_coords) if len(line_coords) >= 2 else None,
        "props": {
            "current_speed_kph": current_speed,
            "free_flow_speed_kph": free_flow_speed,
            "congestion_ratio": congestion_ratio,
            "congestion_level": congestion,
            "road_closure": road_closure,
            "confidence": confidence,
            "current_travel_time_s": current_travel_time,
            "free_flow_travel_time_s": free_flow_travel_time,
            "travel_time_delay_s": travel_time_delay,
            "frc": frc,  # Functional Road Class — useful for GNN node features
        },
    }


# ── Main snapshot builder ─────────────────────────────────────────────────────

def build_traffic_snapshot(city_name: str, geojson: dict) -> dict:
    """
    1. Build a grid over the city bounding box
    2. Sample TomTom flowSegmentData at each grid point
    3. Deduplicate segments by their coordinate fingerprint
    4. Match each OSM edge to its nearest TomTom segment
    5. Return annotated GeoJSON FeatureCollection

    Per-edge fields (GNN-ready):
      From OSM  : osmid, name, length, highway, maxspeed, lanes, oneway, geometry
      From TomTom: current_speed_kph, free_flow_speed_kph, congestion_ratio,
                   congestion_level, road_closure, confidence,
                   current_travel_time_s, free_flow_travel_time_s,
                   travel_time_delay_s, frc
    """
    city_lower = city_name.lower()

    # 1. Extract ordered edge midpoints from OSM data
    sample_points = _edge_midpoints(geojson)
    print(
        f"[{city_lower}] Sampling {len(sample_points)} OSM edge midpoints "
        f"(early stop after {EARLY_STOP_WINDOW} requests with no new segments)..."
    )

    # 2. Fetch + deduplicate + early stop
    seen_keys: set[str] = set()
    traffic_segments: list[dict] = []
    requests_since_new = 0
    errors = 0
    total_requests = 0

    for i, (lat, lng) in enumerate(sample_points):
        try:
            flow = _fetch_flow_segment(lat, lng)
        except RuntimeError as e:
            print(f"[{city_lower}] Aborting — {e}")
            print(f"[{city_lower}] Saving partial snapshot with {len(traffic_segments)} segments collected so far")
            break
        flow = _fetch_flow_segment(lat, lng)
        total_requests += 1

        if flow is None:
            errors += 1
            requests_since_new += 1
        else:
            key = _segment_key(flow)
            if key and key not in seen_keys:
                seen_keys.add(key)
                seg = _flow_to_segment(flow)
                if seg["geometry"] is not None:
                    traffic_segments.append(seg)
                requests_since_new = 0  # reset window — found something new
            else:
                requests_since_new += 1

        if REQUEST_DELAY_S:
            time.sleep(REQUEST_DELAY_S)

        if (i + 1) % 100 == 0:
            print(
                f"  {i+1}/{len(sample_points)} sampled, "
                f"{len(traffic_segments)} unique segments, "
                f"{requests_since_new} since last new"
            )

        # Early stop — no new segments for a while, we've likely found everything
        if requests_since_new >= EARLY_STOP_WINDOW:
            print(
                f"[{city_lower}] Early stop at {total_requests} requests — "
                f"no new segments in last {EARLY_STOP_WINDOW} requests"
            )
            break

    print(
        f"[{city_lower}] Done. {total_requests} requests → "
        f"{len(traffic_segments)} unique segments ({errors} no-road responses skipped)"
    )

    # 4. Match OSM edges → nearest TomTom segment
    timestamp = datetime.utcnow().isoformat()
    annotated_features = []
    matched = 0

    for feat in geojson["features"]:
        geom = feat.get("geometry", {})
        props = dict(feat.get("properties", {}))

        try:
            line = shape(geom)
            midpoint = line.interpolate(0.5, normalized=True)
        except Exception:
            annotated_features.append(feat)
            continue

        best = None
        best_dist = float("inf")
        for seg in traffic_segments:
            dist = midpoint.distance(seg["geometry"])
            if dist < best_dist:
                best_dist = dist
                best = seg

        if best and best_dist < MATCH_TOLERANCE_DEG:
            props.update(best["props"])
            matched += 1
        else:
            props.update({
                "current_speed_kph": None,
                "free_flow_speed_kph": None,
                "congestion_ratio": None,
                "congestion_level": "unknown",
                "road_closure": None,
                "confidence": None,
                "current_travel_time_s": None,
                "free_flow_travel_time_s": None,
                "travel_time_delay_s": None,
                "frc": None,
            })

        annotated_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        })

    print(f"[{city_lower}] Matched {matched}/{len(annotated_features)} OSM edges to TomTom segments")

    return {
        "type": "FeatureCollection",
        "metadata": {
            "city": city_lower,
            "timestamp": timestamp,
            "osm_edge_midpoints_sampled": total_requests,
            "unique_tomtom_segments": len(traffic_segments),
            "osm_edges_total": len(annotated_features),
            "osm_edges_matched": matched,
            "no_road_responses_skipped": errors,
        },
        "features": annotated_features,
    }


# ── Snapshot persistence ──────────────────────────────────────────────────────

def save_snapshot(city_name: str, snapshot: dict) -> Path:
    city_lower = city_name.lower()
    snapshots_dir = CITIES_DIR / city_lower / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = snapshot["metadata"]["timestamp"].replace(":", "-")
    filepath = snapshots_dir / f"{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(snapshot, f)

    print(f"[{city_lower}] Snapshot saved → {filepath}")
    return filepath


def list_snapshots(city_name: str) -> list[str]:
    snapshots_dir = CITIES_DIR / city_name.lower() / "snapshots"
    if not snapshots_dir.exists():
        return []
    return [f.stem for f in sorted(snapshots_dir.glob("*.json"))]


def load_snapshot(city_name: str, filename: str) -> dict | None:
    filepath = CITIES_DIR / city_name.lower() / "snapshots" / f"{filename}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def load_latest_snapshot(city_name: str) -> dict | None:
    snapshots_dir = CITIES_DIR / city_name.lower() / "snapshots"
    if not snapshots_dir.exists():
        return None
    files = sorted(snapshots_dir.glob("*.json"))
    if not files:
        return None
    with open(files[-1], "r") as f:
        return json.load(f)


# ── Legacy single-point helper ────────────────────────────────────────────────

def get_traffic_for_point(lat: float, lng: float) -> dict | None:
    return _fetch_flow_segment(lat, lng)