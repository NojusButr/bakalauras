import json
import requests
from datetime import datetime
from pathlib import Path
from shapely.geometry import LineString, shape
from shapely.strtree import STRtree

from app.config import HERE_API_KEY, CITIES_DIR, CITIES_CONFIG, CITY_RADIUS_M


HERE_FLOW_URL      = "https://data.traffic.hereapi.com/v7/flow"
HERE_INCIDENTS_URL = "https://data.traffic.hereapi.com/v7/incidents"


def _city_circle(city_key: str) -> str:
    """Return HERE circle filter string for a city, e.g. 'circle:54.6872,25.2797;r=15000'"""
    _, (lat, lng) = CITIES_CONFIG[city_key]
    radius = CITY_RADIUS_M.get(city_key, 10000)
    return f"circle:{lat},{lng};r={radius}"


def _fetch_flow(city_key: str) -> list[dict]:
    """
    Fetch HERE /v7/flow for a city — one request covers the whole city.
    Returns list of result dicts with 'location' and 'currentFlow'.
    """
    params = {
        "in": _city_circle(city_key),
        "locationReferencing": "shape",
        "functionalClasses": "1,2,3,4,5",
        "advancedFeatures": "deepCoverage",
        "apiKey": HERE_API_KEY,
    }
    r = requests.get(HERE_FLOW_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def _fetch_incidents(city_key: str) -> list[dict]:
    """
    Fetch HERE /v7/incidents for a city — one request covers the whole city.
    Returns list of result dicts with 'location' and 'incidentDetails'.
    """
    params = {
        "in": _city_circle(city_key),
        "locationReferencing": "shape",
        "apiKey": HERE_API_KEY,
    }
    r = requests.get(HERE_INCIDENTS_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def _congestion_level(jam_factor: float) -> str:
    """Convert HERE jamFactor (0-10) to green/yellow/red."""
    if jam_factor is None:
        return "unknown"
    if jam_factor <= 2.0:
        return "green"
    elif jam_factor <= 5.0:
        return "yellow"
    else:
        return "red"


def _links_to_linestring(links: list[dict]):
    """Flatten HERE shape links into a single shapely LineString (lng, lat)."""
    coords = []
    for link in links:
        for pt in link.get("points", []):
            coords.append((pt["lng"], pt["lat"]))
    deduped = [coords[0]] if coords else []
    for c in coords[1:]:
        if c != deduped[-1]:
            deduped.append(c)
    return LineString(deduped) if len(deduped) >= 2 else None


def _parse_flow_results(results: list[dict]) -> list[dict]:
    """
    Convert HERE flow results into internal segment list:
    [{ "geometry": LineString, "props": {...} }, ...]
    """
    segments = []
    for result in results:
        links = result.get("location", {}).get("shape", {}).get("links", [])
        flow  = result.get("currentFlow", {})

        geom = _links_to_linestring(links)
        if geom is None:
            continue

        speed_ms        = flow.get("speed")
        speed_uncapped  = flow.get("speedUncapped")
        free_flow_ms    = flow.get("freeFlow")
        jam_factor      = flow.get("jamFactor")
        confidence      = flow.get("confidence")
        traversability  = flow.get("traversability", "open")
        road_closure    = traversability != "open"

        speed_kph      = round(speed_ms * 3.6, 1)      if speed_ms      is not None else None
        free_flow_kph  = round(free_flow_ms * 3.6, 1)  if free_flow_ms  is not None else None
        congestion_ratio = (
            round(speed_ms / free_flow_ms, 3)
            if speed_ms is not None and free_flow_ms and free_flow_ms > 0
            else None
        )

        functional_class = links[0].get("functionalClass") if links else None

        segments.append({
            "geometry": geom,
            "props": {
                "current_speed_kph":   speed_kph,
                "free_flow_speed_kph": free_flow_kph,
                "congestion_ratio":    congestion_ratio,
                "congestion_level":    _congestion_level(jam_factor),
                "jam_factor":          jam_factor,
                "confidence":          confidence,
                "road_closure":        road_closure,
                "functional_class":    functional_class,
            }
        })
    return segments


def _parse_incident_results(results: list[dict]) -> list[dict]:
    """
    Convert HERE incident results into internal segment list.
    Each incident has geometry + incident metadata.
    """
    incidents = []
    for result in results:
        links   = result.get("location", {}).get("shape", {}).get("links", [])
        details = result.get("incidentDetails", {})

        geom = _links_to_linestring(links)
        if geom is None:
            continue

        incidents.append({
            "geometry": geom,
            "props": {
                "incident_id":          details.get("id"),
                "incident_type":        details.get("type"),
                "incident_description": details.get("description", {}).get("value"),
                "incident_criticality": details.get("criticality"),
                "incident_road_closed": details.get("roadClosed", False),
                "incident_start_time":  details.get("startTime"),
                "incident_end_time":    details.get("endTime"),
            }
        })
    return incidents


# ── Spatial matching ──────────────────────────────────────────────────────────

MATCH_TOLERANCE_DEG = 0.005


def _match_segments_to_edges(
    features: list[dict],
    traffic_segments: list[dict],
    null_traffic_props: dict,
) -> list[dict]:
    """
    For each GeoJSON feature (OSM edge), find the nearest traffic segment
    using an STRtree spatial index for efficiency.
    Returns annotated feature list.
    """
    if not traffic_segments:
        return [
            {**feat, "properties": {**feat.get("properties", {}), **null_traffic_props}}
            for feat in features
        ]

    seg_geoms = [s["geometry"] for s in traffic_segments]
    tree = STRtree(seg_geoms)

    annotated = []
    matched = 0

    for feat in features:
        geom = feat.get("geometry", {})
        props = dict(feat.get("properties", {}))

        try:
            line = shape(geom)
            midpoint = line.interpolate(0.5, normalized=True)
        except Exception:
            annotated.append(feat)
            continue

        nearest_idx = tree.nearest(midpoint)
        nearest_geom = seg_geoms[nearest_idx]
        dist = midpoint.distance(nearest_geom)

        if dist < MATCH_TOLERANCE_DEG:
            props.update(traffic_segments[nearest_idx]["props"])
            matched += 1
        else:
            props.update(null_traffic_props)

        annotated.append({"type": "Feature", "geometry": geom, "properties": props})

    print(f"  Matched {matched}/{len(features)} OSM edges")
    return annotated


NULL_TRAFFIC_PROPS = {
    "current_speed_kph":   None,
    "free_flow_speed_kph": None,
    "congestion_ratio":    None,
    "congestion_level":    "unknown",
    "jam_factor":          None,
    "confidence":          None,
    "road_closure":        None,
    "functional_class":    None,
}

NULL_INCIDENT_PROPS = {
    "incident_id":          None,
    "incident_type":        None,
    "incident_description": None,
    "incident_criticality": None,
    "incident_road_closed": None,
    "incident_start_time":  None,
    "incident_end_time":    None,
}


def build_traffic_snapshot(city_name: str, geojson: dict) -> dict:
    """
    Fetch HERE flow + incident data for a city,
    match onto OSM GeoJSON edges via spatial index, return annotated snapshot.

    GNN-ready fields per edge:
      OSM:      osmid, name, length, highway, maxspeed, lanes, oneway
      Flow:     current_speed_kph, free_flow_speed_kph, congestion_ratio,
                congestion_level, jam_factor, confidence, road_closure,
                functional_class
      Incident: incident_type, incident_criticality, incident_road_closed,
                incident_start_time, incident_end_time, incident_description
    """
    city_key = city_name.lower()
    print(f"[{city_key}] Fetching HERE flow data...")
    flow_results     = _fetch_flow(city_key)
    print(f"[{city_key}] Fetching HERE incident data...")
    incident_results = _fetch_incidents(city_key)

    print(f"[{city_key}] Received {len(flow_results)} flow segments, {len(incident_results)} incidents")

    flow_segments     = _parse_flow_results(flow_results)
    incident_segments = _parse_incident_results(incident_results)

    print(f"[{city_key}] Matching flow data to OSM edges...")
    features = geojson.get("features", [])
    annotated = _match_segments_to_edges(features, flow_segments, NULL_TRAFFIC_PROPS)

    incident_points = []
    for seg in incident_segments:
        centroid = seg["geometry"].centroid
        incident_points.append({
            "lat": round(centroid.y, 6),
            "lng": round(centroid.x, 6),
            **seg["props"],
        })

    timestamp = datetime.utcnow().isoformat()

    return {
        "type": "FeatureCollection",
        "metadata": {
            "city":               city_key,
            "timestamp":          timestamp,
            "here_flow_segments": len(flow_segments),
            "here_incidents":     len(incident_points),
            "osm_edges_total":    len(annotated),
            "incidents":          incident_points,   
        },
        "features": annotated,
    }


def save_snapshot(city_name: str, snapshot: dict) -> Path:
    city_key = city_name.lower()
    snapshots_dir = CITIES_DIR / city_key / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = snapshot["metadata"]["timestamp"].replace(":", "-")
    filepath  = snapshots_dir / f"{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"[{city_key}] Snapshot saved → {filepath}")
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