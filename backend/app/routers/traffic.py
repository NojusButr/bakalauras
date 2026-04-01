from fastapi import APIRouter, HTTPException

from app.config import CITIES_CONFIG
from app.services.graph_service import load_or_create_city_geojson
from app.services.traffic_service import (
    build_traffic_snapshot,
    list_snapshots,
    load_latest_snapshot,
    load_snapshot,
    save_snapshot,
)

router = APIRouter(prefix="/traffic", tags=["traffic"])


@router.post("/snapshot/{city_name}")
def create_snapshot(city_name: str):
    """
    Trigger a traffic snapshot for a city.
    Makes 2 HTTP requests to HERE (flow + incidents), merges onto OSM edges.
    Fast — typically completes in a few seconds.
    """
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

    place_name, coords = CITIES_CONFIG[city_key]
    geojson = load_or_create_city_geojson(city_key, place_name, coords)

    try:
        snapshot = build_traffic_snapshot(city_key, geojson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot failed: {e}")

    saved_path = save_snapshot(city_key, snapshot)
    snapshot["metadata"]["saved_as"] = saved_path.name
    return snapshot


@router.get("/snapshot/{city_name}/latest")
def get_latest_snapshot(city_name: str):
    """Return the most recently saved traffic snapshot for a city."""
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

    snapshot = load_latest_snapshot(city_key)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No snapshots yet for '{city_name}'. Trigger POST /traffic/snapshot/{city_name} first."
        )
    return snapshot


@router.get("/snapshot/{city_name}/list")
def get_snapshot_list(city_name: str):
    """List all saved snapshot filenames for a city (oldest → newest)."""
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")
    return {"city": city_key, "snapshots": list_snapshots(city_key)}


@router.get("/snapshot/{city_name}/{filename}")
def get_snapshot(city_name: str, filename: str):
    """Load a specific historical snapshot by filename."""
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

    snapshot = load_snapshot(city_key, filename)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Snapshot '{filename}' not found")
    return snapshot