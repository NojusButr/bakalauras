from fastapi import APIRouter, HTTPException
from app.config import CITIES_CONFIG, CITY_RADIUS_M
from app.services.graph_service import load_or_create_city_geojson, delete_city_cache

router = APIRouter(prefix="/cities", tags=["cities"])


@router.get("/{city_name}")
def read_city(city_name: str):
    """Get GeoJSON data for a city."""
    city_name = city_name.lower()
    if city_name not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City {city_name} not found")
    place_name, coords = CITIES_CONFIG[city_name]
    return load_or_create_city_geojson(city_name, place_name, coords)


@router.delete("/{city_name}/cache")
def clear_city_cache(city_name: str):
    """
    Delete cached OSM graph and GeoJSON for a city so it re-fetches
    from OpenStreetMap with the current CITY_RADIUS_M on next request.
    Use this after changing CITY_RADIUS_M or to force a data refresh.
    """
    city_name = city_name.lower()
    if city_name not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City {city_name} not found")

    deleted = delete_city_cache(city_name)
    radius = CITY_RADIUS_M.get(city_name, 10000)

    return {
        "city": city_name,
        "deleted_files": deleted,
        "next_fetch_radius_m": radius,
        "message": f"Cache cleared. Next request will re-fetch OSM data at {radius}m radius."
    }