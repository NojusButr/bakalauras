from fastapi import APIRouter
from app.config import CITIES_CONFIG
from app.services.graph_service import load_or_create_city_geojson

router = APIRouter(prefix="/cities", tags=["cities"])


@router.get("/{city_name}")
def read_city(city_name: str):
    """Get GeoJSON data for a city."""
    city_name = city_name.lower()

    if city_name not in CITIES_CONFIG:
        return {"error": f"City {city_name} not found"}, 404

    place_name, coords = CITIES_CONFIG[city_name]
    return load_or_create_city_geojson(city_name, place_name, coords)
