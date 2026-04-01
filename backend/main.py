from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CITIES_CONFIG
from app.routers import cities, routes, traffic, simulation
from app.services.graph_service import load_or_create_city_geojson

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(traffic.router)
app.include_router(cities.router)
app.include_router(routes.router)
app.include_router(simulation.router)


@app.on_event("startup")
def preload_cities():
    """Pre-load all city graphs and GeoJSON on startup."""
    print("Initializing city data...")
    for city_key, (place_name, coords) in CITIES_CONFIG.items():
        try:
            print(f"Loading {city_key}...")
            load_or_create_city_geojson(city_key, place_name, coords)
            print(f"Successfully loaded {city_key}")
        except Exception as e:
            print(f"Error loading {city_key}: {e}")
    print("All cities loaded.")
