import json
import pickle
import osmnx as ox
from app.config import CITIES_DIR, CITY_RADIUS_M

# In-memory graph cache
GRAPHS = {}


def load_or_create_city_graph(city_name: str, place_name: str, coords: tuple = None):
    """Load city graph from cache or create it if not exists."""
    city_name_lower = city_name.lower()

    if city_name_lower in GRAPHS:
        return GRAPHS[city_name_lower]

    city_dir = CITIES_DIR / city_name_lower
    city_dir.mkdir(exist_ok=True)
    graph_file = city_dir / "graph.pkl"

    if graph_file.exists():
        print(f"Loading {city_name} graph from cache...")
        with open(graph_file, "rb") as f:
            G = pickle.load(f)
    else:
        radius = CITY_RADIUS_M.get(city_name_lower, 10000)
        print(f"Fetching {city_name} graph from OpenStreetMap (radius={radius}m)...")
        if coords:
            G = ox.graph_from_point(coords, dist=radius, network_type="drive")
        else:
            G = ox.graph_from_place(place_name, network_type="drive")

        with open(graph_file, "wb") as f:
            pickle.dump(G, f)
        print(f"{city_name} graph loaded and cached.")

    GRAPHS[city_name_lower] = G
    return G


def load_or_create_city_geojson(city_name: str, place_name: str, coords: tuple = None) -> dict:
    """Load city GeoJSON from cache or create it if not exists."""
    city_name_lower = city_name.lower()
    city_dir = CITIES_DIR / city_name_lower
    city_dir.mkdir(exist_ok=True)
    geojson_file = city_dir / "geojson.json"

    if geojson_file.exists():
        print(f"Loading {city_name} GeoJSON from cache...")
        with open(geojson_file, "r") as f:
            return json.load(f)

    G = load_or_create_city_graph(city_name, place_name, coords)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    geojson = json.loads(gdf_edges.to_json())

    with open(geojson_file, "w") as f:
        json.dump(geojson, f)

    print(f"{city_name} GeoJSON cached.")
    return geojson


def delete_city_cache(city_name: str):
    """
    Delete cached graph and GeoJSON for a city so it gets re-fetched
    next time with the current CITY_RADIUS_M value.
    """
    city_name_lower = city_name.lower()
    city_dir = CITIES_DIR / city_name_lower

    deleted = []
    for filename in ("graph.pkl", "geojson.json"):
        f = city_dir / filename
        if f.exists():
            f.unlink()
            deleted.append(filename)

    # Also clear from memory cache
    GRAPHS.pop(city_name_lower, None)

    return deleted