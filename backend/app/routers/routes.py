import networkx as nx
import osmnx as ox
from fastapi import APIRouter, HTTPException

from app.config import CITIES_CONFIG
from app.schemas import RouteRequest
from app.services.graph_service import load_or_create_city_graph
from app.services.traffic_service import load_latest_snapshot, load_snapshot

router = APIRouter(prefix="/route", tags=["route"])

# Default speed fallback per OSM highway type (kph) —
# used when a road has no TomTom data
HIGHWAY_DEFAULT_SPEED_KPH = {
    "motorway": 110,
    "motorway_link": 60,
    "trunk": 90,
    "trunk_link": 50,
    "primary": 70,
    "primary_link": 40,
    "secondary": 60,
    "secondary_link": 30,
    "tertiary": 50,
    "tertiary_link": 25,
    "residential": 30,
    "living_street": 10,
    "service": 20,
    "unclassified": 40,
}
DEFAULT_SPEED_KPH = 30  # fallback if highway type unknown


def _build_traffic_weights(G: nx.MultiDiGraph, snapshot: dict) -> dict:
    """
    Build a dict of {(u, v, key): travel_time_seconds} from snapshot data.
    For each OSM edge, travel_time = length(m) / speed(m/s).
    Falls back to OSM maxspeed or highway-type default if no TomTom data.
    """
    # Index snapshot features by osmid for fast lookup
    # OSM edge properties include 'osmid' which may be int or list
    traffic_by_osmid: dict[int, dict] = {}
    for feat in snapshot.get("features", []):
        props = feat.get("properties", {})
        current_speed = props.get("current_speed_kph")
        if current_speed is None:
            continue
        osmid = props.get("osmid")
        if osmid is None:
            continue
        # osmid can be a list in OSM (merged edges)
        if isinstance(osmid, list):
            for oid in osmid:
                traffic_by_osmid[int(oid)] = props
        else:
            traffic_by_osmid[int(osmid)] = props

    weights = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        length = data.get("length", 1)  # meters

        # Try TomTom speed first
        osmid = data.get("osmid")
        speed_kph = None

        if osmid is not None:
            lookup_ids = osmid if isinstance(osmid, list) else [osmid]
            for oid in lookup_ids:
                if int(oid) in traffic_by_osmid:
                    speed_kph = traffic_by_osmid[int(oid)].get("current_speed_kph")
                    break

        # Fallback 1: OSM maxspeed tag
        if speed_kph is None:
            maxspeed = data.get("maxspeed")
            if maxspeed:
                try:
                    # maxspeed can be "50" or "50 mph" etc.
                    speed_kph = float(str(maxspeed).split()[0])
                except (ValueError, TypeError):
                    pass

        # Fallback 2: highway type default
        if speed_kph is None:
            highway = data.get("highway")
            if isinstance(highway, list):
                highway = highway[0]
            speed_kph = HIGHWAY_DEFAULT_SPEED_KPH.get(highway, DEFAULT_SPEED_KPH)

        speed_ms = max(speed_kph / 3.6, 0.5)  # convert kph → m/s, min 0.5 m/s
        weights[(u, v, key)] = length / speed_ms

    return weights


def _run_route(G, start_node, end_node, weight_attr: str) -> list:
    route_nodes = nx.shortest_path(G, start_node, end_node, weight=weight_attr)
    return [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in route_nodes]


@router.post("/classic")
def calculate_route_classic(req: RouteRequest):
    """Shortest path by distance only — ignores traffic completely."""
    return _calculate_route(req, force_weight="length")


@router.post("/traffic")
def calculate_route_traffic(req: RouteRequest):
    """Fastest path using TomTom current speeds as edge weights."""
    return _calculate_route(req, force_weight="travel_time")


@router.post("/compare")
def compare_routes(req: RouteRequest):
    """
    Returns both classic and traffic-aware routes in one call.
    Useful for frontend comparison view.
    """
    classic = _calculate_route(req, force_weight="length")
    traffic = _calculate_route(req, force_weight="travel_time")
    return {
        "classic": classic,
        "traffic": traffic,
    }


def _calculate_route(req: RouteRequest, force_weight: str):
    try:
        city_name = req.city.lower()

        if city_name not in CITIES_CONFIG:
            raise HTTPException(status_code=404, detail=f"City {city_name} not found")

        place_name, coords = CITIES_CONFIG[city_name]
        G = load_or_create_city_graph(city_name, place_name, coords)

        start_node = ox.distance.nearest_nodes(G, req.start[1], req.start[0])
        end_node = ox.distance.nearest_nodes(G, req.end[1], req.end[0])

        weight_attr = force_weight

        if weight_attr == "travel_time":
            # Load snapshot and inject travel_time as edge attribute
            snapshot = (
                load_snapshot(city_name, req.snapshot)
                if req.snapshot
                else load_latest_snapshot(city_name)
            )

            if snapshot is None:
                raise HTTPException(
                    status_code=404,
                    detail="No traffic snapshot found. Collect one first via POST /traffic/snapshot/{city}"
                )

            weights = _build_traffic_weights(G, snapshot)

            # Temporarily set travel_time attribute on edges
            for (u, v, key), t in weights.items():
                if G.has_edge(u, v, key):
                    G[u][v][key]["travel_time"] = t

        route_coords = _run_route(G, start_node, end_node, weight_attr)

        # Calculate total distance and estimated travel time
        route_nodes = nx.shortest_path(G, start_node, end_node, weight=weight_attr)
        total_length = sum(
            G[u][v][0].get("length", 0)
            for u, v in zip(route_nodes[:-1], route_nodes[1:])
        )
        total_time = sum(
            G[u][v][0].get("travel_time", G[u][v][0].get("length", 0) / (DEFAULT_SPEED_KPH / 3.6))
            for u, v in zip(route_nodes[:-1], route_nodes[1:])
        ) if weight_attr == "travel_time" else None

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": route_coords},
                "properties": {
                    "weight": weight_attr,
                    "total_length_m": round(total_length),
                    "estimated_time_s": round(total_time) if total_time else None,
                    "snapshot": req.snapshot or "latest",
                },
            }],
        }

    except nx.NetworkXNoPath:
        raise HTTPException(status_code=400, detail="No path found between points")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating route: {e}")
        raise HTTPException(status_code=500, detail=str(e))