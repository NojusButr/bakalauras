import math
import networkx as nx
import osmnx as ox
from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.config import CITIES_CONFIG, CITIES_DIR
from app.schemas import RouteRequest
from app.services.graph_service import load_or_create_city_graph
from app.services.traffic_service import load_latest_snapshot, load_snapshot

router = APIRouter(prefix="/route", tags=["route"])

HIGHWAY_DEFAULT_SPEED_KPH = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 70, "primary_link": 40, "secondary": 60, "secondary_link": 30,
    "tertiary": 50, "tertiary_link": 25, "residential": 30, "living_street": 10,
    "service": 20, "unclassified": 40,
}
DEFAULT_SPEED_KPH = 30
IMPASSABLE_WEIGHT = 1_000_000.0


def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_traffic_weights(G: nx.MultiDiGraph, snapshot: dict) -> dict:
    """Build edge weights from snapshot. Also index by u/v for reliable matching."""
    traffic_by_osmid: dict[int, dict] = {}
    traffic_by_uv: dict[tuple[int, int], dict] = {}

    for feat in snapshot.get("features", []):
        props = feat.get("properties", {})
        osmid = props.get("osmid")
        if osmid is not None:
            if isinstance(osmid, list):
                for oid in osmid:
                    traffic_by_osmid[int(oid)] = props
            else:
                traffic_by_osmid[int(osmid)] = props
        u_node, v_node = props.get("u"), props.get("v")
        if u_node is not None and v_node is not None:
            traffic_by_uv[(int(u_node), int(v_node))] = props

    weights = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        length = data.get("length", 1)
        tp = None

        osmid = data.get("osmid")
        if osmid is not None:
            lookup_ids = osmid if isinstance(osmid, list) else [osmid]
            for oid in lookup_ids:
                if int(oid) in traffic_by_osmid:
                    tp = traffic_by_osmid[int(oid)]
                    break
        if tp is None:
            tp = traffic_by_uv.get((u, v))

        if tp is not None:
            if tp.get("impassable") or (tp.get("road_closure") and tp.get("current_speed_kph", 1) == 0):
                weights[(u, v, key)] = IMPASSABLE_WEIGHT
                continue
            speed_kph = tp.get("current_speed_kph")
            if speed_kph is not None and speed_kph > 0:
                weights[(u, v, key)] = length / max(speed_kph / 3.6, 0.5)
                continue

        speed_kph = None
        maxspeed = data.get("maxspeed")
        if maxspeed:
            try:
                speed_kph = float(str(maxspeed).split()[0])
            except (ValueError, TypeError):
                pass
        if speed_kph is None:
            highway = data.get("highway")
            if isinstance(highway, list):
                highway = highway[0]
            speed_kph = HIGHWAY_DEFAULT_SPEED_KPH.get(highway, DEFAULT_SPEED_KPH)

        weights[(u, v, key)] = length / max(speed_kph / 3.6, 0.5)

    return weights


def _apply_simulation_to_graph(G: nx.MultiDiGraph, snapshot: dict) -> int:
    """
    Directly remove impassable edges from the graph using the same spatial
    approach the simulation uses. This is the RELIABLE way to block routing
    through destroyed roads — no osmid matching needed.

    For each simulation event with speed=0 or impassable, find graph edges
    whose midpoint is within the event radius and remove them.

    Returns number of edges removed.
    """
    events = snapshot.get("metadata", {}).get("events", [])
    if not events:
        removed = 0
        edges_to_remove = []
        for feat in snapshot.get("features", []):
            props = feat.get("properties", {})
            if not (props.get("impassable") or (props.get("road_closure") and props.get("current_speed_kph", 1) == 0)):
                continue
            u_node, v_node = props.get("u"), props.get("v")
            if u_node is not None and v_node is not None:
                u_int, v_int = int(u_node), int(v_node)
                if G.has_edge(u_int, v_int):
                    edges_to_remove.append((u_int, v_int))
        if edges_to_remove:
            G.remove_edges_from(edges_to_remove)
            removed = len(edges_to_remove)
        return removed

    # Use events to find and remove edges spatially
    removed = 0
    for event in events:
        speed_pct = event.get("speed_reduction_pct", 100)
        event_type = event.get("type", "congestion")
        if event_type == "damage":
            speed_pct = 0
        if speed_pct > 0:
            continue

        center = event.get("center", [0, 0])
        radius_m = event.get("radius_m", 500)
        lat, lng = center[0], center[1]

        edges_to_remove = []
        for u, v, key, data in G.edges(keys=True, data=True):
            # Get edge midpoint from node coordinates
            u_lat, u_lng = G.nodes[u].get("y", 0), G.nodes[u].get("x", 0)
            v_lat, v_lng = G.nodes[v].get("y", 0), G.nodes[v].get("x", 0)
            mid_lat = (u_lat + v_lat) / 2
            mid_lng = (u_lng + v_lng) / 2

            if _haversine_m(lat, lng, mid_lat, mid_lng) <= radius_m:
                edges_to_remove.append((u, v, key))

        for u, v, key in edges_to_remove:
            if G.has_edge(u, v, key):
                G.remove_edge(u, v, key)
                removed += 1

    return removed


def _run_route(G, start_node, end_node, weight_attr: str) -> list:
    """Build route coordinates using full edge geometry."""
    route_nodes = nx.shortest_path(G, start_node, end_node, weight=weight_attr)
    coords = []
    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        edge_data = G[u][v]
        key = min(edge_data.keys())
        data = edge_data[key]

        if "geometry" in data:
            line_coords = list(data["geometry"].coords)
            u_coord = (G.nodes[u]["x"], G.nodes[u]["y"])
            dist_to_first = (u_coord[0] - line_coords[0][0])**2 + (u_coord[1] - line_coords[0][1])**2
            dist_to_last = (u_coord[0] - line_coords[-1][0])**2 + (u_coord[1] - line_coords[-1][1])**2
            if dist_to_last < dist_to_first:
                line_coords = list(reversed(line_coords))
            for j, (lng, lat) in enumerate(line_coords):
                if j == 0 and coords and abs(coords[-1][0] - lng) < 1e-8 and abs(coords[-1][1] - lat) < 1e-8:
                    continue
                coords.append([lng, lat])
        else:
            pt = [G.nodes[u]["x"], G.nodes[u]["y"]]
            if not coords or coords[-1] != pt:
                coords.append(pt)

    if route_nodes:
        last = [G.nodes[route_nodes[-1]]["x"], G.nodes[route_nodes[-1]]["y"]]
        if not coords or coords[-1] != last:
            coords.append(last)
    return coords


@router.post("/classic")
def calculate_route_classic(req: RouteRequest):
    return _calculate_route(req, force_weight="length")

@router.post("/traffic")
def calculate_route_traffic(req: RouteRequest):
    return _calculate_route(req, force_weight="travel_time")

@router.post("/gnn")
def calculate_route_gnn(req: RouteRequest):
    return _calculate_route(req, force_weight="gnn_travel_time")

@router.post("/classifier")
def calculate_route_classifier(req: RouteRequest):
    return _calculate_classifier_route(req)

def _degrade_snapshot(snapshot, keep_pct, seed=None, mode="random",
                      start=None, end=None, corridor_width=500, zone_radius=2000,
                      graph=None):
    """
    Strip traffic data with different realistic patterns.
    
    For random: keep_pct controls how much remains.
    For corridor/zone/minor: removes ALL matching roads. Slider ignored.
    """
    import random as _rnd
    import math
    from copy import deepcopy

    degraded = deepcopy(snapshot)
    features = degraded.get("features", [])
    
    # Only consider stripping features that have traffic data
    # Protect ONLY direct event impacts (propagation_hop=0, impassable/damaged)
    # Propagated congestion (hop >= 1) CAN be stripped
    has_traffic = []
    for i, f in enumerate(features):
        props = f.get("properties", {})
        if props.get("current_speed_kph") is None:
            continue
        # Protect direct hits only
        if props.get("simulated") and props.get("propagation_hop", 99) == 0:
            continue
        has_traffic.append(i)
    if not has_traffic:
        return degraded, set()
    if mode == "random" and keep_pct >= 100:
        return degraded, set()

    if seed is not None:
        _rnd.seed(seed)

    def _get_road_coords(feat_idx):
        f = features[feat_idx]
        geom = f.get("geometry", {}).get("coordinates", [])
        if not geom:
            return None, None
        if isinstance(geom[0], list):
            pt = geom[len(geom) // 2]
        else:
            pt = geom
        return pt[1], pt[0]

    def _hav(lat1, lng1, lat2, lng2):
        R = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlng/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    to_remove = set()

    if mode == "corridor" and start and end:
        s_lat, s_lng = math.radians(start[0]), math.radians(start[1])
        e_lat, e_lng = math.radians(end[0]), math.radians(end[1])
        for i in has_traffic:
            lat, lng = _get_road_coords(i)
            if lat is None:
                continue
            p_lat, p_lng = math.radians(lat), math.radians(lng)
            d_sp = _hav(start[0], start[1], lat, lng) / 6_371_000
            bearing_sp = math.atan2(
                math.sin(p_lng - s_lng) * math.cos(p_lat),
                math.cos(s_lat) * math.sin(p_lat) - math.sin(s_lat) * math.cos(p_lat) * math.cos(p_lng - s_lng)
            )
            bearing_se = math.atan2(
                math.sin(e_lng - s_lng) * math.cos(e_lat),
                math.cos(s_lat) * math.sin(e_lat) - math.sin(s_lat) * math.cos(e_lat) * math.cos(e_lng - s_lng)
            )
            xt = abs(math.asin(max(-1, min(1, math.sin(d_sp) * math.sin(bearing_sp - bearing_se))))) * 6_371_000
            d_se = _hav(start[0], start[1], end[0], end[1])
            d_pe = _hav(lat, lng, end[0], end[1])
            d_ps = _hav(lat, lng, start[0], start[1])
            dist = min(d_ps, d_pe) if (d_ps > d_se or d_pe > d_se) else xt
            if dist <= corridor_width:
                to_remove.add(i)

    elif mode == "minor" and graph is not None:
        MINOR_TYPES = {"residential", "living_street", "service", "unclassified",
                       "tertiary", "tertiary_link"}
        osmid_to_highway = {}
        for u, v, k, d in graph.edges(keys=True, data=True):
            osmid = d.get("osmid")
            hw = d.get("highway", "unclassified")
            if isinstance(hw, list):
                hw = hw[0]
            if osmid is not None:
                ids = osmid if isinstance(osmid, list) else [osmid]
                for oid in ids:
                    osmid_to_highway[int(oid)] = hw
        for i in has_traffic:
            props = features[i].get("properties", {})
            osmid = props.get("osmid")
            hw = "unclassified"
            if osmid is not None:
                ids = osmid if isinstance(osmid, list) else [osmid]
                for oid in ids:
                    if int(oid) in osmid_to_highway:
                        hw = osmid_to_highway[int(oid)]
                        break
            if hw in MINOR_TYPES:
                to_remove.add(i)

    elif mode == "zone" and start and end:
        mid_lat = (start[0] + end[0]) / 2
        mid_lng = (start[1] + end[1]) / 2
        for i in has_traffic:
            lat, lng = _get_road_coords(i)
            if lat is None:
                continue
            if _hav(mid_lat, mid_lng, lat, lng) <= zone_radius:
                to_remove.add(i)

    else:
        n_remove = int(len(has_traffic) * (1 - keep_pct / 100))
        to_remove = set(_rnd.sample(has_traffic, min(n_remove, len(has_traffic))))

    for i in to_remove:
        props = features[i]["properties"]
        props["degraded"] = True
        props["current_speed_kph"] = None
        props["free_flow_speed_kph"] = None
        props["jam_factor"] = None
        props["congestion_ratio"] = None
        props["congestion_level"] = "unknown"
        props["confidence"] = None

    degraded.setdefault("metadata", {})["degraded_pct"] = keep_pct
    degraded["metadata"]["degrade_mode"] = mode
    degraded["metadata"]["edges_stripped"] = len(to_remove)
    return degraded, to_remove




@router.post("/compare")
def compare_routes(req: RouteRequest):
    data_pct = max(0, min(100, req.data_pct))
    classic = _calculate_route(req, force_weight="length")
    degraded_info = None
    degraded_geojson = None

    # Degrade if: random mode with pct < 100, OR any non-random mode
    should_degrade = (req.degrade_mode == "random" and data_pct < 100) or req.degrade_mode != "random"

    if should_degrade:
        city_name = req.city.lower()
        place_name, coords = CITIES_CONFIG[city_name]
        G_for_degrade = load_or_create_city_graph(city_name, place_name, coords)
        
        full_snap = None
        if req.simulation:
            from app.services.simulation_service import load_simulation
            full_snap = load_simulation(city_name, req.simulation)
        elif req.snapshot:
            full_snap = load_snapshot(city_name, req.snapshot)
        else:
            full_snap = load_latest_snapshot(city_name)

        if full_snap:
            seed = int(abs(req.start[0] * 1e6) + abs(req.start[1] * 1e6) +
                       abs(req.end[0] * 1e6) + abs(req.end[1] * 1e6) + data_pct) % (2**31)

            degraded, stripped_indices = _degrade_snapshot(
                full_snap, data_pct, seed=seed, mode=req.degrade_mode,
                start=req.start, end=req.end,
                corridor_width=req.corridor_width, zone_radius=req.zone_radius,
                graph=G_for_degrade
            )

            total_with_traffic = sum(
                1 for f in full_snap.get("features", [])
                if f.get("properties", {}).get("current_speed_kph") is not None
            )
            actual_pct = round((total_with_traffic - len(stripped_indices)) / max(total_with_traffic, 1) * 100)
            degraded_info = {
                "data_pct": actual_pct,
                "mode": req.degrade_mode,
                "total_edges": len(full_snap.get("features", [])),
                "edges_with_traffic": total_with_traffic,
                "edges_stripped": len(stripped_indices),
                "edges_remaining": total_with_traffic - len(stripped_indices),
            }

            stripped_features = []
            for i in stripped_indices:
                feat = full_snap["features"][i]
                stripped_features.append({
                    "type": "Feature",
                    "geometry": feat.get("geometry", {}),
                    "properties": {
                        "stripped": True,
                        "original_speed": feat.get("properties", {}).get("current_speed_kph"),
                        "original_jam": feat.get("properties", {}).get("jam_factor"),
                        "name": feat.get("properties", {}).get("name", ""),
                    }
                })
            degraded_geojson = {
                "type": "FeatureCollection",
                "features": stripped_features,
            }

        traffic = _calculate_route_degraded(req, data_pct, method="traffic", seed=seed)
        result = {"classic": classic, "traffic": traffic, "data_pct": actual_pct if degraded_info else data_pct}

        try:
            result["gnn"] = _calculate_route_degraded(req, data_pct, method="gnn", seed=seed)
        except HTTPException as e:
            if "GNN" in str(e.detail):
                result["gnn"] = None
            else:
                raise
        except Exception:
            result["gnn"] = None
    else:
        traffic = _calculate_route(req, force_weight="travel_time")
        result = {"classic": classic, "traffic": traffic, "data_pct": data_pct}

        try:
            result["gnn"] = _calculate_route(req, force_weight="gnn_travel_time")
        except HTTPException as e:
            if "GNN" in str(e.detail):
                result["gnn"] = None
            else:
                raise
        except Exception:
            result["gnn"] = None

    if degraded_info:
        result["degraded_info"] = degraded_info
    if degraded_geojson:
        result["degraded_geojson"] = degraded_geojson

    # Classifier
    try:
        result["classifier"] = _calculate_classifier_route(req)
    except HTTPException as e:
        if "classifier" in str(e.detail).lower() or "not found" in str(e.detail).lower():
            result["classifier"] = None
        else:
            raise
    except Exception as e:
        print(f"Classifier route failed: {e}")
        result["classifier"] = None

    return result


def _calculate_route_degraded(req: RouteRequest, data_pct: int, method: str = "traffic", seed: int = None):
    """
    Route with degraded data, evaluate with full data.
    Both Dijkstra (traffic) and GNN receive the SAME degraded snapshot.
    The difference: Dijkstra uses degraded speeds directly as weights,
    GNN uses degraded features as input but its trained model fills gaps.
    """
    city_name = req.city.lower()
    if city_name not in CITIES_CONFIG:
        raise HTTPException(404, f"City {city_name} not found")

    full_snapshot = None
    if req.simulation:
        from app.services.simulation_service import load_simulation
        full_snapshot = load_simulation(city_name, req.simulation)
    elif req.snapshot:
        full_snapshot = load_snapshot(city_name, req.snapshot)
    else:
        full_snapshot = load_latest_snapshot(city_name)

    if full_snapshot is None:
        raise HTTPException(404, "No snapshot found")

    place_name, coords = CITIES_CONFIG[city_name]
    G_original = load_or_create_city_graph(city_name, place_name, coords)

    degraded, _ = _degrade_snapshot(full_snapshot, data_pct, seed=seed,
                                     mode=req.degrade_mode, start=req.start, end=req.end,
                                     corridor_width=req.corridor_width, zone_radius=req.zone_radius,
                                     graph=G_original)

    G = G_original.copy()

    start_node = ox.distance.nearest_nodes(G, req.start[1], req.start[0])
    end_node = ox.distance.nearest_nodes(G, req.end[1], req.end[0])

    is_simulation = req.simulation is not None
    if is_simulation and method != "length":
        removed = _apply_simulation_to_graph(G, full_snapshot)
        if removed > 0:
            start_node = ox.distance.nearest_nodes(G, req.start[1], req.start[0])
            end_node = ox.distance.nearest_nodes(G, req.end[1], req.end[0])

    weight_attr = "travel_time"

    if method == "gnn":
        weight_attr = "gnn_travel_time"
        model_path = CITIES_DIR / city_name / "models" / "best_model.pt"
        if not model_path.exists():
            model_path = Path("models") / "best_model.pt"
        if not model_path.exists():
            raise HTTPException(404, "GNN model not found")

        graph_path = CITIES_DIR / city_name / "graph.pkl"
        try:
            import sys
            project_root = Path(__file__).parent.parent.parent
            gnn_dir = str(project_root / "gnn")
            if gnn_dir not in sys.path:
                sys.path.insert(0, gnn_dir)
            from gnn_service import gnn_route_weights
            gnn_weights = gnn_route_weights(G, graph_path, model_path, degraded)
        except ImportError as e:
            raise HTTPException(500, f"GNN service not available: {e}")

        for (u, v, key), t in gnn_weights.items():
            if G.has_edge(u, v, key):
                G[u][v][key]["gnn_travel_time"] = t

        # Only override for truly impassable edges (destroyed roads)
        # Both Dijkstra and GNN see these from the protected direct-hit features
        degraded_weights = _build_traffic_weights(G, degraded)
        for (u, v, key), t in degraded_weights.items():
            if G.has_edge(u, v, key) and t >= IMPASSABLE_WEIGHT:
                G[u][v][key]["gnn_travel_time"] = IMPASSABLE_WEIGHT
    else:
        # Dijkstra route with degraded traffic weights
        degraded_weights = _build_traffic_weights(G, degraded)
        for (u, v, key), t in degraded_weights.items():
            if G.has_edge(u, v, key):
                G[u][v][key]["travel_time"] = t

    try:
        route_coords = _run_route(G, start_node, end_node, weight_attr)
        route_nodes = nx.shortest_path(G, start_node, end_node, weight=weight_attr)
    except nx.NetworkXNoPath:
        raise HTTPException(400, "No path found")

    total_length = sum(
        G[u][v][0].get("length", 0) for u, v in zip(route_nodes[:-1], route_nodes[1:])
    )

    # Evaluate ALL routes with FULL data for fair comparison
    full_weights = _build_traffic_weights(G_original, full_snapshot)
    total_time = 0
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        w = full_weights.get((u, v, 0))
        if w is not None and w < IMPASSABLE_WEIGHT:
            total_time += w
        else:
            edge = G[u][v][0]
            length = edge.get("length", 0)
            hw = edge.get("highway", "unclassified")
            if isinstance(hw, list): hw = hw[0]
            speed = HIGHWAY_DEFAULT_SPEED_KPH.get(hw, DEFAULT_SPEED_KPH)
            total_time += length / max(speed / 3.6, 0.5)

    gnn_est = None
    if method == "gnn":
        gnn_est = sum(
            G[u][v][0].get("gnn_travel_time", 0)
            for u, v in zip(route_nodes[:-1], route_nodes[1:])
        )
        if gnn_est >= IMPASSABLE_WEIGHT:
            gnn_est = None

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": route_coords},
            "properties": {
                "weight": weight_attr,
                "total_length_m": round(total_length),
                "estimated_time_s": round(total_time),
                "gnn_predicted_time_s": round(gnn_est) if gnn_est else None,
                "data_pct": data_pct,
                "snapshot": req.snapshot or "latest",
            },
        }],
    }


def _calculate_route(req: RouteRequest, force_weight: str):
    try:
        city_name = req.city.lower()
        if city_name not in CITIES_CONFIG:
            raise HTTPException(404, f"City {city_name} not found")

        place_name, coords = CITIES_CONFIG[city_name]
        G_original = load_or_create_city_graph(city_name, place_name, coords)
        G = G_original.copy()

        start_node = ox.distance.nearest_nodes(G, req.start[1], req.start[0])
        end_node = ox.distance.nearest_nodes(G, req.end[1], req.end[0])

        weight_attr = force_weight

        # Load snapshot or simulation
        snapshot = None
        is_simulation = False
        if req.simulation:
            from app.services.simulation_service import load_simulation
            sim = load_simulation(city_name, req.simulation)
            if sim:
                snapshot = sim
                is_simulation = True
                print(f"  [{weight_attr}] Using simulation: {req.simulation}")
        elif req.snapshot:
            snapshot = load_snapshot(city_name, req.snapshot)
        else:
            snapshot = load_latest_snapshot(city_name)

        if weight_attr in ("travel_time", "gnn_travel_time") and snapshot is None:
            raise HTTPException(404, "No traffic snapshot found")

        # If this is a simulation, remove destroyed edges from graph
        # Only for traffic-aware and GNN
        if is_simulation and snapshot is not None and weight_attr != "length":
            removed = _apply_simulation_to_graph(G, snapshot)
            if removed > 0:
                print(f"  [{weight_attr}] Removed {removed} impassable edges from graph")
                start_node = ox.distance.nearest_nodes(G, req.start[1], req.start[0])
                end_node = ox.distance.nearest_nodes(G, req.end[1], req.end[0])

        # Build traffic weights for time evaluation
        real_weights = None
        if snapshot is not None:
            real_weights = _build_traffic_weights(G, snapshot)
            for (u, v, key), t in real_weights.items():
                if G.has_edge(u, v, key):
                    G[u][v][key]["travel_time"] = t

        if weight_attr == "gnn_travel_time":
            model_path = CITIES_DIR / city_name / "models" / "best_model.pt"
            if not model_path.exists():
                model_path = Path("models") / "best_model.pt"
            if not model_path.exists():
                raise HTTPException(404, "GNN model not found")

            graph_path = CITIES_DIR / city_name / "graph.pkl"
            try:
                import sys
                project_root = Path(__file__).parent.parent.parent
                gnn_dir = str(project_root / "gnn")
                if gnn_dir not in sys.path:
                    sys.path.insert(0, gnn_dir)
                from gnn_service import gnn_route_weights
                gnn_weights = gnn_route_weights(G, graph_path, model_path, snapshot)
            except ImportError as e:
                raise HTTPException(500, f"GNN service not available: {e}")

            for (u, v, key), t in gnn_weights.items():
                if G.has_edge(u, v, key):
                    if real_weights and real_weights.get((u, v, key), 0) >= IMPASSABLE_WEIGHT:
                        G[u][v][key]["gnn_travel_time"] = IMPASSABLE_WEIGHT
                    else:
                        G[u][v][key]["gnn_travel_time"] = t

        route_coords = _run_route(G, start_node, end_node, weight_attr)

        route_nodes = nx.shortest_path(G, start_node, end_node, weight=weight_attr)
        total_length = sum(
            G[u][v][0].get("length", 0) for u, v in zip(route_nodes[:-1], route_nodes[1:])
        )

        # If eval_snapshot is provided, load it and build weights for fair evaluation
        eval_weights = real_weights
        if req.eval_snapshot:
            eval_snap = load_snapshot(city_name, req.eval_snapshot)
            if eval_snap:
                eval_weights = _build_traffic_weights(G_original, eval_snap)

        # Time calculation depends on route type
        total_time = None
        if weight_attr == "length":
            # Shortest route: estimate time from speed limits
            total_time = 0
            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                edge = G[u][v][0]
                length = edge.get("length", 0)
                speed_kph = None
                maxspeed = edge.get("maxspeed")
                if maxspeed:
                    try:
                        speed_kph = float(str(maxspeed).split()[0])
                    except (ValueError, TypeError):
                        pass
                if speed_kph is None:
                    highway = edge.get("highway")
                    if isinstance(highway, list):
                        highway = highway[0]
                    speed_kph = HIGHWAY_DEFAULT_SPEED_KPH.get(highway, DEFAULT_SPEED_KPH)
                total_time += length / max(speed_kph / 3.6, 0.5)
        elif eval_weights is not None:
            # Use eval weights (full data) for fair time evaluation
            total_time = 0
            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                key = 0
                w = eval_weights.get((u, v, key))
                if w is not None:
                    total_time += w
                else:
                    # Fallback: use whatever is on the edge
                    total_time += G[u][v][0].get("travel_time", G[u][v][0].get("length", 0) / (DEFAULT_SPEED_KPH / 3.6))
            if total_time >= IMPASSABLE_WEIGHT:
                total_time = None

        gnn_predicted_time = None
        if weight_attr == "gnn_travel_time":
            gnn_predicted_time = sum(
                G[u][v][0].get("gnn_travel_time", G[u][v][0].get("length", 0) / (DEFAULT_SPEED_KPH / 3.6))
                for u, v in zip(route_nodes[:-1], route_nodes[1:])
            )
            if gnn_predicted_time >= IMPASSABLE_WEIGHT:
                gnn_predicted_time = None

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": route_coords},
                "properties": {
                    "weight": weight_attr,
                    "total_length_m": round(total_length),
                    "estimated_time_s": round(total_time) if total_time else None,
                    "gnn_predicted_time_s": round(gnn_predicted_time) if gnn_predicted_time else None,
                    "snapshot": req.snapshot or "latest",
                },
            }],
        }

    except nx.NetworkXNoPath:
        raise HTTPException(400, "No path found between points — road network may be severed by simulation")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating route: {e}")
        raise HTTPException(500, str(e))


def _calculate_classifier_route(req: RouteRequest):
    """Run the GNN route classifier — predicts route directly, no Dijkstra."""
    try:
        city_name = req.city.lower()
        if city_name not in CITIES_CONFIG:
            raise HTTPException(404, f"City {city_name} not found")

        place_name, coords = CITIES_CONFIG[city_name]
        G_original = load_or_create_city_graph(city_name, place_name, coords)
        G = G_original.copy()

        start_node = ox.distance.nearest_nodes(G, req.start[1], req.start[0])
        end_node = ox.distance.nearest_nodes(G, req.end[1], req.end[0])

        # Load classifier model
        model_path = CITIES_DIR / city_name / "models" / "route_classifier.pt"
        if not model_path.exists():
            raise HTTPException(404, "Route classifier model not found")

        # Load snapshot for edge features
        snapshot = None
        is_simulation = False
        if req.simulation:
            from app.services.simulation_service import load_simulation
            sim = load_simulation(city_name, req.simulation)
            if sim:
                snapshot = sim
                is_simulation = True
        elif req.snapshot:
            snapshot = load_snapshot(city_name, req.snapshot)
        else:
            snapshot = load_latest_snapshot(city_name)

        # Remove destroyed edges for simulations
        if is_simulation and snapshot is not None:
            removed = _apply_simulation_to_graph(G, snapshot)
            if removed > 0:
                print(f"  [classifier] Removed {removed} impassable edges from graph")
                start_node = ox.distance.nearest_nodes(G, req.start[1], req.start[0])
                end_node = ox.distance.nearest_nodes(G, req.end[1], req.end[0])

        # Run classifier
        import sys
        project_root = Path(__file__).parent.parent.parent
        gnn_dir = str(project_root / "gnn")
        if gnn_dir not in sys.path:
            sys.path.insert(0, gnn_dir)
        from app.services.classifier_service import classifier_route

        route_nodes, edge_probs = classifier_route(G, start_node, end_node, model_path, snapshot)

        if not route_nodes or len(route_nodes) < 2:
            raise HTTPException(400, "Classifier failed to find a route")

        # Build coordinates from route nodes (use edge geometry where available)
        coords_list = []
        for i in range(len(route_nodes) - 1):
            u, v = route_nodes[i], route_nodes[i + 1]
            if u not in G.nodes or v not in G.nodes:
                continue
            if G.has_edge(u, v):
                edge_data = G[u][v]
                key = min(edge_data.keys())
                data = edge_data[key]
                if "geometry" in data:
                    line_coords = list(data["geometry"].coords)
                    u_coord = (G.nodes[u]["x"], G.nodes[u]["y"])
                    dist_first = (u_coord[0] - line_coords[0][0]) ** 2 + (u_coord[1] - line_coords[0][1]) ** 2
                    dist_last = (u_coord[0] - line_coords[-1][0]) ** 2 + (u_coord[1] - line_coords[-1][1]) ** 2
                    if dist_last < dist_first:
                        line_coords = list(reversed(line_coords))
                    for j, (lng, lat) in enumerate(line_coords):
                        if j == 0 and coords_list and abs(coords_list[-1][0] - lng) < 1e-8:
                            continue
                        coords_list.append([lng, lat])
                else:
                    pt = [G.nodes[u]["x"], G.nodes[u]["y"]]
                    if not coords_list or coords_list[-1] != pt:
                        coords_list.append(pt)
            else:
                pt = [G.nodes[u]["x"], G.nodes[u]["y"]]
                if not coords_list or coords_list[-1] != pt:
                    coords_list.append(pt)

        # Always add last node
        if route_nodes and route_nodes[-1] in G.nodes:
            last = [G.nodes[route_nodes[-1]]["x"], G.nodes[route_nodes[-1]]["y"]]
            if not coords_list or coords_list[-1] != last:
                coords_list.append(last)

        total_length = 0
        total_time = 0
        
        eval_snap = snapshot
        if req.eval_snapshot:
            eval_snap = load_snapshot(city_name, req.eval_snapshot)
        real_weights = _build_traffic_weights(G_original, eval_snap) if eval_snap else None

        for i in range(len(route_nodes) - 1):
            u, v = route_nodes[i], route_nodes[i + 1]
            if G.has_edge(u, v):
                key = min(G[u][v].keys())
                edge = G[u][v][key]
                length = edge.get("length", 0)
                total_length += length
                
                w = None
                if real_weights:
                    w = real_weights.get((u, v, key)) or real_weights.get((u, v, 0))
                
                if w is not None and w < IMPASSABLE_WEIGHT:
                    total_time += w
                else:
                    hw = edge.get("highway", "unclassified")
                    if isinstance(hw, list):
                        hw = hw[0]
                    speed = HIGHWAY_DEFAULT_SPEED_KPH.get(hw, DEFAULT_SPEED_KPH)
                    total_time += length / max(speed / 3.6, 0.5)

        if total_time >= IMPASSABLE_WEIGHT:
            total_time = None

        reached_dest = route_nodes[-1] == end_node

        print(f"  [classifier] Route: {len(route_nodes)} nodes, {len(coords_list)} coords, "
              f"{round(total_length)}m, reached_dest={reached_dest}")

        if len(coords_list) < 2:
            coords_list = []
            for n in route_nodes:
                if n in G.nodes:
                    coords_list.append([G.nodes[n]["x"], G.nodes[n]["y"]])
            print(f"  [classifier] Fallback: built {len(coords_list)} coords from node positions")

        if len(coords_list) < 2:
            raise HTTPException(400, "Classifier route has no valid coordinates")

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords_list},
                "properties": {
                    "weight": "classifier",
                    "total_length_m": round(total_length),
                    "estimated_time_s": round(total_time) if total_time else None,
                    "route_nodes": len(route_nodes),
                    "reached_destination": reached_dest,
                    "snapshot": req.snapshot or "latest",
                },
            }],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Classifier route error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))