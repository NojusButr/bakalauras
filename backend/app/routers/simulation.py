from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

from app.config import CITIES_CONFIG
from app.services.traffic_service import load_latest_snapshot, load_snapshot
from app.services.simulation_service import (
    apply_simulation,
    run_preset,
    save_simulation,
    list_simulations,
    load_simulation,
    PRESET_SCENARIOS,
)

router = APIRouter(prefix="/simulation", tags=["simulation"])


class CrisisEvent(BaseModel):
    type: Literal["road_closure", "district_congestion", "infrastructure_damage"]
    center: list[float]          # [lat, lng]
    radius_m: float = 500
    congestion_factor: float = 0.2   # only for district_congestion
    description: str = ""


class SimulationRequest(BaseModel):
    city: str = "vilnius"
    events: list[CrisisEvent]
    scenario_name: str = "custom"
    scenario_description: str = ""
    base_snapshot: str | None = None
    propagation_depth: int = 3     # 1-6 hops
    propagation_decay: float = 0.55  # 0.3-0.8 per hop


@router.get("/presets/{city_name}")
def get_presets(city_name: str):
    """List available preset scenarios for a city."""
    city_key = city_name.lower()
    presets = PRESET_SCENARIOS.get(city_key, {})
    return {
        "city": city_key,
        "presets": [
            {
                "key": k,
                "name": v["name"],
                "description": v["description"],
                "event_count": len(v["events"]),
            }
            for k, v in presets.items()
        ]
    }


@router.post("/preset/{city_name}/{preset_key}")
def run_preset_scenario(
    city_name: str,
    preset_key: str,
    base_snapshot: str | None = None,
    propagation_depth: int = 3,
    propagation_decay: float = 0.55,
):
    """Run a named preset crisis scenario with optional propagation params."""
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

    base = None
    if base_snapshot:
        base = load_snapshot(city_key, base_snapshot)
        if base is None:
            raise HTTPException(status_code=404, detail=f"Snapshot '{base_snapshot}' not found")

    try:
        simulation = run_preset(
            city_key, preset_key, base,
            propagation_depth=propagation_depth,
            propagation_decay=propagation_decay,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    saved_path = save_simulation(city_key, simulation)
    simulation["metadata"]["saved_as"] = saved_path.stem
    return simulation


@router.post("/custom")
def run_custom_simulation(req: SimulationRequest):
    """
    Run a fully custom simulation with arbitrary crisis events.
    """
    city_key = req.city.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{req.city}' not found")

    if req.base_snapshot:
        base = load_snapshot(city_key, req.base_snapshot)
        if base is None:
            raise HTTPException(status_code=404, detail=f"Snapshot '{req.base_snapshot}' not found")
    else:
        base = load_latest_snapshot(city_key)
        if base is None:
            raise HTTPException(
                status_code=404,
                detail=f"No snapshot found for '{city_key}'. Collect one first."
            )

    try:
        simulation = apply_simulation(
            city_name            = city_key,
            base_snapshot        = base,
            events               = [e.model_dump() for e in req.events],
            scenario_name        = req.scenario_name,
            scenario_description = req.scenario_description,
            propagation_depth    = req.propagation_depth,
            propagation_decay    = req.propagation_decay,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    saved_path = save_simulation(city_key, simulation)
    simulation["metadata"]["saved_as"] = saved_path.stem
    return simulation


@router.get("/list/{city_name}")
def get_simulation_list(city_name: str):
    """List all saved simulations for a city with their metadata."""
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")
    return {"city": city_key, "simulations": list_simulations(city_key)}


@router.get("/{city_name}/{filename}")
def get_simulation(city_name: str, filename: str):
    """Load a specific simulation by filename."""
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

    sim = load_simulation(city_key, filename)
    if sim is None:
        raise HTTPException(status_code=404, detail=f"Simulation '{filename}' not found")
    return sim