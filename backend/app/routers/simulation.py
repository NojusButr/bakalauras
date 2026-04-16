from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

from app.config import CITIES_CONFIG
from app.services.traffic_service import load_latest_snapshot, load_snapshot
from app.services.simulation_service import (
    apply_simulation, preview_simulation, run_preset,
    save_simulation, list_simulations, load_simulation,
    get_all_presets, save_user_preset, delete_user_preset,
)

router = APIRouter(prefix="/simulation", tags=["simulation"])


class CrisisEvent(BaseModel):
    type: Literal["construction", "congestion", "damage"]
    center: list[float]
    radius_m: float = 500
    speed_reduction_pct: float = 50
    description: str = ""


class SimulationRequest(BaseModel):
    city: str = "vilnius"
    events: list[CrisisEvent]
    scenario_name: str = "custom"
    scenario_description: str = ""
    base_snapshot: str | None = None
    propagation_depth: int = 3
    propagation_decay: float = 0.55


class PreviewRequest(BaseModel):
    city: str = "vilnius"
    events: list[CrisisEvent]
    propagation_depth: int = 3
    propagation_decay: float = 0.55
    base_snapshot: str | None = None


class SavePresetRequest(BaseModel):
    city: str = "vilnius"
    key: str            
    name: str             
    description: str = ""
    events: list[CrisisEvent]


@router.get("/presets/{city_name}")
def get_presets(city_name: str):
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{city_name}' not found")
    return {"city": city_key, "presets": get_all_presets(city_key)}


@router.post("/presets/save")
def save_preset(req: SavePresetRequest):
    """Save user-created preset for reuse."""
    city_key = req.city.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{req.city}' not found")
    if not req.events:
        raise HTTPException(400, "No events to save")
    if not req.name.strip():
        raise HTTPException(400, "Preset name required")

    preset = save_user_preset(
        city_key,
        req.key or req.name,
        req.name.strip(),
        req.description.strip(),
        [e.model_dump() for e in req.events],
    )
    return {"status": "saved", "key": req.key, "preset": preset}


@router.delete("/presets/{city_name}/{preset_key}")
def delete_preset(city_name: str, preset_key: str):
    """Delete a user-created preset."""
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{city_name}' not found")
    deleted = delete_user_preset(city_key, preset_key)
    if not deleted:
        raise HTTPException(404, f"Preset '{preset_key}' not found or is a built-in preset")
    return {"status": "deleted", "key": preset_key}


@router.post("/preview")
def preview_simulation_endpoint(req: PreviewRequest):
    city_key = req.city.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{req.city}' not found")
    base = (load_snapshot(city_key, req.base_snapshot) if req.base_snapshot
            else load_latest_snapshot(city_key))
    if base is None:
        raise HTTPException(404, f"No snapshot for '{city_key}'")
    try:
        return preview_simulation(
            city_key, base, [e.model_dump() for e in req.events],
            req.propagation_depth, req.propagation_decay,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/preset/{city_name}/{preset_key}")
def run_preset_scenario(city_name: str, preset_key: str,
                        base_snapshot: str | None = None,
                        propagation_depth: int = 3, propagation_decay: float = 0.55):
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{city_name}' not found")
    base = load_snapshot(city_key, base_snapshot) if base_snapshot else None
    try:
        sim = run_preset(city_key, preset_key, base, propagation_depth, propagation_decay)
    except ValueError as e:
        raise HTTPException(400, str(e))
    path = save_simulation(city_key, sim)
    sim["metadata"]["saved_as"] = path.stem
    return sim


@router.post("/custom")
def run_custom_simulation(req: SimulationRequest):
    city_key = req.city.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{req.city}' not found")
    base = (load_snapshot(city_key, req.base_snapshot) if req.base_snapshot
            else load_latest_snapshot(city_key))
    if base is None:
        raise HTTPException(404, f"No snapshot for '{city_key}'")
    try:
        sim = apply_simulation(
            city_key, base, [e.model_dump() for e in req.events],
            req.scenario_name, req.scenario_description,
            req.propagation_depth, req.propagation_decay,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    path = save_simulation(city_key, sim)
    sim["metadata"]["saved_as"] = path.stem
    return sim


@router.get("/list/{city_name}")
def get_simulation_list(city_name: str):
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{city_name}' not found")
    return {"city": city_key, "simulations": list_simulations(city_key)}


@router.get("/{city_name}/{filename}")
def get_simulation(city_name: str, filename: str):
    city_key = city_name.lower()
    if city_key not in CITIES_CONFIG:
        raise HTTPException(404, f"City '{city_name}' not found")
    sim = load_simulation(city_key, filename)
    if sim is None:
        raise HTTPException(404, f"Simulation '{filename}' not found")
    return sim