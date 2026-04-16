from pydantic import BaseModel
from typing import Literal


class RouteRequest(BaseModel):
    start: list        # [lat, lng]
    end: list          # [lat, lng]
    city: str = "Vilnius"
    weight: Literal["length", "travel_time", "gnn_travel_time"] = "length"
    snapshot: str | None = None
    simulation: str | None = None
    eval_snapshot: str | None = None
    data_pct: int = 100
    # Degradation mode:
    #   "random"    — random edges stripped (original behavior)
    #   "corridor"  — strip edges near the route corridor first
    #   "minor"     — strip minor roads first (residential, service, tertiary)
    #   "zone"      — strip from a geographic cluster near midpoint
    degrade_mode: Literal["random", "corridor", "minor", "zone"] = "random"
    corridor_width: int = 500    # meters, for corridor mode
    zone_radius: int = 2000      # meters, for zone mode