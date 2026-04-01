from pydantic import BaseModel
from typing import Literal


class RouteRequest(BaseModel):
    start: list        # [lat, lng]
    end: list          # [lat, lng]
    city: str = "Vilnius"
    weight: Literal["length", "travel_time"] = "length"
    # pin to a specific real snapshot filename
    snapshot: str | None = None
    # pin to a specific simulation filename (overrides snapshot)
    simulation: str | None = None
