from pydantic import BaseModel
from typing import Literal


class RouteRequest(BaseModel):
    start: list        # [lat, lng]
    end: list          # [lat, lng]
    city: str = "Vilnius"
    # "length"       → shortest distance (classic Dijkstra, ignores traffic)
    # "travel_time"  → fastest time using TomTom current speeds
    weight: Literal["length", "travel_time"] = "length"
    # optional: pin to a specific snapshot filename, otherwise uses latest
    snapshot: str | None = None