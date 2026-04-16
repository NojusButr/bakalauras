import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HERE_API_KEY = os.getenv("HERE_API_KEY")

CITIES_DIR = Path("cities")
CITIES_DIR.mkdir(exist_ok=True)

CITIES_CONFIG = {
    "vilnius":    ("Vilnius, Lithuania",    (54.6872, 25.2797)),
    "kaunas":     ("Kaunas, Lithuania",     (54.8982, 23.8961)),
    "klaipėda":  ("Klaipėda, Lithuania",  (55.7206, 21.1545)),
    "šiauliai":  ("Šiauliai, Lithuania",  (55.9315, 23.3115)),
    "panevėžys": ("Panevėžys, Lithuania", (55.7333, 24.3667)),
}

# Radius in meters — used for BOTH OSM graph queries and HERE API circle queries.
CITY_RADIUS_M = {
    "vilnius":    15000,
    "kaunas":     12000,
    "klaipėda":  10000,
    "šiauliai":   8000,
    "panevėžys":  8000,
}