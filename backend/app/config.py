import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOMTOM_API = os.getenv("TOMTOM_API_KEY")

CITIES_DIR = Path("cities")
CITIES_DIR.mkdir(exist_ok=True)

CITIES_CONFIG = {
    "vilnius": ("Vilnius, Lithuania", (54.6872, 25.2797)),
    "kaunas": ("Kaunas, Lithuania", (54.8982, 23.8961)),
    "klaipėda": ("Klaipėda, Lithuania", (55.7206, 21.1545)),
    "šiauliai": ("Šiauliai, Lithuania", (55.9315, 23.3115)),
    "panevėžys": ("Panevėžys, Lithuania", (55.7333, 24.3667)),
}

CITY_BOUNDING_BOX = 5000  # meters around city center to fetch data for

GRID_SPACING_TOMTOM = 250 
