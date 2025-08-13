from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict

app = FastAPI(title="Casae API", version="0.1.0")

# Example comps dataset
comps_data = [
    {"id": 1, "address": "123 Main St", "price": 300000, "beds": 3, "baths": 2, "sqft": 1500},
    {"id": 2, "address": "456 Oak Ave", "price": 320000, "beds": 4, "baths": 3, "sqft": 1800},
    {"id": 3, "address": "789 Pine Rd", "price": 280000, "beds": 3, "baths": 2, "sqft": 1400},
    {"id": 4, "address": "101 Cedar Blvd", "price": 310000, "beds": 3, "baths": 2, "sqft": 1600},
    {"id": 5, "address": "202 Maple St", "price": 295000, "beds": 2, "baths": 1, "sqft": 1200},
]

# CORS configuration will be overridden via env vars (ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # to be replaced with config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to Casae API"}


def compute_similarity(comp: Dict, price: Optional[float], beds: Optional[int], baths: Optional[int], sqft: Optional[float]) -> float:
    """Compute a simple similarity score between a comp and the subject parameters."""
    diff = 0.0
    if price is not None:
        diff += abs(comp["price"] - price) / comp["price"]
    if beds is not None:
        diff += abs(comp["beds"] - beds)
    if baths is not None:
        diff += abs(comp["baths"] - baths)
    if sqft is not None:
        diff += abs(comp["sqft"] - sqft) / comp["sqft"]
    return diff

@app.get("/comps/suggest", tags=["comps"])
async def comps_suggest(
    price: Optional[float] = None,
    beds: Optional[int] = None,
    baths: Optional[int] = None,
    sqft: Optional[float] = None,
    n: int = 5,
) -> dict:
    """Suggest comparable properties based on provided parameters."""
    sorted_comps = sorted(comps_data, key=lambda c: compute_similarity(c, price, beds, baths, sqft))
    return {"comps": sorted_comps[:n]}
