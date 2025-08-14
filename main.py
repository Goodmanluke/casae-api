from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict

# Import advanced comps scoring classes and helpers
from comps_scoring import Property, find_comps, default_weights

# Initialize FastAPI application
app = FastAPI(title="Casae API", version="0.1.0")

# Example comps dataset with year built and lot size (sqft)
comps_data = [
    {
        "id": 1,
        "address": "123 Main St",
        "price": 300000,
        "beds": 3,
        "baths": 2,
        "sqft": 1500,
        "year_built": 1995,
        "lot_size": 6000,
    },
    {
        "id": 2,
        "address": "456 Oak Ave",
        "price": 320000,
        "beds": 4,
        "baths": 3,
        "sqft": 1800,
        "year_built": 2000,
        "lot_size": 6500,
    },
    {
        "id": 3,
        "address": "789 Pine Rd",
        "price": 280000,
        "beds": 3,
        "baths": 2,
        "sqft": 1400,
        "year_built": 1990,
        "lot_size": 5500,
    },
    {
        "id": 4,
        "address": "101 Cedar Blvd",
        "price": 310000,
        "beds": 4,
        "baths": 2,
        "sqft": 1600,
        "year_built": 1980,
        "lot_size": 6200,
    },
    {
        "id": 5,
        "address": "202 Maple St",
        "price": 295000,
        "beds": 3,
        "baths": 2,
        "sqft": 1200,
        "year_built": 1985,
        "lot_size": 5000,
    },
]

# CORS configuration will be overridden via environment variables in deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/", tags=["root"])
async def root() -> Dict[str, str]:
    return {"message": "Welcome to Casae API"}


def compute_similarity(
    comp: dict,
    price: Optional[float],
    beds: Optional[int],
    baths: Optional[int],
    sqft: Optional[int],
    year_built: Optional[int],
    lot_size: Optional[int],
) -> float:
    """Compute a similarity score based on weighted differences.
    Scores are higher when the comp is closer to the subject parameters.
    """
    score = 0.0
    total_weight = 0.0

    def add_component(comp_val: float, target_val: Optional[float], weight: float):
        nonlocal score, total_weight
        if target_val is not None and target_val > 0:
            diff = abs(comp_val - target_val)
            normalized_diff = diff / target_val
            component_score = max(0.0, 1 - normalized_diff)
            score += component_score * weight
            total_weight += weight

    # Weights sum to 1.0
    add_component(comp["price"], price, 0.3)
    add_component(comp["beds"], beds, 0.15)
    add_component(comp["baths"], baths, 0.15)
    add_component(comp["sqft"], sqft, 0.2)
    add_component(comp["year_built"], year_built, 0.1)
    add_component(comp["lot_size"], lot_size, 0.1)

    return score / total_weight if total_weight > 0 else 0.0


@app.get("/comps/suggest", tags=["comps"])
async def comps_suggest(
    price: Optional[float] = None,
    beds: Optional[int] = None,
    baths: Optional[int] = None,
    sqft: Optional[int] = None,
    year_built: Optional[int] = None,
    lot_size: Optional[int] = None,
    n: int = 5,
) -> List[dict]:
    """
    Suggest comparable properties sorted by similarity score.
    The higher the score, the more similar the comp is to the subject property.
    """
    comps_with_scores = [
        (comp, compute_similarity(comp, price, beds, baths, sqft, year_built, lot_size))
        for comp in comps_data
    ]
    sorted_comps = sorted(comps_with_scores, key=lambda x: x[1], reverse=True)
    top_comps = [comp for comp, _ in sorted_comps[:n]]
    return top_comps


@app.post("/comps/search", tags=["comps"])
async def comps_search(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    price: Optional[float] = None,
    beds: Optional[int] = None,
    baths: Optional[int] = None,
    sqft: Optional[float] = None,
    year_built: Optional[int] = None,
    lot_size: Optional[int] = None,
    n: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Search for comparable properties using the advanced comps scoring algorithm.
    Provide subject property details and return the top n comps with similarity scores.
    """
    # Construct the subject Property object; default to zero or None where data is missing
    subject = Property(
        id="subject",
        lat=lat or 0.0,
        lng=lng or 0.0,
        property_type="SFR",
        living_sqft=sqft or 0.0,
        lot_sqft=lot_size,
        beds=beds or 0,
        baths=baths or 0,
        year_built=year_built,
        condition_rating=None,
        features=set(),
        sale_date=None,
        raw_price=price,
        market_index_geo=None,
    )

    # Convert comps_data entries to Property objects
    comps_list: List[Property] = []
    for entry in comps_data:
        comp = Property(
            id=str(entry.get("id")),
            lat=entry.get("lat", 0.0),
            lng=entry.get("lng", 0.0),
            property_type="SFR",
            living_sqft=entry.get("sqft", 0.0),
            lot_sqft=entry.get("lot_size"),
            beds=entry.get("beds"),
            baths=entry.get("baths"),
            year_built=entry.get("year_built"),
            condition_rating=None,
            features=set(),
            sale_date=None,
            raw_price=entry.get("price"),
            market_index_geo=None,
        )
        comps_list.append(comp)

    # Use advanced comps scoring to find the most similar comps
    scored = find_comps(subject, comps_list, n, weights=default_weights)

    # Build response: convert dataclass to dict and include similarity score
    results: List[Dict] = []
    for comp, score in scored:
        comp_dict = comp.__dict__.copy()
        comp_dict["features"] = list(comp_dict["features"])
        comp_dict["similarity"] = score
        results.append(comp_dict)

    return {"results": results}
