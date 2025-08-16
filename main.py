from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Tuple
import os
from pydantic import BaseModel
from datetime import datetime

# Import advanced comps scoring classes and helpers
from comps_scoring import Property, find_comps, default_weights

try:
    # Import the Supabase client if available.  This package should be installed
    # via requirements.txt as ``supabase-py``.  We wrap this in a try/except so
    # that the API can still start even if the library is missing (for example
    # during local development before dependencies are installed).
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore

app = FastAPI(title="Casae API", version="0.1.0")

# -----------------------------------------------------------------------------
# Supabase configuration
#
# We create a Supabase client using environment variables.  If the required
# variables are not present or the client cannot be created, ``supabase`` will
# remain ``None`` and the API will fall back to the built‑in sample comps
# dataset.  To query your real properties table, set ``SUPABASE_URL`` and
# ``SUPABASE_SERVICE_ROLE_KEY`` (or ``SUPABASE_KEY``) in your deployment
# environment.
# -----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
supabase: Optional["Client"] = None
if create_client is not None and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore
    except Exception:
        supabase = None

# -----------------------------------------------------------------------------
# Sample comps dataset
# -----------------------------------------------------------------------------
comps_data = [
    {
        "id": 1,
        "address": "123 Main St",
        "price": 300_000,
        "beds": 3,
        "baths": 2,
        "sqft": 1_500,
        "year_built": 1995,
        "lot_size": 6_000,
    },
    {
        "id": 2,
        "address": "456 Oak Ave",
        "price": 320_000,
        "beds": 4,
        "baths": 3,
        "sqft": 1_800,
        "year_built": 2000,
        "lot_size": 6_500,
    },
    {
        "id": 3,
        "address": "789 Pine Rd",
        "price": 280_000,
        "beds": 3,
        "baths": 2,
        "sqft": 1_400,
        "year_built": 1990,
        "lot_size": 5_500,
    },
    {
        "id": 4,
        "address": "101 Cedar Blvd",
        "price": 310_000,
        "beds": 4,
        "baths": 2,
        "sqft": 1_600,
        "year_built": 1980,
        "lot_size": 6_200,
    },
    {
        "id": 5,
        "address": "202 Maple St",
        "price": 295_000,
        "beds": 3,
        "baths": 2,
        "sqft": 1_200,
        "year_built": 1985,
        "lot_size": 5_000,
    },
]

# -----------------------------------------------------------------------------
# CORS configuration
#
# In production, the allowed origins will typically be configured via
# environment variables or deployment settings.  During development, we allow
# all origins so that local frontends can make requests to this API.
# -----------------------------------------------------------------------------
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
    """
    Compute a similarity score based on weighted differences.  This helper
    function is used by the `/comps/suggest` endpoint.  Scores are higher
    when the comp is closer to the subject parameters.
    """
    score = 0.0
    total_weight = 0.0

    def add_component(comp_val: float, target_val: Optional[float], weight: float) -> None:
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
    Suggest comparable properties sorted by similarity score using the in‑memory
    sample data.  The higher the score, the more similar the comp is to the
    subject parameters.
    """
    comps_with_scores = [
        (comp, compute_similarity(comp, price, beds, baths, sqft, year_built, lot_size))
        for comp in comps_data
    ]
    sorted_comps = sorted(comps_with_scores, key=lambda x: x[1], reverse=True)
    top_comps = [comp for comp, _ in sorted_comps[:n]]
    return top_comps


@app.api_route("/comps/search", methods=["GET", "POST"], tags=["comps"])
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
    Subject property details are provided via query parameters.  If a Supabase
    client is configured and available, this function will query the ``properties``
    table for candidate comps.  Otherwise it will fall back to the built‑in
    sample dataset.
    """
    # -------------------------------------------------------------------------
    # Construct the subject ``Property``.  If fields are missing, default to
    # zero or None as appropriate.
    # -------------------------------------------------------------------------
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

    comps_list: List[Property] = []

    # -------------------------------------------------------------------------
    # Attempt to fetch comps from Supabase.  We wrap this in a conditional so
    # that if the client or environment variables are missing the code still
    # executes and falls back gracefully to the local sample data.  We also
    # surround the query with try/except to catch any runtime errors from the
    # Supabase client.
    # -------------------------------------------------------------------------
    if supabase is not None:
        try:
            query = supabase.table("properties").select("*")
            # Apply basic filters when provided.  Additional filtering logic
            # (such as lat/lng radius or price windows) can be added later.
            if beds is not None:
                query = query.eq("beds", beds)
            if baths is not None:
                query = query.eq("baths", baths)
            if sqft is not None:
                # The database may store living area under ``living_sqft`` or ``sqft``.
                query = query.eq("living_sqft", sqft)
            # Execute the query.  ``execute()`` returns an object whose
            # ``data`` attribute contains the result list.
            response = query.execute()
            data = None
            # ``supabase-py`` returns ``data`` either as an attribute or in the
            # returned dictionary depending on the version.
            if hasattr(response, "data"):
                data = response.data  # type: ignore[attr-defined]
            elif isinstance(response, dict):
                data = response.get("data")
            if data:
                for entry in data:
                    comps_list.append(
                        Property(
                            id=str(entry.get("id")),
                            lat=entry.get("lat", 0.0),
                            lng=entry.get("lng", 0.0),
                            property_type=entry.get("property_type", "SFR"),
                            living_sqft=entry.get("living_sqft")
                            or entry.get("sqft", 0.0)
                            or 0.0,
                            lot_sqft=entry.get("lot_size"),
                            beds=entry.get("beds") or 0,
                            baths=entry.get("baths") or 0,
                            year_built=entry.get("year_built"),
                            condition_rating=entry.get("condition_rating"),
                            features=set(entry.get("features"))
                            if entry.get("features")
                            else set(),
                            sale_date=None,
                            raw_price=entry.get("raw_price") or entry.get("price"),
                            market_index_geo=None,
                        )
                    )
        except Exception:
            # If anything goes wrong (e.g. network error, invalid response),
            # ignore Supabase results and fall back to the local sample data.
            comps_list = []

    # -------------------------------------------------------------------------
    # If Supabase returned no comps or the client was unavailable, use the
    # built‑in sample data to ensure the endpoint still responds.  This helps
    # development and acts as a last‑resort fallback in production if the
    # database is unreachable.
    # -------------------------------------------------------------------------
    if not comps_list:
        for entry in comps_data:
            comps_list.append(
                Property(
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
            )

    # -------------------------------------------------------------------------
    # Score the candidate comps using the advanced scoring algorithm.  We use an
    # empty ``filters`` dictionary (no hard filters) and an empty market index.
    # The ``return_limit`` parameter ensures we only take the top ``n`` comps.
    # -------------------------------------------------------------------------
    filters: Dict[str, float] = {}
    market_index: Dict[Tuple[str, str], float] = {}
    _, scored = find_comps(subject, comps_list, filters, default_weights, market_index, n)

    results: List[Dict] = []
    for comp, score in scored:
        comp_dict = comp.__dict__.copy()
        # Convert the features set back to a list for JSON serialization
        comp_dict["features"] = list(comp_dict.get("features", []))
        comp_dict["similarity"] = score
        results.append(comp_dict)

    return {"results": results}

# -----------------------------------------------------------------------------
# Saved searches endpoints
#
# These endpoints allow the frontend to save and list users' saved search
# criteria.  The ``/searches/save`` endpoint accepts a POST request with
# ``user_id`` and ``params`` and stores the search in the ``saved_searches``
# table.  The ``/searches/list`` endpoint accepts a GET request with a
# ``user_id`` query parameter and returns all saved searches for that user.
# -----------------------------------------------------------------------------

class SaveSearchRequest(BaseModel):
    user_id: str
    params: Dict

@app.post("/searches/save", tags=["searches"])
async def save_search(request: SaveSearchRequest) -> Dict[str, str]:
    """
    Save a search for the given user.  Requires Supabase to be configured.
    
    Parameters:
      user_id: identifier of the user saving the search
      params: dictionary of search parameters

    Returns a status object or an error message if Supabase is not available or
    the insert fails.
    """
    if supabase is None:
        return {"error": "supabase not configured"}
    try:
        payload = {
            "user_id": request.user_id,
            "params": request.params,
            "created_at": datetime.utcnow().isoformat(),
        }
        supabase.table("saved_searches").insert(payload).execute()
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/searches/list", tags=["searches"])
async def list_saved_searches(user_id: str) -> Dict[str, List[Dict]]:
    """
    List all saved searches for a given user.  If Supabase is not available
    returns an empty list.
    """
    if supabase is None:
        return {"results": []}
    try:
        query = supabase.table("saved_searches").select("*").eq("user_id", user_id)
        response = query.execute()
        data: Optional[List[Dict]] = None
        if hasattr(response, "data"):
            data = response.data  # type: ignore[attr-defined]
        elif isinstance(response, dict):
            data = response.get("data")
        return {"results": data or []}
    except Exception:
        return {"results": []}