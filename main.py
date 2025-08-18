
from fastapi import FastAPI
from fastapi.middleware.cors import 
from typing import Optional, List, Dict, Tuple, Any
import os
import httpx
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

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

# Import CMA schemas
from cma_models import Subject, CMAInput, AdjustmentInput, Comp, CMAResponse, SummaryRequest
from fastapi.responses import StreamingResponse
from io import BytesIO
from pdf_utils import create_cma_pdf
from services.ai import rank_comparables, compute_adjusted_cma, generate_cma_summary

app = FastAPI(title="Casae API", version="0.2.0")

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
# In-memory storage for CMA runs
#
# For demonstration purposes, we store CMA runs in a global dictionary keyed by
# the run ID.  In a production deployment this should be replaced with
# persistent storage (e.g. Supabase or another database).
# -----------------------------------------------------------------------------
cma_runs_storage: Dict[str, Dict[str, Any]] = {}

def _save_cma_run(run_id: str, subject: Property, comps: List[Tuple[Property, float]], estimate: float, baseline: bool = True, adjustments: Optional[Dict[str, Any]] = None) -> None:
    """Save a CMA run in the in-memory storage."""
    cma_runs_storage[run_id] = {
        "subject": subject,
        "comps": comps,
        "estimate": estimate,
        "baseline": baseline,
        "adjustments": adjustments,
    }

def _load_cma_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a saved CMA run from in-memory storage."""
    return cma_runs_storage.get(run_id)

# Adjustment helper functions
def _condition_uplift(condition: Optional[str]) -> float:
    """Return the percentage uplift based on condition."""
    if not condition:
        return 0.0
    mapping = {"poor": -0.08, "fair": -0.03, "good": 0.0, "very_good": 0.03, "excellent": 0.06}
    return mapping.get(condition.lower(), 0.0)

def _renovations_uplift(items: List[str]) -> float:
    """Return the percentage uplift based on renovation items."""
    weights = {"kitchen": 0.06, "bath": 0.05, "flooring": 0.02, "roof": 0.01, "dock": 0.03, "hvac": 0.01}
    return sum(weights.get(item.lower(), 0.0) for item in items)

def _additions_uplift(add_beds: int, add_baths: int, add_sqft: int, comps: list, dock_length: int = 0) -> float:
    uplift = 0.0

    # Bedrooms / baths
    uplift += 0.025 * max(0, int(add_beds or 0))
    uplift += 0.020 * max(0, int(add_baths or 0))

    # Sqft
    if add_sqft and add_sqft > 0:
        uplift += min(0.10, 0.00006 * add_sqft)

    # Dock length
    if dock_length and dock_length > 0:
        dock_uplift = 0.001 * dock_length   # 0.1% per foot
        uplift += min(dock_uplift, 0.03)    # cap 3%

    return uplift


def _to_comp_model(comp: Property, similarity: float) -> Comp:
    """Convert internal Property to the API Comp model."""
    return Comp(
        id=str(comp.id),
        address=getattr(comp, "address", ""),
        raw_price=comp.raw_price or 0.0,
        living_sqft=int(comp.living_sqft or 0),
        beds=comp.beds or 0,
        baths=float(comp.baths or 0),
        year_built=comp.year_built,
        lot_sqft=comp.lot_sqft,
        distance_mi=None,
        similarity=similarity,
    )

# -----------------------------------------------------------------------------
# CORS configuration
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

# -----------------------------------------------------------------------------
# CMA endpoints
#
# These endpoints implement the baseline and adjustment flows for AI-powered CMA.
# They make use of the in-memory comps scoring engine and the adjustment helpers
# defined above.  In a production environment these should save CMA runs and
# associated comps to a persistent datastore (e.g. Supabase).
# -----------------------------------------------------------------------------

@app.post("/cma/baseline", response_model=CMAResponse, tags=["cma"])
async def cma_baseline(payload: CMAInput) -> CMAResponse:
    """
    Create a baseline CMA for the given subject property.
    Returns the estimated value and the list of top comps.
    """
    s = payload.subject
    # Construct the internal Property for the subject
    subject_prop = Property(
        id="subject",
        lat=s.lat,
        lng=s.lng,
        property_type="SFR",
        living_sqft=s.sqft,
        lot_sqft=s.lot_sqft,
        beds=s.beds,
        baths=int(s.baths),
        year_built=s.year_built,
        condition_rating=None,
        features=set(),
        sale_date=None,
        raw_price=None,
        market_index_geo=None,
    )

    # --- RentCast AVM path ---
    rentcast_api_key = os.getenv("RENTCAST_API_KEY")
    if rentcast_api_key:
        params = {
            "address": s.address,
            "beds": s.beds or "",
            "baths": s.baths or "",
            "squareFootage": s.sqft or "",
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.rentcast.io/v1/avm/value",
                    params=params,
                    headers={"X-Api-Key": rentcast_api_key},
                )
            if resp.status_code == 200:
                data = resp.json()
                rc_price = data.get("price")
                rc_comps = data.get("comparables", []) or []
                if rc_price and rc_comps:
                    comps_list: List[Property] = []
                    for comp in rc_comps:
                        comps_list.append(
                            Property(
                                id=str(comp.get("id", "rentcast")),
                                address=comp.get("formattedAddress") or "",
                                lat=comp.get("latitude"),
                                lng=comp.get("longitude"),
                                property_type="SFR",
                                living_sqft=comp.get("squareFootage") or 0.0,
                                lot_sqft=comp.get("lotSize"),
                                beds=comp.get("bedrooms"),
                                baths=comp.get("bathrooms"),
                                year_built=comp.get("yearBuilt"),
                                condition_rating=None,
                                features=set(),
                                sale_date=None,
                                raw_price=comp.get("price"),
                                market_index_geo=None,
                            )
                        )

                    estimate = round(rc_price or 0)

                    # --- AI: rank comps + reasoning ---
                    subject_dict = {
                        "address": getattr(s, "address", ""),
                        "lat": s.lat,
                        "lng": s.lng,
                        "beds": s.beds,
                        "baths": s.baths,
                        "living_sqft": s.sqft,
                        "lot_sqft": s.lot_sqft,
                        "year_built": s.year_built,
                        "avm_value": estimate,
                    }
                    comp_dicts = []
                    for c in comps_list:
                        comp_dicts.append({
                            "id": getattr(c, "id", "rc"),
                            "address": getattr(c, "address", ""),
                            "lat": c.lat,
                            "lng": c.lng,
                            "property_type": c.property_type,
                            "living_sqft": c.living_sqft,
                            "lot_sqft": c.lot_sqft,
                            "beds": c.beds,
                            "baths": c.baths,
                            "year_built": c.year_built,
                            "raw_price": c.raw_price,
                        })

                    ranked = await rank_comparables(subject_dict, comp_dicts)
                    explanation = ranked.get("reasoning", "Estimate from RentCast AVM.")

                    # Optional: reorder comps by AI-suggested order
                    ordered_ids = [d.get("id") for d in (ranked.get("comps") or [])]
                    if ordered_ids:
                        id2comp = {getattr(c, "id", str(i)): c for i, c in enumerate(comps_list)}
                        comps_list = [id2comp[i] for i in ordered_ids if i in id2comp] or comps_list

                    cma_run_id = str(uuid4())
                    await _save_cma_run(
                        s,
                        cma_run_id,
                        {"source": "rentcast"},
                        comps_list,
                        estimate,
                        explanation,
                    )
                    return CMAResponse(
                        estimate=estimate,
                        comps=comps_list,
                        explanation=explanation,
                        cma_run_id=cma_run_id,
                    )
        except Exception:
            # In case of any error from RentCast, continue with internal comps logic
            pass

    # --- Supabase / fallback comps path ---
    comps_list: List[Property] = []
    if supabase is not None:
        try:
            query = supabase.table("properties").select("*")
            query = query.eq("beds", s.beds)
            response = query.execute()
            data = None
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
                            living_sqft=entry.get("living_sqft") or entry.get("sqft", 0.0) or 0.0,
                            lot_sqft=entry.get("lot_size"),
                            beds=entry.get("beds") or 0,
                            baths=entry.get("baths") or 0,
                            year_built=entry.get("year_built"),
                            condition_rating=entry.get("condition_rating"),
                            features=set(entry.get("features")) if entry.get("features") else set(),
                            sale_date=None,
                            raw_price=entry.get("raw_price") or entry.get("price"),
                            market_index_geo=None,
                        )
                    )
        except Exception:
            comps_list = []

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

    # --- Advanced comps scoring engine ---
    filters: Dict[str, float] = {}
    market_index: Dict[Tuple[str, str], float] = {}
    _, scored = find_comps(
        subject_prop,
        comps_list,
        filters,
        default_weights,
        market_index,
        payload.rules.get("n", 8) if isinstance(payload.rules.get("n", None), int) else 8
    )

    # Compute estimate as similarity-weighted average of raw prices
    total_price = 0.0
    total_weight = 0.0
    for comp, score in scored:
        if comp.raw_price:
            weight = max(score, 0.01)
            total_price += comp.raw_price * weight
            total_weight += weight
    estimate = round(total_price / total_weight, 0) if total_weight > 0 else 0.0

    # --- AI reasoning + optional reorder ---
    subject_dict = {
        "address": getattr(s, "address", ""),
        "lat": s.lat,
        "lng": s.lng,
        "beds": s.beds,
        "baths": s.baths,
        "living_sqft": s.sqft,
        "lot_sqft": s.lot_sqft,
        "year_built": s.year_built,
        "avm_value": estimate,
    }
    comp_dicts = []
    for comp, score in scored:
        comp_dicts.append({
            "id": getattr(comp, "id", ""),
            "address": getattr(comp, "address", ""),
            "lat": comp.lat,
            "lng": comp.lng,
            "property_type": comp.property_type,
            "living_sqft": comp.living_sqft,
            "lot_sqft": comp.lot_sqft,
            "beds": comp.beds,
            "baths": comp.baths,
            "year_built": comp.year_built,
            "raw_price": comp.raw_price,
        })

    ranked2 = await rank_comparables(subject_dict, comp_dicts)
    explanation2 = ranked2.get("reasoning", "Estimate based on similarity-weighted top comps.")

    # Optional reorder of scored comps
    ordered_ids2 = [d.get("id") for d in (ranked2.get("comps") or [])]
    if ordered_ids2:
        id2tuple = {getattr(comp, "id", str(i)): (comp, score) for i, (comp, score) in enumerate(scored)}
        new_scored = [id2tuple[i] for i in ordered_ids2 if i in id2tuple]
        if new_scored:
            scored = new_scored

    run_id = str(uuid4())
    _save_cma_run(run_id, subject_prop, scored, estimate, baseline=True)

    return CMAResponse(
        estimate=estimate,
        comps=[_to_comp_model(comp, score) for comp, score in scored],
        explanation=explanation2,
        cma_run_id=run_id,
    )



@app.post("/cma/adjust", response_model=CMAResponse, tags=["cma"])
async def cma_adjust(payload: AdjustmentInput) -> CMAResponse:
    """
    Recompute the CMA after user adjustments (condition, renos, sqft/beds/baths, etc.).
    Uses AI to estimate the adjusted value and (optionally) suggest a refined comp list.
    """
    # Fetch the baseline run
    baseline = _get_cma_run(payload.cma_run_id)
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline CMA run not found.")

    # Baseline pieces
    subject_prop = baseline["subject"]            # Property model used in baseline
    baseline_comps = baseline["comps"]           # List[Tuple[Property, float]] as (comp, score)
    baseline_estimate = baseline.get("estimate") # float

    # Prepare compact dicts for the AI
    subject_dict: Dict[str, Any] = {
        "address": getattr(subject_prop, "address", ""),
        "lat": subject_prop.lat,
        "lng": subject_prop.lng,
        "beds": subject_prop.beds,
        "baths": subject_prop.baths,
        "living_sqft": subject_prop.living_sqft,
        "lot_sqft": subject_prop.lot_sqft,
        "year_built": subject_prop.year_built,
        "avm_value": baseline_estimate,
    }

    comp_dicts: List[Dict[str, Any]] = []
    for comp, score in baseline_comps:
        comp_dicts.append({
            "id": getattr(comp, "id", ""),
            "address": getattr(comp, "address", ""),
            "lat": comp.lat,
            "lng": comp.lng,
            "property_type": comp.property_type,
            "living_sqft": comp.living_sqft,
            "lot_sqft": comp.lot_sqft,
            "beds": comp.beds,
            "baths": comp.baths,
            "year_built": comp.year_built,
            "raw_price": comp.raw_price,
        })

    # Convert pydantic model -> dict (supports pydantic v2 .model_dump())
    try:
        adjustments_dict: Dict[str, Any] = payload.model_dump()  # type: ignore[attr-defined]
    except Exception:
        adjustments_dict = payload.dict()  # pydantic v1 fallback

    # Ask AI for adjusted valuation (with optional new comp suggestions)
    ai = await compute_adjusted_cma(subject_dict, comp_dicts, adjustments_dict)
    new_estimate = round(ai.get("value", baseline_estimate) or baseline_estimate, 0)
    explanation = ai.get("reasoning", "Adjusted estimate based on requested changes.")

    # Optionally adopt AI-suggested comps by id, mapping back to (Property, score)
    new_comps_tuples = baseline_comps
    ai_comps = ai.get("comps")
    if isinstance(ai_comps, list) and ai_comps:
        id2orig: Dict[str, Tuple[Property, float]] = {
            getattr(comp, "id", ""): (comp, score) for comp, score in baseline_comps
        }
        mapped: List[Tuple[Property, float]] = []
        for c in ai_comps:
            cid = (c or {}).get("id")
            if cid and cid in id2orig:
                mapped.append(id2orig[cid])
        if mapped:
            new_comps_tuples = mapped

    # Persist a new CMA run (child of baseline) and return
    new_run_id = str(uuid4())
    # Note: _save_cma_run is synchronous in the advanced baseline path; use the same style here.
    # If your local signature differs, keep param order consistent with your implementation.
    _save_cma_run(
        new_run_id,
        subject_prop,
        new_comps_tuples,
        new_estimate,
        baseline=False,
        adjustments=adjustments_dict,
    )

    return CMAResponse(
        estimate=new_estimate,
        comps=[_to_comp_model(comp, score) for comp, score in new_comps_tuples],
        explanation=explanation,
        cma_run_id=new_run_id,
    )



@app.post("/pdf", tags=["cma"])
async def generate_pdf(cma_run_id: str) -> StreamingResponse:
    """
    Generate and return a PDF report for the given CMA run.

    This endpoint looks up the CMA run stored in memory and generates a
    simple PDF that includes the subject's estimated value and the list
    of comparable properties.  The PDF is returned directly in the
    response as a streaming download.  If the requested run ID does
    not exist, a 404 PDF is returned.
    """
    cma_run = _load_cma_run(cma_run_id)
    if not cma_run:
        # If the run is missing, return a tiny PDF with a "not found" message
        empty_pdf = create_cma_pdf(cma_run_id, {"subject": None, "estimate": 0.0, "comps": []})
        return StreamingResponse(BytesIO(empty_pdf), media_type="application/pdf", status_code=404)
    # Build the PDF bytes using our utility
    pdf_bytes = create_cma_pdf(cma_run_id, cma_run)
    filename = f"cma_{cma_run_id}.pdf"
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename={filename}"
    })
    
@app.get("/vendor/rentcast/value", tags=["vendor", "rentcast"])
async def rentcast_value(address: str, beds: Optional[int] = None, baths: Optional[float] = None, sqft: Optional[int] = None):
    """
    Call RentCast AVM to retrieve property value estimate and comparables.
    """
    rentcast_api_key = os.getenv("RENTCAST_API_KEY")
if rentcast_api_key:
    params = {
        "address": s.address,
        "beds": s.beds or "",
        "baths": s.baths or "",
        "squareFootage": s.sqft or "",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.rentcast.io/v1/avm/value",
                params=params,
                headers={"X-Api-Key": rentcast_api_key},
            )
        if resp.status_code == 200:
            data = resp.json()
            rc_price = data.get("price")
            rc_comps = data.get("comparables", [])
            if rc_price and rc_comps:
                # map comparables into your Property model
                comps_list: List[Property] = []
                for comp in rc_comps:
                    comps_list.append(
                        Property(
                            address=comp.get("formattedAddress"),
                            lat=comp.get("latitude"),
                            lng=comp.get("longitude"),
                            beds=comp.get("bedrooms"),
                            baths=comp.get("bathrooms"),
                            sqft=comp.get("squareFootage"),
                            raw_price=comp.get("price"),
                            # add other fields as needed
                        )
                    )
                estimate = round(rc_price or 0)
                cma_run_id = str(uuid4())
                await _save_cma_run(
                    s,
                    cma_run_id,
                    {"source": "rentcast"},
                    comps_list,
                    estimate,
                    "Estimate from RentCast AVM",
                )
                return CMAResponse(
                    estimate=estimate,
                    comps=comps_list,
                    explanation="Estimate from RentCast AVM.",
                    cma_run_id=cma_run_id,
                )
    except Exception:
        pass

        return response.json()


class SummaryRequest(BaseModel):
    """Request body for generating a CMA summary narrative."""
    subject: Dict[str, Any]
    comps: List[Dict[str, Any]]
    adjustments: Dict[str, Any]
    value: float


@app.post("/cma/summary", tags=["cma"])
async def cma_summary(payload: SummaryRequest) -> Dict[str, str]:
    """Generate a CMA narrative summary using the AI."""
    summary = await generate_cma_summary(payload.subject, payload.comps, payload.adjustments, payload.value)
    return {"summary": summary}

