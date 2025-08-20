from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Tuple, Any
from typing import List as _ListType
import os
import httpx
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

# Import advanced comps scoring classes and helpers
from comps_scoring import Property, find_comps, default_weights

# Attempt to import Supabase client; fall back gracefully if unavailable
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore

# Import CMA schemas and utilities
from cma_models import Subject, CMAInput, AdjustmentInput, Comp, CMAResponse
from fastapi.responses import StreamingResponse
from io import BytesIO
from pdf_utils import create_cma_pdf
from services.ai import rank_comparables, compute_adjusted_cma, generate_cma_summary

# Initialize FastAPI app
app = FastAPI(title="Casae API", version="0.2.0")

# ----------------------------------------------------------------------------
# Supabase configuration
#
# We create a Supabase client using environment variables. If the required
# variables are not present or the client cannot be created, ``supabase`` will
# remain ``None`` and the API will fall back to the built‑in sample comps
# dataset. To query your real properties table, set ``SUPABASE_URL`` and
# ``SUPABASE_SERVICE_ROLE_KEY`` (or ``SUPABASE_KEY``) in your deployment
# environment.
# ----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
supabase: Optional["Client"] = None
if create_client is not None and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore
    except Exception:
        supabase = None

# ----------------------------------------------------------------------------
# Sample comps dataset used as a fallback when no database is configured
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# In-memory storage for CMA runs
#
# For demonstration purposes, we store CMA runs in a global dictionary keyed by
# the run ID. In a production deployment this should be replaced with
# persistent storage (e.g. Supabase or another database).
# ----------------------------------------------------------------------------
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
    """Return the percentage uplift based on additions (beds, baths, sqft, dock)."""
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

# ----------------------------------------------------------------------------
# CORS configuration
#
# In production, the allowed origins will typically be configured via
# environment variables or deployment settings. During development, we allow
# all origins so that local frontends can make requests to this API.
# ----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Basic endpoints: health and root
# ----------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/", tags=["root"])
async def root() -> Dict[str, str]:
    return {"message": "Welcome to Casae API"}

# ----------------------------------------------------------------------------
# Comps suggestion and search endpoints
# ----------------------------------------------------------------------------
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
    Compute a similarity score based on weighted differences. Scores are higher
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
    sample data. The higher the score, the more similar the comp is to the
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
    Subject property details are provided via query parameters. If a Supabase
    client is configured and available, this function will query the
    ``properties`` table for candidate comps. Otherwise it will fall back to
    the built‑in sample dataset.
    """
    # Construct the subject Property
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

    # Attempt to fetch comps from Supabase
    if supabase is not None:
        try:
            query = supabase.table("properties").select("*")
            # Apply filters
            if beds is not None:
                query = query.eq("beds", beds)
            if baths is not None:
                query = query.eq("baths", baths)
            if sqft is not None:
                query = query.eq("living_sqft", sqft)
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

    # If no comps from Supabase, use sample data
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

    # Score comps using advanced scoring
    filters: Dict[str, float] = {}
    market_index: Dict[Tuple[str, str], float] = {}
    _, scored = find_comps(subject, comps_list, filters, default_weights, market_index, n)

    results: List[Dict] = []
    for comp, score in scored:
        comp_dict = comp.__dict__.copy()
        comp_dict["features"] = list(comp_dict.get("features", []))
        comp_dict["similarity"] = score
        results.append(comp_dict)
    return {"results": results}

# ----------------------------------------------------------------------------
# Saved searches endpoints
# ----------------------------------------------------------------------------
class SaveSearchRequest(BaseModel):
    user_id: str
    params: Dict

@app.post("/searches/save", tags=["searches"])
async def save_search(request: SaveSearchRequest) -> Dict[str, str]:
    """
    Save a search for the given user. Requires Supabase to be configured.

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
    List all saved searches for a given user. If Supabase is not available
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

# ----------------------------------------------------------------------------
# CMA endpoints: baseline and adjustments
# ----------------------------------------------------------------------------
@app.post("/cma/baseline", response_model=CMAResponse, tags=["cma"])
async def cma_baseline(payload: CMAInput) -> CMAResponse:
    """
    Create a baseline CMA for the given subject property.
    Returns the estimated value and the list of top comps.
    """
    s = payload.subject
        # Auto-fill missing property details using Supabase if available
    if supabase is not None and (
        (s.beds is None or s.beds == 0) or
        (s.baths is None or s.baths == 0) or
        (s.sqft is None or s.sqft == 0)
    ):
        try:
            query = supabase.table("properties").select("beds,baths,living_sqft")
            # Filter by lat and lng to identify the subject property
            if s.lat is not None and s.lat != 0:
                query = query.eq("lat", s.lat)
            if s.lng is not None and s.lng != 0:
                query = query.eq("lng", s.lng)
            # Optionally also filter by address if provided
            if hasattr(s, "address") and s.address:
                query = query.eq("address", s.address)
            response = query.limit(1).execute()
            data = None
            if hasattr(response, "data"):
                data = response.data  # type: ignore[attr-defined]
            elif isinstance(response, dict):
                data = response.get("data")
            if data:
                entry = data[0]
                # Update missing or zero values with fetched values
                if (s.beds is None or s.beds == 0) and entry.get("beds") is not None:
                    s.beds = entry.get("beds")
                if (s.baths is None or s.baths == 0) and entry.get("baths") is not None:
                    s.baths = entry.get("baths")
                if (s.sqft is None or s.sqft == 0) and entry.get("living_sqft") is not None:
                    s.sqft = entry.get("living_sqft")
        except Exception:
            # Ignore errors when attempting to fetch details
            pass

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
    # Use RentCast AVM if API key available to compute estimate and comps.
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
                    cma_run_id = str(uuid4())
                    # Persist the CMA run and comps for audit
                    _save_cma_run(
                        cma_run_id,
                        subject_prop,
                        [(c, 1.0) for c in comps_list],  # treat each comp equally when using RentCast
                        estimate,
                        baseline=True,
                        adjustments=None,
                    )
                    return CMAResponse(
                        estimate=estimate,
                        comps=comps_list,
                        explanation="Estimate from RentCast AVM.",
                        cma_run_id=cma_run_id,
                    )
        except Exception:
            # In case of any error from RentCast, continue with internal comps logic
            pass

    # Prepare candidate comps (reuse logic from comps_search)
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

    # Use advanced comps scoring engine
    filters: Dict[str, float] = {}
    market_index: Dict[Tuple[str, str], float] = {}
    _, scored = find_comps(
        subject_prop,
        comps_list,
        filters,
        default_weights,
        market_index,
        payload.rules.get("n", 8) if isinstance(payload.rules.get("n", None), int) else 8,
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

    run_id = str(uuid4())
    _save_cma_run(run_id, subject_prop, scored, estimate, baseline=True)

    return CMAResponse(
        estimate=estimate,
        comps=[_to_comp_model(comp, score) for comp, score in scored],
        explanation="Estimate based on similarity-weighted top comps.",
        cma_run_id=run_id,
    )

@app.get("/cma/baseline", response_model=CMAResponse, tags=["cma"])
async def cma_baseline_get(
    address: str,
    beds: int,
    baths: float,
    sqft: int,
    lat: float = 0.0,
    lng: float = 0.0,
    year_built: Optional[int] = None,
    lot_sqft: Optional[int] = None,
) -> CMAResponse:
    """GET wrapper for baseline CMA; constructs CMAInput and delegates to POST logic."""
    subject = Subject(
        address=address,
        lat=lat,
        lng=lng,
        beds=beds,
        baths=baths,
        sqft=sqft,
        year_built=year_built,
        lot_sqft=lot_sqft,
    )
    payload = CMAInput(subject=subject, rules={})
    return await cma_baseline(payload)

@app.post("/cma/adjust", response_model=CMAResponse, tags=["cma"])
async def cma_adjust(payload: AdjustmentInput) -> CMAResponse:
    """
    Adjust a previous CMA run based on changes to the property.
    Applies uplift factors and optionally reselects comps if the property class changes.
    """
    baseline = _load_cma_run(payload.cma_run_id)
    if not baseline:
        return CMAResponse(
            estimate=0.0,
            comps=[],
            explanation="Invalid cma_run_id",
            cma_run_id=payload.cma_run_id,
        )
    est = baseline["estimate"]
    comps = baseline["comps"]
    # Calculate uplift
    uplift = 0.0
    uplift += _condition_uplift(payload.condition)
    uplift += _renovations_uplift(payload.renovations or [])
    uplift += _additions_uplift(
        payload.add_beds,
        payload.add_baths,
        payload.add_sqft,
        comps,
        getattr(payload, "dock_length", 0) or 0,
    )
    uplift = min(uplift, 0.35)  # cap uplift at 35%
    new_estimate = round(est * (1.0 + uplift), 0)
    new_comps = comps
    # Reselect comps if adding bedrooms or significant sqft
    if payload.add_beds > 0 or payload.add_sqft > 300:
        subject_prop: Property = baseline["subject"]
        updated_subject = Property(
            id=subject_prop.id,
            lat=subject_prop.lat,
            lng=subject_prop.lng,
            property_type=subject_prop.property_type,
            living_sqft=subject_prop.living_sqft + payload.add_sqft,
            lot_sqft=subject_prop.lot_sqft,
            beds=subject_prop.beds + payload.add_beds,
            baths=subject_prop.baths + int(payload.add_baths),
            year_built=subject_prop.year_built,
            condition_rating=subject_prop.condition_rating,
            features=subject_prop.features,
            sale_date=None,
            raw_price=None,
            market_index_geo=None,
        )
        comps_list: List[Property] = []
        if supabase is not None:
            try:
                query = supabase.table("properties").select("*").eq("beds", updated_subject.beds)
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
        _, rescored = find_comps(updated_subject, comps_list, {}, default_weights, {}, len(comps))
        new_comps = rescored
    new_run_id = str(uuid4())
    _save_cma_run(
        new_run_id,
        baseline["subject"],
        new_comps,
        new_estimate,
        baseline=False,
        adjustments=payload.model_dump(),
    )
    return CMAResponse(
        estimate=new_estimate,
        comps=[_to_comp_model(comp, score) for comp, score in new_comps],
        explanation="Adjusted estimate based on requested changes.",
        cma_run_id=new_run_id,
    )

@app.get("/cma/adjust", response_model=CMAResponse, tags=["cma"])
async def cma_adjust_get(
    cma_run_id: str,
    condition: Optional[str] = None,
    renovations: Optional[_ListType[str]] = Query(
        None,
        description=(
            "Repeat param, e.g. ?renovations=kitchen&renovations=bath OR pass comma-separated in 'renovations_csv'"
        ),
    ),
    renovations_csv: Optional[str] = None,
    add_beds: int = 0,
    add_baths: float = 0.0,
    add_sqft: int = 0,
    dock_length: int = 0,
) -> CMAResponse:
    """GET wrapper that builds AdjustmentInput and delegates to POST logic."""
    renos: List[str] = renovations or []
    if renovations_csv:
        renos += [r.strip() for r in renovations_csv.split(",") if r.strip()]
    payload = AdjustmentInput(
        cma_run_id=cma_run_id,
        condition=condition,
        renovations=renos,
        add_beds=add_beds,
        add_baths=add_baths,
        add_sqft=add_sqft,
        dock_length=dock_length,
    )
    return await cma_adjust(payload)

# ----------------------------------------------------------------------------
# PDF endpoints
# ----------------------------------------------------------------------------
@app.get("/pdfx", tags=["cma"])
async def pdfx(cma_run_id: str, request: Request) -> Dict[str, str]:
    """
    Returns a direct link the browser can fetch.
    It points at the streaming /pdf endpoint with the run id as a query param.
    """
    return {"url": f"/pdf?cma_run_id={cma_run_id}"}

@app.post("/pdf", tags=["cma"])
async def generate_pdf(cma_run_id: str) -> StreamingResponse:
    """
    Generate and return a PDF report for the given CMA run.

    This endpoint looks up the CMA run stored in memory and generates a
    simple PDF that includes the subject's estimated value and the list
    of comparable properties. The PDF is returned directly in the
    response as a streaming download. If the requested run ID does
    not exist, a 404 PDF is returned.
    """
    cma_run = _load_cma_run(cma_run_id)
    if not cma_run:
        empty_pdf = create_cma_pdf(cma_run_id, {"subject": None, "estimate": 0.0, "comps": []})
        return StreamingResponse(BytesIO(empty_pdf), media_type="application/pdf", status_code=404)
    pdf_bytes = create_cma_pdf(cma_run_id, cma_run)
    filename = f"cma_{cma_run_id}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

# ----------------------------------------------------------------------------
# RentCast vendor endpoint
# ----------------------------------------------------------------------------
@app.get("/vendor/rentcast/value", tags=["vendor", "rentcast"])
async def rentcast_value(address: str, beds: Optional[int] = None, baths: Optional[float] = None, sqft: Optional[int] = None):
    """
    Call RentCast AVM to retrieve property value estimate and comparables.
    """
    rentcast_api_key = os.getenv("RENTCAST_API_KEY")
    if not rentcast_api_key:
        return {"error": "RentCast API key not configured"}
    params = {
        "address": address,
        "beds": beds or "",
        "baths": baths or "",
        "squareFootage": sqft or "",
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
                cma_run_id = str(uuid4())
                return CMAResponse(
                    estimate=estimate,
                    comps=comps_list,
                    explanation="Estimate from RentCast AVM.",
                    cma_run_id=cma_run_id,
                )
        return {"error": "Unable to fetch RentCast data"}
    except Exception:
        return {"error": "Error calling RentCast"}

# ----------------------------------------------------------------------------
# CMA narrative summary endpoint
# ----------------------------------------------------------------------------
class SummaryRequest(BaseModel):
    """Request body for generating a CMA summary narrative."""
    subject: Dict[str, Any]
    comps: List[Dict[str, Any]]
    adjustments: Dict[str, Any]
    value: float

@app.post("/cma/summary", tags=["cma"])
async def cma_summary(payload: SummaryRequest) -> Dict[str, str]:
    """Generate a CMA narrative summary using the AI."""
    summary = await generate_cma_summary(
        payload.subject,
        payload.comps,
        payload.adjustments,
        payload.value,
    )
    return {"summary": summary}
