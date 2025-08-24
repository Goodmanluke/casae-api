from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Tuple, Any
from typing import List as _ListType
import os
import json
import time
import logging
import httpx
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4
from fastapi.responses import StreamingResponse  # if PDF endpoints are used
from io import BytesIO

# Advanced comps scoring helpers (existing in your repo)
from comps_scoring import Property, find_comps, default_weights

# Attempt to import Supabase client; fall back gracefully if unavailable
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore

# API schemas (existing in your repo)
from cma_models import Subject, CMAInput, AdjustmentInput, Comp, CMAResponse

# PDF and AI services (existing in your repo)
from pdf_utils import create_cma_pdf
from services.ai import rank_comparables, compute_adjusted_cma, generate_cma_summary

app = FastAPI(title="Casae API", version="0.2.2")

# ------------------------------- CORS ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Logging --------------------------------
logger = logging.getLogger("uvicorn.error")

# -------------------------- Supabase config ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

supabase: Optional["Client"] = None
if create_client is not None and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore
    except Exception:
        supabase = None

# -------------------------- Sample dataset ----------------------------
comps_data = [
    {"id": 1, "address": "123 Main St", "price": 300_000, "beds": 3, "baths": 2, "sqft": 1_500, "year_built": 1995, "lot_size": 6_000},
    {"id": 2, "address": "456 Oak Ave", "price": 320_000, "beds": 4, "baths": 3, "sqft": 1_800, "year_built": 2000, "lot_size": 6_500},
    {"id": 3, "address": "789 Pine Rd", "price": 280_000, "beds": 3, "baths": 2, "sqft": 1_400, "year_built": 1990, "lot_size": 5_500},
    {"id": 4, "address": "101 Cedar Blvd", "price": 310_000, "beds": 4, "baths": 2, "sqft": 1_600, "year_built": 1980, "lot_size": 6_200},
    {"id": 5, "address": "202 Maple St", "price": 295_000, "beds": 3, "baths": 2, "sqft": 1_200, "year_built": 1985, "lot_size": 5_000},
]

# ------------------------ In-memory CMA storage -----------------------
cma_runs_storage: Dict[str, Dict[str, Any]] = {}


def _save_cma_run(
    run_id: str,
    subject: Property,
    comps: List[Tuple[Property, float]],
    estimate: float,
    baseline: bool = True,
    adjustments: Optional[Dict[str, Any]] = None,
) -> None:
    cma_runs_storage[run_id] = {
        "subject": subject,
        "comps": comps,
        "estimate": estimate,
        "baseline": baseline,
        "adjustments": adjustments,
    }


def _load_cma_run(run_id: str) -> Optional[Dict[str, Any]]:
    return cma_runs_storage.get(run_id)

# ------------------------- RentCast helper ----------------------------
RENTCAST_TTL_SECONDS = 15 * 60  # cache same address for 15 minutes
_rentcast_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # addr -> (ts, details)


def _from_cache(address: str) -> Optional[Dict[str, Any]]:
    v = _rentcast_cache.get(address.lower().strip())
    if not v:
        return None
    ts, payload = v
    if (time.time() - ts) < RENTCAST_TTL_SECONDS:
        return payload
    _rentcast_cache.pop(address.lower().strip(), None)
    return None


def _to_cache(address: str, payload: Dict[str, Any]) -> None:
    _rentcast_cache[address.lower().strip()] = (time.time(), payload)


async def fetch_property_details(address: str) -> Optional[Dict[str, Any]]:
    """
    Fetch beds, baths, sqft, year_built, lot_sqft for address from RentCast.
    Handles multiple response shapes and logs issues for debugging.
    Caches results for a short TTL to avoid redundant calls.
    """
    cached = _from_cache(address)
    if cached is not None:
        return cached

    api_key = os.getenv("RENTCAST_API_KEY")
    if not api_key:
        logger.warning("RENTCAST_API_KEY not set – skipping RentCast lookup")
        return None

    params = {"address": address}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.rentcast.io/v1/properties",
                params=params,
                headers={"X-Api-Key": api_key},
            )
        if resp.status_code != 200:
            logger.warning("RentCast non‑200 %s: %s", resp.status_code, resp.text[:300])
            return None

        data = resp.json()
        candidates = None
        # rentcast wraps response different ways
        if isinstance(data, dict):
            if isinstance(data.get("properties"), list):
                candidates = data["properties"]
            elif isinstance(data.get("results"), list):
                candidates = data["results"]
            elif isinstance(data.get("data"), list):
                candidates = data["data"]
            elif any(k in data for k in ("beds", "bedrooms", "sqft", "livingArea")):
                candidates = [data]

        if not candidates:
            logger.warning(
                "RentCast unexpected JSON shape: %s",
                list(data.keys()) if isinstance(data, dict) else type(data),
            )
            return None

        prop = candidates[0] or {}

        def _i(*keys):
            for k in keys:
                v = prop.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        try:
                            return int(float(v))
                        except Exception:
                            continue
            return None

        beds = _i("beds", "bedrooms")
        baths = _i("baths", "bathrooms")
        sqft = _i("livingArea", "sqft", "squareFootage")
        year_built = _i("yearBuilt", "year_built")
        lot_sqft = _i("lotSize", "lot_sqft", "lotSizeSqft")

        if any(v is not None for v in (beds, baths, sqft, year_built, lot_sqft)):
            payload = {
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
                "year_built": year_built,
                "lot_sqft": lot_sqft,
            }
            _to_cache(address, payload)
            return payload
        else:
            logger.warning("RentCast returned no usable fields for %s: %s", address, json.dumps(prop)[:300])
            return None

    except Exception as e:
        logger.exception("RentCast exception for %s: %s", address, e)
        return None


# --------------------------- /comps/search ----------------------------
@app.api_route("/comps/search", methods=["GET", "POST"])
async def comps_search(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    price: Optional[float] = None,
    beds: Optional[int] = None,
    baths: Optional[float] = None,
    sqft: Optional[int] = None,
    year_built: Optional[int] = None,
    lot_size: Optional[int] = None,
    n: int = 5,
) -> Dict[str, List[Dict]]:
    subject = Property(
        id="subject",
        lat=lat or 0.0,
        lng=lng or 0.0,
        property_type="SFR",
        living_sqft=sqft or 0.0,
        lot_sqft=lot_size,
        beds=beds or 0,
        baths=baths or 0.0,
        year_built=year_built,
        condition_rating=None,
        features=set(),
        sale_date=None,
        raw_price=price,
        market_index_geo=None,
    )

    comps_list: List[Property] = []

    if supabase is not None:
        try:
            query = supabase.table("properties").select("*")
            if beds is not None:
                query = query.eq("beds", beds)
            if baths is not None:
                query = query.eq("baths", baths)
            if sqft is not None:
                query = query.eq("living_sqft", sqft)
            response = query.execute()
            data = getattr(response, "data", None) if not isinstance(response, dict) else response.get("data")
            if data:
                for entry in data:
                    comps_list.append(
                        Property(
                            id=str(entry.get("id")),
                            lat=entry.get("lat", 0.0),
                            lng=entry.get("lng", 0.0),
                            property_type=entry.get("property_type", "SFR"),
                            living_sqft=entry.get("living_sqft") or entry.get("sqft", 0.0) or 0.0),
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


# --------------------------- Saved Searches ---------------------------
class SaveSearchRequest(BaseModel):
    user_id: str
    params: Dict


@app.post("/searches/save")
async def save_search(request: SaveSearchRequest) -> Dict[str, str]:
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


@app.get("/searches/list")
async def list_saved_searches(user_id: str) -> Dict[str, List[Dict]]:
    if supabase is None:
        return {"results": []}
    try:
        query = supabase.table("saved_searches").select("*").eq("user_id", user_id)
        response = query.execute()
        data: Optional[List[Dict]] = getattr(response, "data", None) if not isinstance(response, dict) else response.get("data")
        return {"results": data or []}
    except Exception:
        return {"results": []}


# --------------------------- CMA Baseline -----------------------------
@app.post("/cma/baseline", response_model=CMAResponse)
async def cma_baseline(payload: CMAInput) -> CMAResponse:
    s = payload.subject

    # Step 1: RentCast autofill when any of the key fields are missing
    try:
        if (not s.beds or s.beds == 0) or (not s.baths or s.baths == 0) or (not s.sqft or s.sqft == 0):
            details = await fetch_property_details(s.address)
            if details:
                if (not s.beds or s.beds == 0) and details.get("beds") is not None:
                    s.beds = details["beds"]
                if (not s.baths or s.baths == 0) and details.get("baths") is not None:
                    s.baths = details["baths"]
                if (not s.sqft or s.sqft == 0) and details.get("sqft") is not None:
                    s.sqft = details["sqft"]
                if details.get("year_built") is not None:
                    s.year_built = s.year_built or details["year_built"]
                if details.get("lot_sqft") is not None:
                    s.lot_sqft = s.lot_sqft or details["lot_sqft"]
    except Exception:
        pass

    # Step 2: Supabase fallback
    if supabase is not None and ((not s.beds) or (not s.baths) or (not s.sqft)):
        try:
            query = supabase.table("properties").select("beds,baths,living_sqft,year_built,lot_size")
            if getattr(s, "lat", None):
                query = query.eq("lat", s.lat)
            if getattr(s, "lng", None):
                query = query.eq("lng", s.lng)
            if getattr(s, "address", None):
                query = query.eq("address", s.address)
            response = query.limit(1).execute()
            data = getattr(response, "data", None) if not isinstance(response, dict) else response.get("data")
            if data:
                row = data[0]
                if not s.beds and row.get("beds") is not None:
                    s.beds = row.get("beds")
                if not s.baths and row.get("baths") is not None:
                    s.baths = row.get("baths")
                if not s.sqft and row.get("living_sqft") is not None:
                    s.sqft = row.get("living_sqft")
                if not s.year_built and row.get("year_built") is not None:
                    s.year_built = row.get("year_built")
                if not s.lot_sqft and row.get("lot_size") is not None:
                    s.lot_sqft = row.get("lot_size")
        except Exception:
            pass

    # Build subject property for comps selection (safe defaults)
    _beds = int(s.beds or 0)
    _baths = float(s.baths or 0.0)
    _sqft = float(s.sqft or 0)

    subject_prop = Property(
        id="subject",
        lat=float(getattr(s, "lat", 0.0) or 0.0),
        lng=float(getattr(s, "lng", 0.0) or 0.0),
        property_type="SFR",
        living_sqft=_sqft,
        lot_sqft=float(s.lot_sqft) if getattr(s, "lot_sqft", None) is not None else None,
        beds=_beds,
        baths=_baths,
        year_built=int(getattr(s, "year_built", 0) or 0) if getattr(s, "year_built", None) is not None else None,
        condition_rating=None,
        features=set(),
        sale_date=None,
        raw_price=None,
        market_index_geo=None,
    )

    # Optional: AVM value lookup (set RENTCAST_USE_AVM=1 or payload.rules["use_avm"] to enable)
    use_avm = os.getenv("RENTCAST_USE_AVM") == "1" or bool(getattr(payload, "rules", {}).get("use_avm", False))
    if use_avm:
        api_key = os.getenv("RENTCAST_API_KEY")
        if api_key:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        "https://api.rentcast.io/v1/avm/value",
                        params={
                            "address": s.address,
                            "beds": s.beds or "",
                            "baths": s.baths or "",
                            "squareFootage": s.sqft or "",
                        },
                        headers={"X-Api-Key": api_key},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    rc_price = data.get("price")
                    if isinstance(rc_price, (int, float)) and rc_price > 0:
                        # Optionally, use rc_price as an external check / narrative add-on.
                        pass
                else:
                    logger.warning("AVM non-200 %s: %s", resp.status_code, resp.text[:200])
            except Exception:
                logger.exception("AVM request failed for address=%s", s.address)

    # Otherwise compute internal comps & estimate
    comps_list: List[Property] = []
    if supabase is not None:
        try:
            response = supabase.table("properties").select("*").eq("beds", s.beds).execute()
            data = getattr(response, "data", None) if not isinstance(response, dict) else response.get("data")
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
        comps=[_to_comp_model(c, sc) for c, sc in scored],
        explanation="Estimate based on similarity-weighted top comps.",
        cma_run_id=run_id,
    )


# ----------------------------- GET wrapper ----------------------------
@app.get("/cma/baseline", response_model=CMAResponse)
async def cma_baseline_get(
    address: str,
    beds: int = 0,
    baths: float = 0.0,
    sqft: int = 0,
    lat: float = 0.0,
    lng: float = 0.0,
    year_built: Optional[int] = None,
    lot_sqft: Optional[int] = None,
) -> CMAResponse:
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
    payload = CMAInput(subject=subject)
    return await cma_baseline(payload)


# ------------------------------ /cma/adjust ---------------------------
@app.post("/cma/adjust", response_model=CMAResponse)
async def cma_adjust(payload: AdjustmentInput) -> CMAResponse:
    baseline = _load_cma_run(payload.cma_run_id)
    if not baseline:
        return CMAResponse(estimate=0.0, comps=[], explanation="Invalid cma_run_id", cma_run_id=payload.cma_run_id)

    subject_prop: Property = baseline["subject"]
    comps = baseline["comps"]
    est = baseline["estimate"]

    uplift = 0.0
    uplift += 0.0 if not payload.condition else _condition_uplift(payload.condition)
    uplift += _renovations_uplift(payload.renovations or [])
    if payload.add_beds > 0 or payload.add_sqft > 300:
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
                response = supabase.table("properties").select("*").eq("beds", updated_subject.beds).execute()
                data = getattr(response, "data", None) if not isinstance(response, dict) else response.get("data")
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
    else:
        new_comps = comps

    new_estimate = round(est * (1.0 + uplift), 0)
    new_run_id = str(uuid4())
    _save_cma_run(new_run_id, baseline["subject"], new_comps, new_estimate, baseline=False, adjustments=payload.model_dump())

    return CMAResponse(
        estimate=new_estimate,
        comps=[_to_comp_model(c, sc) for c, sc in new_comps],
        explanation="Adjusted estimate based on requested changes.",
        cma_run_id=new_run_id,
    )


# ------------------------------ Debug ---------------------------------
@app.get("/debug/rentcast")
async def debug_rentcast(address: str) -> Dict[str, Any]:
    details = await fetch_property_details(address)
    return {"address": address, "details": details}
