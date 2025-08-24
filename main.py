from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Tuple, Any
import os
import json
import time
import logging
import httpx
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

# Advanced comps scoring helpers (from your repo)
from comps_scoring import Property, find_comps, default_weights

# Attempt to import Supabase client
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None

# API schemas (from your repo)
from cma_models import Subject, CMAInput, AdjustmentInput, Comp, CMAResponse

# PDF/AI services (from your repo)
from pdf_utils import create_cma_pdf
from services.ai import rank_comparables, compute_adjusted_cma, generate_cma_summary

app = FastAPI(title="Casae API", version="0.2.3")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# ---------------- Supabase config ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

supabase: Optional["Client"] = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore
    except Exception:
        supabase = None

# ---------------- Sample dataset ----------------
comps_data = [
    {"id": 1, "address": "123 Main St", "price": 300000, "beds": 3, "baths": 2, "sqft": 1500, "year_built": 1995, "lot_size": 6000},
    {"id": 2, "address": "456 Oak Ave", "price": 320000, "beds": 4, "baths": 3, "sqft": 1800, "year_built": 2000, "lot_size": 6500},
]

# ---------------- In-memory CMA storage ----------------
cma_runs_storage: Dict[str, Dict[str, Any]] = {}


def _save_cma_run(run_id: str, subject: Property, comps: List[Tuple[Property, float]], estimate: float) -> None:
    cma_runs_storage[run_id] = {
        "subject": subject,
        "comps": comps,
        "estimate": estimate,
    }


def _load_cma_run(run_id: str) -> Optional[Dict[str, Any]]:
    return cma_runs_storage.get(run_id)


# ---------------- RentCast helper ----------------
RENTCAST_TTL_SECONDS = 15 * 60
_rentcast_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _from_cache(address: str) -> Optional[Dict[str, Any]]:
    v = _rentcast_cache.get(address.lower().strip())
    if not v:
        return None
    ts, payload = v
    if (time.time() - ts) < RENTCAST_TTL_SECONDS:
        return payload
    return None


def _to_cache(address: str, payload: Dict[str, Any]) -> None:
    _rentcast_cache[address.lower().strip()] = (time.time(), payload)


async def fetch_property_details(address: str) -> Optional[Dict[str, Any]]:
    cached = _from_cache(address)
    if cached:
        return cached

    api_key = os.getenv("RENTCAST_API_KEY")
    if not api_key:
        logger.warning("RENTCAST_API_KEY not set")
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.rentcast.io/v1/properties",
                params={"address": address},
                headers={"X-Api-Key": api_key},
            )
        if resp.status_code != 200:
            logger.warning("RentCast non-200 %s: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        candidates = None
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
            return None

        prop = candidates[0] or {}
        def _i(*keys):
            for k in keys:
                v = prop.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        continue
            return None

        payload = {
            "beds": _i("beds", "bedrooms"),
            "baths": _i("baths", "bathrooms"),
            "sqft": _i("livingArea", "sqft", "squareFootage"),
            "year_built": _i("yearBuilt"),
            "lot_sqft": _i("lotSize", "lot_sqft"),
        }
        _to_cache(address, payload)
        return payload
    except Exception as e:
        logger.exception("RentCast exception: %s", e)
        return None


# ---------------- CMA Baseline ----------------
@app.post("/cma/baseline", response_model=CMAResponse)
async def cma_baseline(payload: CMAInput) -> CMAResponse:
    s = payload.subject

    try:
        if not s.beds or not s.baths or not s.sqft:
            details = await fetch_property_details(s.address)
            if details:
                s.beds = s.beds or details.get("beds")
                s.baths = s.baths or details.get("baths")
                s.sqft = s.sqft or details.get("sqft")
                s.year_built = s.year_built or details.get("year_built")
                s.lot_sqft = s.lot_sqft or details.get("lot_sqft")
    except Exception:
        pass

    # Subject property
    subject_prop = Property(
        id="subject",
        lat=float(getattr(s, "lat", 0.0)),
        lng=float(getattr(s, "lng", 0.0)),
        property_type="SFR",
        living_sqft=float(s.sqft or 0),
        lot_sqft=float(s.lot_sqft) if s.lot_sqft else None,
        beds=int(s.beds or 0),
        baths=float(s.baths or 0),
        year_built=int(s.year_built or 0),
        condition_rating=None,
        features=set(),
        sale_date=None,
        raw_price=None,
        market_index_geo=None,
    )

    # -------------------------------------------------------------------
    # OPTIONAL: RentCast AVM value lookup
    # Commented out to avoid multiple API calls per search.
    # You can re-enable later by uncommenting this block.
    # -------------------------------------------------------------------
    #
    # use_avm = os.getenv("RENTCAST_USE_AVM") == "1"
    # if use_avm:
    #     api_key = os.getenv("RENTCAST_API_KEY")
    #     if api_key:
    #         try:
    #             async with httpx.AsyncClient(timeout=10) as client:
    #                 resp = await client.get(
    #                     "https://api.rentcast.io/v1/avm/value",
    #                     params={
    #                         "address": s.address,
    #                         "beds": s.beds or "",
    #                         "baths": s.baths or "",
    #                         "squareFootage": s.sqft or "",
    #                     },
    #                     headers={"X-Api-Key": api_key},
    #                 )
    #             if resp.status_code == 200:
    #                 data = resp.json()
    #                 rc_price = data.get("price")
    #                 if isinstance(rc_price, (int, float)):
    #                     pass
    #         except Exception:
    #             logger.exception("AVM request failed for %s", s.address)

    # Compute estimate from comps (simplified here)
    comps_list: List[Property] = []
    for entry in comps_data:
        comps_list.append(
            Property(
                id=str(entry["id"]),
                lat=0.0,
                lng=0.0,
                property_type="SFR",
                living_sqft=entry["sqft"],
                lot_sqft=entry["lot_size"],
                beds=entry["beds"],
                baths=entry["baths"],
                year_built=entry["year_built"],
                condition_rating=None,
                features=set(),
                sale_date=None,
                raw_price=entry["price"],
                market_index_geo=None,
            )
        )

    _, scored = find_comps(subject_prop, comps_list, {}, default_weights, {}, 5)
    total_price, total_weight = 0.0, 0.0
    for comp, score in scored:
        if comp.raw_price:
            total_price += comp.raw_price * max(score, 0.01)
            total_weight += max(score, 0.01)
    estimate = round(total_price / total_weight, 0) if total_weight > 0 else 0.0

    run_id = str(uuid4())
    _save_cma_run(run_id, subject_prop, scored, estimate)

    return CMAResponse(
        estimate=estimate,
        comps=[Comp(id=str(c.id), address=c.address, raw_price=c.raw_price, living_sqft=c.living_sqft,
                    beds=c.beds, baths=c.baths, year_built=c.year_built, lot_sqft=c.lot_sqft,
                    distance_mi=None, similarity=sc) for c, sc in scored],
        explanation="Estimate based on similarity-weighted comps.",
        cma_run_id=run_id,
    )


# ---------------- Debug endpoint ----------------
@app.get("/debug/rentcast")
async def debug_rentcast(address: str) -> Dict[str, Any]:
    details = await fetch_property_details(address)
    return {"address": address, "details": details}
