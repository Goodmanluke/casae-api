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
import math

from dotenv import load_dotenv
load_dotenv()

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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


# ---------------- Utility functions ----------------
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    R = 3959  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def generate_ai_narrative(subject: Property, estimate: float, comps: List[Tuple[Property, float]], is_adjusted: bool = False, adjustments: Dict[str, Any] = None) -> str:
    """Generate AI-powered narrative for the CMA"""
    if not comps:
        return "Insufficient comparable data available for analysis."
    
    avg_price = sum(c[0].raw_price or 0 for c in comps) / len(comps)
    price_range = max(c[0].raw_price or 0 for c in comps) - min(c[0].raw_price or 0 for c in comps)
    
    narrative = f"Based on analysis of {len(comps)} comparable properties, "
    
    if is_adjusted and adjustments:
        narrative += f"the adjusted valuation of ${estimate:,.0f} reflects "
        if adjustments.get('condition'):
            narrative += f"a {adjustments['condition'].lower()} condition rating, "
        if adjustments.get('renovations'):
            narrative += f"recent {', '.join(adjustments['renovations'])} renovations, "
        if adjustments.get('add_beds', 0) > 0:
            narrative += f"an additional {adjustments['add_beds']} bedroom(s), "
        if adjustments.get('add_baths', 0) > 0:
            narrative += f"an additional {adjustments['add_baths']} bathroom(s), "
        if adjustments.get('add_sqft', 0) > 0:
            narrative += f"an additional {adjustments['add_sqft']} square feet, "
    else:
        narrative += f"the estimated market value of ${estimate:,.0f} is "
    
    if estimate > avg_price * 1.1:
        narrative += "above the local market average, suggesting premium features or superior condition."
    elif estimate < avg_price * 0.9:
        narrative += "below the local market average, potentially indicating needed updates or market positioning."
    else:
        narrative += "in line with local market conditions."
    
    narrative += f" The comparable properties range from ${min(c[0].raw_price or 0 for c in comps):,.0f} to ${max(c[0].raw_price or 0 for c in comps):,.0f}, "
    narrative += f"with an average of ${avg_price:,.0f}. "
    
    if price_range > avg_price * 0.3:
        narrative += "The wide price range suggests diverse property conditions and features in the area."
    else:
        narrative += "The tight price range indicates consistent property values in this neighborhood."
    
    return narrative


def get_property_photo_url(address: str) -> str:
    """Get property photo URL. Uses Google Street View if key is present, else placeholder."""
    try:
        gkey = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("NEXT_PUBLIC_GOOGLE_MAPS_KEY")
        if gkey:
            from urllib.parse import quote
            loc = quote(address)
            return f"https://maps.googleapis.com/maps/api/streetview?size=640x400&location={loc}&fov=80&heading=70&pitch=0&key={gkey}"
    except Exception:
        pass
    return f"https://via.placeholder.com/640x400/4F46E5/FFFFFF?text={address.split(',')[0]}"


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
    logger.info(f"[RentCast] API key present: {bool(api_key)}")
    if not api_key:
        logger.warning("[RentCast] RENTCAST_API_KEY not set")
        return None

    try:
        logger.info(f"[RentCast] Making API call for address: {address}")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.rentcast.io/v1/properties",
                params={"address": address},
                headers={"X-Api-Key": api_key},
            )
        logger.info(f"[RentCast] API response status: {resp.status_code}")
        
        if resp.status_code != 200:
            logger.warning(f"[RentCast] Non-200 response: {resp.status_code}")
            logger.warning(f"[RentCast] Response text: {resp.text[:200]}")
            return None

        data = resp.json()
        logger.info(f"[RentCast] Raw API response for {address}: {data}")
        
        candidates = None
        if isinstance(data, list):
            # Direct list response (as seen in logs)
            candidates = data
            logger.info(f"[RentCast] Found direct list response with {len(data)} items")
        elif isinstance(data, dict):
            if isinstance(data.get("properties"), list):
                candidates = data["properties"]
                logger.info(f"[RentCast] Found properties list with {len(data['properties'])} items")
            elif isinstance(data.get("results"), list):
                candidates = data["results"]
                logger.info(f"[RentCast] Found results list with {len(data['results'])} items")
            elif isinstance(data.get("data"), list):
                candidates = data["data"]
                logger.info(f"[RentCast] Found data list with {len(data['data'])} items")
            elif any(k in data for k in ("propertyType", "bedrooms", "bathrooms", "squareFootage")):
                candidates = [data]
                logger.info(f"[RentCast] Found single property data")

        if not candidates:
            logger.warning(f"[RentCast] No candidates found for {address}")
            return None

        prop = candidates[0] or {}
        logger.info(f"[RentCast] First candidate property: {prop}")
        logger.info(f"[RentCast] Available keys in property: {list(prop.keys()) if isinstance(prop, dict) else 'Not a dict'}")
        
        def _i(*keys):
            for k in keys:
                v = prop.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        continue
            return None

        def _f(*keys):
            for k in keys:
                v = prop.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        continue
            return None

        # Try multiple field names for each property attribute
        payload = {
            "beds": _i("beds", "bedrooms", "bedroomCount", "bedroom_count"),
            "baths": _f("baths", "bathrooms", "bathroomCount", "bathroom_count", "fullBathrooms", "full_bathrooms"),
            "sqft": _i("livingArea", "sqft", "squareFootage", "squareFootage", "living_sqft", "livingSqft", "totalSqft", "total_sqft"),
            "year_built": _i("yearBuilt", "year_built", "constructionYear", "construction_year", "builtYear", "built_year"),
            "lot_sqft": _i("lotSize", "lot_sqft", "lotSizeSqft", "lot_size_sqft", "acres", "lotAcres"),
            "lat": _f("latitude", "lat"),
            "lng": _f("longitude", "lng"),
        }
        
        # Log the extracted payload for debugging
        logger.info(f"[RentCast] Extracted payload for {address}: {payload}")
        
        # Also log individual field extraction attempts
        logger.info(f"[RentCast] Field extraction details:")
        logger.info(f"  - beds: {_i('beds', 'bedrooms', 'bedroomCount', 'bedroom_count')}")
        logger.info(f"  - baths: {_f('baths', 'bathrooms', 'bathroomCount', 'bathroom_count', 'fullBathrooms', 'full_bathrooms')}")
        logger.info(f"  - sqft: {_i('livingArea', 'sqft', 'squareFootage', 'squareFootage', 'living_sqft', 'livingSqft', 'totalSqft', 'total_sqft')}")
        logger.info(f"  - year_built: {_i('yearBuilt', 'year_built', 'constructionYear', 'construction_year', 'builtYear', 'built_year')}")
        
        # Log the final payload
        logger.info(f"[RentCast] Final payload: {payload}")
        
        _to_cache(address, payload)
        return payload
    except Exception as e:
        logger.exception("RentCast exception: %s", e)
        return None


# ---------------- CMA Baseline ----------------
@app.post("/cma/baseline", response_model=CMAResponse)
async def cma_baseline(payload: CMAInput) -> CMAResponse:
    s = payload.subject
    logger.info("[CMA Baseline] address=%s", getattr(s, "address", None))

    try:
        if not s.beds or not s.baths or not s.sqft:
            logger.info(f"[CMA Baseline] Fetching property details for {s.address}")
            details = await fetch_property_details(s.address)
            if details:
                logger.info(f"[CMA Baseline] Received details: {details}")
                s.beds = s.beds or details.get("beds")
                s.baths = s.baths or details.get("baths")
                s.sqft = s.sqft or details.get("sqft")
                s.year_built = s.year_built or details.get("year_built")
                s.lot_sqft = s.lot_sqft or details.get("lot_sqft")
                s.lat = s.lat or details.get("lat")
                s.lng = s.lng or details.get("lng")
                logger.info(f"[CMA Baseline] Updated subject: beds={s.beds}, baths={s.baths}, sqft={s.sqft}, year_built={s.year_built}")
            else:
                logger.warning(f"[CMA Baseline] No property details found for {s.address}")
                # Fallback to sample data if no details found
                logger.info(f"[CMA Baseline] Using fallback sample data")
                s.beds = s.beds or 3
                s.baths = s.baths or 2
                s.sqft = s.sqft or 1500
                s.year_built = s.year_built or 2000
                s.lot_sqft = s.lot_sqft or 6000
        else:
            logger.info(f"[CMA Baseline] Using provided subject details: beds={s.beds}, baths={s.baths}, sqft={s.sqft}, year_built={s.year_built}")
    except Exception as e:
        logger.exception(f"[CMA Baseline] Error fetching property details: {e}")
        # Fallback to sample data on error
        logger.info(f"[CMA Baseline] Using fallback sample data due to error")
        s.beds = s.beds or 3
        s.baths = s.baths or 2
        s.sqft = s.sqft or 1500
        s.year_built = s.year_built or 2000
        s.lot_sqft = s.lot_sqft or 6000
        pass

    # Subject property
    subject_prop = Property(
        id="subject",
        address=s.address,  # Add address
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
    
    # Log the subject property details for debugging
    logger.info(f"[CMA Baseline] Subject property details: beds={s.beds}, baths={s.baths}, sqft={s.sqft}, year_built={s.year_built}")

    # -------------------------------------------------------------------
    # OPTIONAL: RentCast AVM value/comparables lookup
    # Controlled via RENTCAST_USE_AVM=1 and RENTCAST_API_KEY
    # -------------------------------------------------------------------
    use_avm = os.getenv("RENTCAST_USE_AVM") == "1"
    avm_estimate: Optional[float] = None
    avm_comparables: Optional[list] = None
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
                    price_key_candidates = ["price", "value", "estimate"]
                    for k in price_key_candidates:
                        v = data.get(k)
                        if isinstance(v, (int, float)):
                            avm_estimate = float(v)
                            break
                    # some schemas embed comparables under different keys
                    for ck in ("comparables", "comps", "sales", "data"):
                        if isinstance(data.get(ck), list) and data.get(ck):
                            avm_comparables = data[ck]
                            break
            except Exception:
                logger.exception("AVM request failed for %s", s.address)

    # Compute estimate from comps (simplified here)
    comps_list: List[Property] = []
    if avm_comparables:
        def _gi(d: dict, *keys: str) -> Optional[int]:
            for k in keys:
                v = d.get(k)
                if v is None:
                    continue
                try:
                    return int(v)
                except Exception:
                    try:
                        return int(float(v))
                    except Exception:
                        continue
            return None
        def _gf(d: dict, *keys: str) -> Optional[float]:
            for k in keys:
                v = d.get(k)
                if v is None:
                    continue
                try:
                    return float(v)
                except Exception:
                    continue
            return None
        def _gp(d: dict) -> Optional[float]:
            """Robust price getter across possible schemas."""
            price_keys = [
                "raw_price", "price", "salePrice", "listPrice", "soldPrice", "sold_price",
                "lastSalePrice", "last_sold_price", "amount", "amountValue", "value",
                "estimate", "estimatedValue", "avm"
            ]
            for k in price_keys:
                v = d.get(k)
                if v is None:
                    continue
                try:
                    return float(v)
                except Exception:
                    try:
                        return float(str(v).replace(",", ""))
                    except Exception:
                        continue
            return None
        for comp in avm_comparables:
            comps_list.append(
                Property(
                    id=str(comp.get("id") or comp.get("propertyId") or comp.get("mlsId") or uuid4()),
                    address=str(comp.get("address") or comp.get("formattedAddress") or "Unknown Address"),
                    lat=float(_gf(comp, "lat", "latitude") or 0.0),
                    lng=float(_gf(comp, "lng", "longitude") or 0.0),
                    property_type=str(comp.get("propertyType") or "SFR"),
                    living_sqft=float(_gi(comp, "livingArea", "squareFootage", "sqft") or 0),
                    lot_sqft=float(_gi(comp, "lotSize", "lot_sqft") or 0) or None,
                    beds=int(_gi(comp, "beds", "bedrooms") or 0),
                    baths=float(_gf(comp, "baths", "bathrooms") or 0),
                    year_built=int(_gi(comp, "yearBuilt") or 0),
                    condition_rating=None,
                    features=set(),
                    sale_date=None,
                    raw_price=_gp(comp),
                    market_index_geo=None,
                )
            )
    else:
        for entry in comps_data:
            comps_list.append(
                Property(
                    id=str(entry["id"]),
                    address=entry["address"],  # Add address from sample data
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
    if avm_estimate is not None and avm_estimate > 0:
        estimate = float(avm_estimate)

    run_id = str(uuid4())
    _save_cma_run(run_id, subject_prop, scored, estimate)

    # Calculate distances and generate narrative
    comps_with_distance = []
    for comp, score in scored:
        try:
            distance = None
            if subject_prop.lat and subject_prop.lng and comp.lat and comp.lng:
                distance = calculate_distance(subject_prop.lat, subject_prop.lng, comp.lat, comp.lng)
            
            comps_with_distance.append(Comp(
                id=str(comp.id),
                address=getattr(comp, 'address', 'Unknown Address'),  # Safe access to address
                raw_price=comp.raw_price,
                living_sqft=comp.living_sqft,
                beds=comp.beds,
                baths=comp.baths,
                year_built=comp.year_built,
                lot_sqft=comp.lot_sqft,
                distance_mi=distance,
                similarity=score,
                photo_url=get_property_photo_url(getattr(comp, 'address', 'Unknown Address'))
            ))
        except Exception as e:
            logger.error(f"[CMA Baseline] Error processing comp {comp.id}: {e}")
            continue

    narrative = generate_ai_narrative(subject_prop, estimate, scored)

    try:
        return CMAResponse(
            estimate=estimate,
            subject=s,
            comps=comps_with_distance,
            explanation=narrative,
            cma_run_id=run_id,
        )
    except Exception as e:
        logger.error(f"[CMA Baseline] Error creating response: {e}")
        # Return a minimal response to prevent 500 error
        return CMAResponse(
            estimate=estimate or 0,
            subject=s,
            comps=[],
            explanation="Error processing property details. Please try again.",
            cma_run_id=run_id,
        )


# ---------------- Debug endpoint ----------------
@app.get("/debug/rentcast")
async def debug_rentcast(address: str) -> Dict[str, Any]:
    details = await fetch_property_details(address)
    return {"address": address, "details": details}

# ---------------- Test CORS endpoint ----------------
@app.get("/test")
async def test_cors() -> Dict[str, str]:
    return {"message": "CORS is working!", "status": "success"}


# ---------------- Rent estimate endpoint ----------------
@app.get("/rent/estimate")
async def rent_estimate(address: str) -> Dict[str, Any]:
    """Fetch estimated monthly rent for an address from RentCast."""
    api_key = os.getenv("RENTCAST_API_KEY")
    if not api_key:
        return {"monthly_rent": None, "error": "RENTCAST_API_KEY not set"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.rentcast.io/v1/avm/rent/long-term",
                params={"address": address},
                headers={"X-Api-Key": api_key},
            )
        if resp.status_code != 200:
            return {"monthly_rent": None, "error": f"status {resp.status_code}", "details": resp.text[:200]}

        data = resp.json()
        # Response may be an object or a list
        candidate = None
        if isinstance(data, list) and data:
            candidate = data[0]
        elif isinstance(data, dict):
            candidate = data
        else:
            return {"monthly_rent": None}

        # Try multiple keys for rent
        for k in [
            "rent", "monthlyRent", "rentEstimate", "estimatedRent", "amount", "value"
        ]:
            v = candidate.get(k)
            if v is not None:
                try:
                    return {"monthly_rent": float(v)}
                except Exception:
                    try:
                        return {"monthly_rent": float(str(v).replace(",", ""))}
                    except Exception:
                        continue

        return {"monthly_rent": None}
    except Exception as e:
        logger.exception("Rent estimate exception: %s", e)
        return {"monthly_rent": None, "error": "exception"}

# ---------------- Saved searches (placeholder) ----------------
@app.get("/searches/list")
async def list_saved_searches(user_id: str) -> Dict[str, Any]:
    # TODO: replace with real storage (e.g., Supabase table)
    # For now, return an empty list to avoid 404s on dashboard load
    return {"results": []}
