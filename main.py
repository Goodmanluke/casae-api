from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Tuple, Any
import os
import json
import time
import logging
import httpx
from pydantic import BaseModel
from datetime import datetime, date
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

# ---------------- RentCast AVM Configuration for Maximum Accuracy ----------------
class RentCastConfig:
    """Configuration class for RentCast AVM parameters to maximize accuracy"""
    
    # Default parameters based on RentCast documentation and website defaults
    DEFAULT_COMP_COUNT = 20    # More comparables = better accuracy
    DEFAULT_MAX_RADIUS = 5     # 5-mile radius (RentCast website default)
    DEFAULT_DAYS_OLD = 270     # 270 days lookback (RentCast website default)
    
    # Property type mapping for consistent API calls
    PROPERTY_TYPE_MAP = {
        "SFR": "Single Family",
        "Single Family": "Single Family", 
        "Condo": "Condo",
        "Townhouse": "Townhouse",
        "Multi-Family": "Multi-Family",
        "Apartment": "Apartment"
    }
    
    @classmethod
    def get_avm_params(cls, address: str, property_type: str = None, 
                      bedrooms: int = None, bathrooms: float = None, 
                      square_footage: int = None) -> dict:
        """Build optimized AVM parameters for maximum accuracy"""
        params = {
            "address": address,
            "lookupSubjectAttributes": "true",  # Enable automatic property lookup
            "compCount": cls.DEFAULT_COMP_COUNT,
            "maxRadius": cls.DEFAULT_MAX_RADIUS,
            "daysOld": cls.DEFAULT_DAYS_OLD,
        }
        
        # Add property details if available - these override automatic lookup
        if property_type:
            params["propertyType"] = cls.PROPERTY_TYPE_MAP.get(property_type, property_type)
        if bedrooms and bedrooms > 0:
            params["bedrooms"] = bedrooms
        if bathrooms and bathrooms > 0:
            params["bathrooms"] = bathrooms
        if square_footage and square_footage > 0:
            params["squareFootage"] = square_footage
            
        return params

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


def _get_cma_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Get a CMA run by ID (alias for _load_cma_run for consistency)"""
    return _load_cma_run(run_id)


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
    logger.info(f"[CMA Baseline] Processing address: {s.address}")

    # Fetch property details from RentCast
    property_details = await fetch_property_details(s.address)
    
    # Create subject property object with safe defaults
    subject_prop = Property(
        id=str(uuid4()),
        address=s.address,
        lat=s.lat or (property_details.get("lat") if property_details else 0.0),
        lng=s.lng or (property_details.get("lng") if property_details else 0.0),
        property_type=s.property_type or "SFR",
        living_sqft=s.sqft or (property_details.get("sqft") if property_details else 0.0),
        lot_sqft=s.lot_sqft or (property_details.get("lot_sqft") if property_details else None),
        beds=s.beds or (property_details.get("beds") if property_details else 0),
        baths=s.baths or (property_details.get("baths") if property_details else 0),
        year_built=s.year_built or (property_details.get("year_built") if property_details else None),
        condition_rating=s.condition,
        features=set(),
        sale_date=None,
        raw_price=None,
    )
    
    logger.info(f"[CMA Baseline] Created subject property: living_sqft={subject_prop.living_sqft}, beds={subject_prop.beds}, baths={subject_prop.baths}")

    # -------------------------------------------------------------------
    # ENHANCED: RentCast AVM value/comparables lookup with accuracy improvements
    # Controlled via RENTCAST_USE_AVM=1 and RENTCAST_API_KEY
    # -------------------------------------------------------------------
    use_avm = os.getenv("RENTCAST_USE_AVM") == "1"
    avm_estimate: Optional[float] = None
    avm_comparables: Optional[list] = None
    if use_avm:
        api_key = os.getenv("RENTCAST_API_KEY")
        if api_key:
            try:
                logger.info(f"[AVM] Making enhanced AVM request for: {s.address}")
                
                # Build enhanced parameters using configuration class
                avm_params = RentCastConfig.get_avm_params(
                    address=s.address,
                    property_type=s.property_type,
                    bedrooms=s.beds,
                    bathrooms=s.baths,
                    square_footage=s.sqft
                )
                    
                # Enhance with property_details from RentCast if user didn't provide complete info
                if property_details:
                    if not s.beds and property_details.get("beds"):
                        avm_params["bedrooms"] = property_details["beds"]
                    if not s.baths and property_details.get("baths"):
                        avm_params["bathrooms"] = property_details["baths"]
                    if not s.sqft and property_details.get("sqft"):
                        avm_params["squareFootage"] = property_details["sqft"]
                
                logger.info(f"[AVM] Request parameters: {avm_params}")
                
                async with httpx.AsyncClient(timeout=15) as client:  # Increased timeout for more accurate results
                    resp = await client.get(
                        "https://api.rentcast.io/v1/avm/value",
                        params=avm_params,
                        headers={"X-Api-Key": api_key},
                    )
                    
                logger.info(f"[AVM] Response status: {resp.status_code}")
                
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(f"[AVM] Raw response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # Enhanced price extraction - RentCast uses 'price' as primary field
                    price_key_candidates = ["price", "value", "estimate", "priceEstimate"]
                    for k in price_key_candidates:
                        v = data.get(k)
                        if isinstance(v, (int, float)) and v > 0:
                            avm_estimate = float(v)
                            logger.info(f"[AVM] Found estimate: ${avm_estimate:,.0f} from field '{k}'")
                            break
                    
                    # Extract comparables with enhanced handling
                    for ck in ("comparables", "comps", "sales", "data"):
                        avm_comps_data = data.get(ck)  # Renamed to avoid variable conflict
                        if isinstance(avm_comps_data, list) and avm_comps_data:
                            avm_comparables = avm_comps_data
                            logger.info(f"[AVM] Found {len(avm_comparables)} comparables from field '{ck}'")
                            break
                    
                    # Log subject property info if returned (for debugging)
                    if "subjectProperty" in data:
                        subject_info = data["subjectProperty"]
                        logger.info(f"[AVM] Subject property info: beds={subject_info.get('bedrooms')}, baths={subject_info.get('bathrooms')}, sqft={subject_info.get('squareFootage')}")
                        
                else:
                    logger.warning(f"[AVM] Non-200 response: {resp.status_code}")
                    logger.warning(f"[AVM] Response text: {resp.text[:500]}")
                    
            except Exception as e:
                logger.exception(f"[AVM] Request failed for {s.address}: {e}")

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
            # Enhanced comparable data extraction
            comp_id = str(comp.get("id") or comp.get("propertyId") or comp.get("mlsId") or uuid4())
            comp_address = str(comp.get("address") or comp.get("formattedAddress") or "Unknown Address")
            comp_lat = float(_gf(comp, "lat", "latitude") or 0.0)
            comp_lng = float(_gf(comp, "lng", "longitude") or 0.0)  # Fixed: was incorrectly parsing from 'lapshot_sqft'
            comp_property_type = str(comp.get("propertyType") or "Single Family")
            comp_sqft = float(_gi(comp, "livingArea", "squareFootage", "sqft") or 0)
            comp_lot_sqft = float(_gi(comp, "lotSize", "lot_sqft") or 0) or None
            comp_beds = int(_gi(comp, "beds", "bedrooms") or 0)
            comp_baths = float(_gf(comp, "baths", "bathrooms") or 0)
            comp_year_built = int(_gi(comp, "yearBuilt", "year_built") or 0)
            comp_price = _gp(comp)
            
            price_formatted = f"${comp_price:,.0f}" if comp_price else "$0"
            sqft_formatted = f"{comp_sqft:,.0f}" if comp_sqft else "0"
            logger.info(f"[AVM Comp] {comp_address}: {price_formatted}, {comp_beds}bd/{comp_baths}ba, {sqft_formatted}sf")
            
            comps_list.append(
                Property(
                    id=comp_id,
                    address=comp_address,
                    lat=comp_lat,
                    lng=comp_lng,
                    property_type=comp_property_type,
                    living_sqft=comp_sqft,
                    lot_sqft=comp_lot_sqft,
                    beds=comp_beds,
                    baths=comp_baths,
                    year_built=comp_year_built,
                    condition_rating=None,
                    features=set(),
                    sale_date=None,
                    raw_price=comp_price,
                )
            )
            
        logger.info(f"[AVM] Successfully processed {len(comps_list)} comparables")

    # Fallback to sample data if no RentCast comps
    if not comps_list:
        comps_list = [
            Property(
                id=str(comp["id"]),
                address=comp["address"],
                lat=0.0,
                lng=0.0,
                property_type="SFR",
                living_sqft=comp["sqft"],
                lot_sqft=comp["lot_size"],
                beds=comp["beds"],
                baths=comp["baths"],
                year_built=comp["year_built"],
                condition_rating=None,
                features=set(),
                sale_date=None,
                raw_price=comp["price"],
            )
            for comp in comps_data
        ]

    # Find best comps using scoring algorithm
    _, scored = find_comps(subject_prop, comps_list, {}, default_weights, {}, 5)
    total_price, total_weight = 0.0, 0.0
    for comp, score in scored:
        if comp.raw_price:
            total_price += comp.raw_price * max(score, 0.01)
            total_weight += max(score, 0.01)
    fallback_estimate = round(total_price / total_weight, 0) if total_weight > 0 else 0.0
    
    # Use AVM estimate if available and valid, otherwise use fallback
    if avm_estimate is not None and avm_estimate > 0:
        estimate = float(avm_estimate)
        logger.info(f"[CMA Baseline] Using RentCast AVM estimate: ${estimate:,.0f}")
    else:
        estimate = fallback_estimate
        logger.info(f"[CMA Baseline] Using fallback estimate (no AVM): ${estimate:,.0f}")
        
    logger.info(f"[CMA Baseline] Final estimate: ${estimate:,.0f}, based on {len(scored)} comparables")

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
        # Build enriched subject from subject_prop for accurate snapshot display
        enriched_subject = Subject(
            address=subject_prop.address,
            lat=subject_prop.lat or 0.0,
            lng=subject_prop.lng or 0.0,
            property_type=subject_prop.property_type,
            beds=subject_prop.beds or 0,
            baths=float(subject_prop.baths or 0),
            sqft=int(subject_prop.living_sqft or 0),
            year_built=subject_prop.year_built,
            lot_sqft=int(subject_prop.lot_sqft or 0) if subject_prop.lot_sqft else None,
            condition=s.condition,
        )

        return CMAResponse(
            estimate=estimate,
            subject=enriched_subject,
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


# ---------------- CMA Adjust ----------------
@app.get("/cma/adjust", response_model=CMAResponse)
async def cma_adjust(
    cma_run_id: str,
    condition: Optional[str] = None,
    renovations: Optional[List[str]] = None,
    add_beds: int = 0,
    add_baths: float = 0.0,
    add_sqft: int = 0
) -> CMAResponse:
    """Apply adjustments to an existing CMA run and return adjusted estimate."""
    logger.info(f"[CMA Adjust] Processing adjustments for run: {cma_run_id}")
    
    # Get the original CMA run
    original_run = _load_cma_run(cma_run_id)
    if not original_run:
        raise HTTPException(status_code=404, detail="CMA run not found")
    
    subject_prop = original_run["subject"]
    original_comps = original_run["comps"]
    original_estimate = original_run["estimate"]
    
    logger.info(f"[CMA Adjust] Retrieved run: subject={subject_prop.address}, comps_count={len(original_comps)}, estimate={original_estimate}")
    logger.info(f"[CMA Adjust] Comps type: {type(original_comps)}, first comp: {original_comps[0] if original_comps else 'None'}")
    
    # Create adjustments dict for AI processing
    adjustments = {
        "condition": condition,
        "renovations": renovations or [],
        "add_beds": add_beds,
        "add_baths": add_baths,
        "add_sqft": add_sqft
    }
    
    # Use AI to compute adjusted value
    try:
        logger.info(f"[CMA Adjust] Attempting AI adjustment with: subject={subject_prop.address}, comps_count={len(original_comps)}, adjustments={adjustments}")
        logger.info(f"[CMA Adjust] OpenAI API key present: {bool(os.getenv('OPENAI_API_KEY'))}")
        
        # Convert Property objects to dictionaries safely
        subject_dict = {
            'address': subject_prop.address,
            'lat': subject_prop.lat,
            'lng': subject_prop.lng,
            'property_type': subject_prop.property_type,
            'living_sqft': subject_prop.living_sqft,
            'lot_sqft': subject_prop.lot_sqft,
            'beds': subject_prop.beds,
            'baths': subject_prop.baths,
            'year_built': subject_prop.year_built,
            'condition_rating': subject_prop.condition_rating,
            'raw_price': subject_prop.raw_price
        }
        
        comps_dicts = []
        for comp, _ in original_comps:
            comp_dict = {
                'id': comp.id,
                'address': comp.address,
                'lat': comp.lat,
                'lng': comp.lng,
                'property_type': comp.property_type,
                'living_sqft': comp.living_sqft,
                'lot_sqft': comp.lot_sqft,
                'beds': comp.beds,
                'baths': comp.baths,
                'year_built': comp.year_built,
                'condition_rating': comp.condition_rating,
                'raw_price': comp.raw_price
            }
            comps_dicts.append(comp_dict)
        
        ai_result = await compute_adjusted_cma(
            subject_dict,
            comps_dicts,
            adjustments
        )
        
        logger.info(f"[CMA Adjust] AI result: {ai_result}")
        
        adjusted_estimate = ai_result.get("value")
        logger.info(f"[CMA Adjust] AI returned value: {adjusted_estimate}")
        
        if adjusted_estimate is None or adjusted_estimate <= 0:
            logger.warning(f"[CMA Adjust] AI returned invalid value: {adjusted_estimate}, using fallback calculation")
            # Force fallback calculation by raising an exception
            raise Exception("AI adjustment failed, using fallback")
        else:
            logger.info(f"[CMA Adjust] AI adjustment successful: original={original_estimate}, adjusted={adjusted_estimate}")
            reasoning = ai_result.get("reasoning", "Adjustments applied using AI analysis")
            
            # Create adjusted subject property
            adjusted_subject = Subject(
                address=subject_prop.address,
                lat=subject_prop.lat,
                lng=subject_prop.lng,
                property_type=subject_prop.property_type,
                sqft=int((subject_prop.living_sqft or 0) + add_sqft),
                lot_sqft=int(subject_prop.lot_sqft or 0) if subject_prop.lot_sqft else None,
                beds=(subject_prop.beds or 0) + add_beds,
                baths=float((subject_prop.baths or 0) + add_baths),
                year_built=subject_prop.year_built,
                condition=condition or subject_prop.condition_rating
            )
            
            # Generate new narrative for adjusted property
            adjusted_narrative = generate_ai_narrative(adjusted_subject, adjusted_estimate, original_comps)
            
            # Convert stored tuples back to Comp objects for the response
            comps_for_response = []
            for comp, score in original_comps:
                try:
                    distance = None
                    if subject_prop.lat and subject_prop.lng and comp.lat and comp.lng:
                        distance = calculate_distance(subject_prop.lat, subject_prop.lng, comp.lat, comp.lng)
                    
                    comps_for_response.append(Comp(
                        id=str(comp.id),
                        address=getattr(comp, 'address', 'Unknown Address'),
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
                    logger.error(f"[CMA Adjust] Error processing comp {comp.id}: {e}")
                    continue
            
                    # Persist adjusted data for later PDF generation
        try:
            cma_runs_storage[cma_run_id]["adjusted"] = {
                "estimate": adjusted_estimate,
                "subject": adjusted_subject,
                "comps": original_comps,
            }
        except Exception:
            pass

        return CMAResponse(
            estimate=adjusted_estimate,
            subject=adjusted_subject,
            comps=comps_for_response,
            explanation=adjusted_narrative,
            cma_run_id=cma_run_id,  # Keep same run ID to link with original
        )
        
    except Exception as e:
        logger.error(f"[CMA Adjust] AI adjustment failed: {e}")
        # Fallback: simple percentage adjustments
        adjustment_multiplier = 1.0
        
        # Condition adjustments (multiplicative)
        if condition == "Poor":
            adjustment_multiplier *= 0.85
        elif condition == "Fair":
            adjustment_multiplier *= 0.92
        elif condition == "Excellent":
            adjustment_multiplier *= 1.15
            
        # Renovation adjustments (multiplicative)
        renovation_bonus = 1.0 + (len(renovations or []) * 0.05)
        adjustment_multiplier *= renovation_bonus
        
        # Size adjustments (multiplicative)
        if add_beds > 0:
            adjustment_multiplier *= (1.0 + add_beds * 0.08)
        if add_baths > 0:
            adjustment_multiplier *= (1.0 + add_baths * 0.06)
        if add_sqft > 0:
            adjustment_multiplier *= (1.0 + (add_sqft / 1000) * 0.1)
            
        adjusted_estimate = round(original_estimate * adjustment_multiplier, 0)
        
        logger.info(f"[CMA Adjust] Fallback calculation: original={original_estimate}, multiplier={adjustment_multiplier}, adjusted={adjusted_estimate}")
        logger.info(f"[CMA Adjust] Adjustment breakdown: condition={condition}, renovations={renovations}, add_beds={add_beds}, add_baths={add_baths}, add_sqft={add_sqft}")
        logger.info(f"[CMA Adjust] Using fallback calculation - AI failed")
        
        # Create adjusted subject property
        adjusted_subject = Subject(
            address=subject_prop.address,
            lat=subject_prop.lat,
            lng=subject_prop.lng,
            property_type=subject_prop.property_type,
            sqft=int((subject_prop.living_sqft or 0) + add_sqft),
            lot_sqft=int(subject_prop.lot_sqft or 0) if subject_prop.lot_sqft else None,
            beds=(subject_prop.beds or 0) + add_beds,
            baths=float((subject_prop.baths or 0) + add_baths),
            year_built=subject_prop.year_built,
            condition=condition or subject_prop.condition_rating
        )
        
        # Convert stored tuples back to Comp objects for the response
        comps_for_response = []
        for comp, score in original_comps:
            try:
                distance = None
                if subject_prop.lat and subject_prop.lng and comp.lat and comp.lng:
                    distance = calculate_distance(subject_prop.lat, subject_prop.lng, comp.lat, comp.lng)
                
                comps_for_response.append(Comp(
                    id=str(comp.id),
                    address=getattr(comp, 'address', 'Unknown Address'),
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
                logger.error(f"[CMA Adjust] Error processing comp {comp.id}: {e}")
                continue
        
        # Persist adjusted data for later PDF generation (fallback path)
        try:
            cma_runs_storage[cma_run_id]["adjusted"] = {
                "estimate": adjusted_estimate,
                "subject": adjusted_subject,
                "comps": original_comps,
            }
        except Exception:
            pass

        return CMAResponse(
            estimate=adjusted_estimate,
            subject=adjusted_subject,
            comps=comps_for_response,
            explanation=f"Applied adjustments using fallback calculation. New estimate: ${adjusted_estimate:,}",
            cma_run_id=cma_run_id,
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


# ---------------- Enhanced Rent estimate endpoint ----------------
@app.get("/rent/estimate")
async def rent_estimate(address: str, bedrooms: Optional[int] = None, bathrooms: Optional[float] = None, 
                       squareFootage: Optional[int] = None, propertyType: Optional[str] = None) -> Dict[str, Any]:
    """Fetch estimated monthly rent for an address from RentCast with enhanced accuracy parameters."""
    api_key = os.getenv("RENTCAST_API_KEY")
    if not api_key:
        return {"monthly_rent": None, "error": "RENTCAST_API_KEY not set"}

    try:
        logger.info(f"[Rent Estimate] Making enhanced rent request for: {address}")
        
        # Build enhanced parameters using configuration class
        rent_params = RentCastConfig.get_avm_params(
            address=address,
            property_type=propertyType,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            square_footage=squareFootage
        )
            
        logger.info(f"[Rent Estimate] Request parameters: {rent_params}")
        
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.rentcast.io/v1/avm/rent/long-term",
                params=rent_params,
                headers={"X-Api-Key": api_key},
            )
            
        logger.info(f"[Rent Estimate] Response status: {resp.status_code}")
        
        if resp.status_code != 200:
            logger.warning(f"[Rent Estimate] Non-200 response: {resp.status_code}")
            logger.warning(f"[Rent Estimate] Response text: {resp.text[:500]}")
            return {"monthly_rent": None, "error": f"status {resp.status_code}", "details": resp.text[:200]}

        data = resp.json()
        logger.info(f"[Rent Estimate] Raw response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Response may be an object or a list
        candidate = None
        if isinstance(data, list) and data:
            candidate = data[0]
        elif isinstance(data, dict):
            candidate = data
        else:
            return {"monthly_rent": None}

        # Enhanced rent value extraction - RentCast uses 'rent' as primary field
        rent_keys = [
            "rent", "monthlyRent", "rentEstimate", "estimatedRent", "amount", "value", "price"
        ]
        
        for k in rent_keys:
            v = candidate.get(k)
            if v is not None and v > 0:
                try:
                    rent_value = float(v)
                    logger.info(f"[Rent Estimate] Found rent: ${rent_value:,.0f}/month from field '{k}'")
                    return {"monthly_rent": rent_value}
                except Exception:
                    try:
                        rent_value = float(str(v).replace(",", ""))
                        logger.info(f"[Rent Estimate] Found rent (parsed): ${rent_value:,.0f}/month from field '{k}'")
                        return {"monthly_rent": rent_value}
                    except Exception:
                        continue

        logger.warning(f"[Rent Estimate] No valid rent found in response")
        return {"monthly_rent": None}
        
    except Exception as e:
        logger.exception(f"[Rent Estimate] Exception for {address}: {e}")
        return {"monthly_rent": None, "error": "exception"}

# ---------------- Enhanced Debug endpoint for testing AVM improvements ----------------
@app.get("/debug/avm")
async def debug_avm(address: str, bedrooms: Optional[int] = None, bathrooms: Optional[float] = None, 
                   squareFootage: Optional[int] = None, propertyType: Optional[str] = None) -> Dict[str, Any]:
    """Debug endpoint to test enhanced AVM accuracy with detailed logging."""
    api_key = os.getenv("RENTCAST_API_KEY")
    if not api_key:
        return {"error": "RENTCAST_API_KEY not set"}
    
    try:
        # Build enhanced parameters using configuration class
        avm_params = RentCastConfig.get_avm_params(
            address=address,
            property_type=propertyType,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            square_footage=squareFootage
        )
        
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.rentcast.io/v1/avm/value",
                params=avm_params,
                headers={"X-Api-Key": api_key},
            )
        
        return {
            "address": address,
            "request_params": avm_params,
            "status_code": resp.status_code,
            "response": resp.json() if resp.status_code == 200 else resp.text[:1000]
        }
        
    except Exception as e:
        return {"address": address, "error": str(e)}

# ---------------- PDF Generation ----------------
@app.get("/pdfx")
async def generate_cma_pdf(cma_run_id: str, adjusted: Optional[bool] = False):
    """Generate a PDF report for a CMA run.

    If adjusted=true and adjusted data exists, use adjusted; else baseline.
    """
    logger.info(f"[PDF] Generating PDF for CMA run: {cma_run_id}, adjusted={adjusted}")
    
    # Get the CMA run data
    base_run = _load_cma_run(cma_run_id)
    if not base_run:
        raise HTTPException(status_code=404, detail="CMA run not found")

    run_for_pdf = base_run
    if adjusted and base_run.get("adjusted"):
        adj = base_run["adjusted"]
        # Normalize shape back to what create_cma_pdf expects
        run_for_pdf = {
            "subject": adj["subject"],
            "estimate": adj["estimate"],
            "comps": adj["comps"],
        }
    
    try:
        # Generate PDF using the existing pdf_utils module
        pdf_bytes = create_cma_pdf(cma_run_id, run_for_pdf)
        
        logger.info(f"[PDF] Successfully generated PDF, size: {len(pdf_bytes)} bytes")
        
        # Return PDF as response with proper headers
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=cma_report_{cma_run_id}{'_adjusted' if adjusted else ''}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"[PDF] Failed to generate PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


# ---------------- Saved searches (placeholder) ----------------
@app.get("/searches/list")
async def list_saved_searches(user_id: str) -> Dict[str, Any]:
    # TODO: replace with real storage (e.g., Supabase table)
    # For now, return an empty list to avoid 404s on dashboard load
    return {"results": []}
