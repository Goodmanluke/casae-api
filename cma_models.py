from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Subject(BaseModel):
    """Pydantic model representing the subject property for CMA."""
    address: str
    lat: float
    lng: float
    beds: int
    baths: float
    sqft: int
    year_built: Optional[int] = None
    lot_sqft: Optional[int] = None
    condition: Optional[str] = None
    waterfront: Optional[bool] = False

class CMAInput(BaseModel):
    """Request body for creating a baseline CMA run."""
    subject: Subject
    rules: Dict[str, Any] = {}

class AdjustmentInput(BaseModel):
    """Request body for adjusting a previous CMA run."""
    cma_run_id: str
    condition: Optional[str] = None
    renovations: List[str] = []
    add_beds: int = 0
    add_baths: float = 0.0
    add_sqft: int = 0

class Comp(BaseModel):
    """Representation of a comparable property returned in a CMA response."""
    id: str
    address: str
    raw_price: float
    living_sqft: int
    beds: int
    baths: float
    year_built: Optional[int] = None
    lot_sqft: Optional[int] = None
    distance_mi: Optional[float] = None
    similarity: float

class CMAResponse(BaseModel):
    """Response schema for both baseline and adjusted CMA runs."""
    estimate: float
    comps: List[Comp]
    explanation: str
    cma_run_id: str
