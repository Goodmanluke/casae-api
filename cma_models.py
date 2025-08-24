from typing import Optional, List
from pydantic import BaseModel


class Subject(BaseModel):
    address: str
    lat: float = 0.0
    lng: float = 0.0
    beds: Optional[int] = None
    baths: Optional[float] = None
    sqft: Optional[int] = None
    year_built: Optional[int] = None
    lot_sqft: Optional[int] = None


class CMAInput(BaseModel):
    subject: Subject
    rules: dict = {}


class AdjustmentInput(BaseModel):
    """Request body for adjusting a previous CMA run."""
    cma_run_id: str
    condition: Optional[str] = None
    renovations: List[str] = []
    add_beds: int = 0
    add_baths: float = 0.0
    add_sqft: int = 0
    # ⬆️ dock_length removed


class Comp(BaseModel):
    id: str
    address: str
    raw_price: float
    living_sqft: int
    beds: int
    baths: float
    year_built: Optional[int]
    lot_sqft: Optional[int]
    distance_mi: Optional[float]
    similarity: float


class CMAResponse(BaseModel):
    estimate: float
                    subject: Subject

    comps: List[Comp]
    explanation: str
    cma_run_id: str
