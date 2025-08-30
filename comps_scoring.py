"""
Implementation of a simple comparable‑property selection and scoring algorithm.

This module defines a `Property` data class and helper functions for filtering
and scoring comparable properties ("comps") relative to a subject property.

The algorithm follows guidelines from the sales comparison approach to real
estate valuation. It filters potential comparables by basic criteria
(property type, sale window, radius, and size/bed/bath tolerances) and then
computes a similarity score based on several normalized feature differences.

Key factors considered:

* Location: distance between subject and comp (in miles).
* Size: living area square footage difference, normalized by subject.
* Bedrooms and bathrooms: differences in equivalent bed/bath counts.
* Age & condition: year built difference and condition rating difference.
* Features & amenities: Jaccard similarity of categorical feature sets.
* Time adjustment: adjusts sale price based on market index values to
  account for market conditions at the time of sale.

The weights for each factor are configurable; they should sum to one or
approximately one. See `default_weights` for an example.

Note: This implementation does not fetch live market index data. Instead,
it accepts a prepopulated dictionary mapping (geo, month) to an index value.
If no index is provided for a given comp, a neutral adjustment factor of 1.0
is used.

Usage:

    from comps_scoring import Property, find_comps, default_weights
    subject = Property(...)
    comps = [Property(...), ...]
    filtered, scored = find_comps(subject, comps, filters, default_weights)
    for comp, score in scored:
        print(comp.id, score)

This file can be integrated into a larger application that persists
properties in a database and exposes an API for comp searches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import math

@dataclass
class Property:
    """Represents a real estate property and its salient attributes."""

    id: str
    address: str = ""  # Add address field
    lat: float = 0.0
    lng: float = 0.0
    property_type: str = "SFR"
    living_sqft: float = 0.0
    lot_sqft: Optional[float] = None
    beds: int = 0
    baths: int = 0
    half_baths: int = 0
    year_built: Optional[int] = None
    condition_rating: Optional[float] = None  # 1 (poor) to 5 (excellent)
    features: Set[str] = field(default_factory=set)
    sale_date: Optional[date] = None
    raw_price: Optional[float] = None
    market_index_geo: Optional[str] = None  # geographic key for market index lookup

    def bed_equiv(self) -> float:
        """Compute equivalent bedroom count (half baths count as 0.5)."""
        return self.beds + 0.5 * self.half_baths

    def bath_equiv(self) -> float:
        """Compute equivalent bathroom count (half baths count as 0.5)."""
        return self.baths + 0.5 * self.half_baths


def haversine_distance_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return the approximate great-circle distance in miles between two points.

    Uses the Haversine formula on a spherical Earth approximation. This function
    is sufficient for small distances (comp radius filtering) and is widely
    used in geospatial calculations.
    """
    # Radius of Earth in miles
    R = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute the Jaccard similarity between two sets of features.

    The Jaccard similarity is defined as the size of the intersection
    divided by the size of the union. If both sets are empty, this
    function returns 1.0.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 1.0


def time_adjustment_factor(
    sale_date: Optional[date],
    market_index_geo: Optional[str],
    market_index: Dict[Tuple[str, str], float],
    current_date: Optional[date] = None,
) -> float:
    """Compute a factor to adjust sale price for market conditions.

    Args:
        sale_date: date when the comp sold.
        market_index_geo: a geographic key (e.g., zip code or county) used
            to index into `market_index`.
        market_index: mapping of (geo, month_key) -> index value.
        current_date: date to which the sale price should be adjusted. If
            omitted, defaults to today.

    Returns:
        A float factor by which to multiply the raw sale price. If no
        appropriate index values are available, returns 1.0.
    """
    if sale_date is None or market_index_geo is None:
        return 1.0
    if current_date is None:
        current_date = date.today()
    sale_key = f"{sale_date.year:04d}-{sale_date.month:02d}"
    current_key = f"{current_date.year:04d}-{current_date.month:02d}"
    try:
        sale_idx = market_index[(market_index_geo, sale_key)]
        current_idx = market_index[(market_index_geo, current_key)]
        if sale_idx > 0:
            return current_idx / sale_idx
    except KeyError:
        # Missing index for this geo/month; fall back to neutral factor
        pass
    return 1.0


def filter_comps(
    subject: Property,
    comps: Iterable[Property],
    filters: Dict[str, float],
) -> List[Property]:
    """Filter a list of candidate comps based on hard criteria.

    Supported filter keys:
        - 'radius_miles': maximum distance from subject (float, miles)
        - 'sale_date_min': minimum sale date (inclusive, date)
        - 'sale_date_max': maximum sale date (inclusive, date)
        - 'living_sqft_pct': tolerance as a fraction of subject's living_sqft
        - 'beds_min': minimum number of beds (inclusive)
        - 'baths_min': minimum number of baths (inclusive)
        - 'property_type': required property type (string)

    Returns:
        A list of comps that satisfy all filters.
    """
    filtered: List[Property] = []
    radius = filters.get("radius_miles")
    sale_date_min = filters.get("sale_date_min")  # type: ignore
    sale_date_max = filters.get("sale_date_max")  # type: ignore
    sqft_tol = filters.get("living_sqft_pct", 0.2)
    beds_min = filters.get("beds_min")
    baths_min = filters.get("baths_min")
    required_type = filters.get("property_type")

    subject_sqft = subject.living_sqft

    for comp in comps:
        # Skip same property
        if comp.id == subject.id:
            continue
        # Property type filter
        if required_type and comp.property_type != required_type:
            continue
        # Distance filter
        if radius is not None:
            dist = haversine_distance_miles(subject.lat, subject.lng, comp.lat, comp.lng)
            if dist > radius:
                continue
        # Sale date filter
        if sale_date_min and comp.sale_date and comp.sale_date < sale_date_min:
            continue
        if sale_date_max and comp.sale_date and comp.sale_date > sale_date_max:
            continue
        # Size tolerance filter
        if comp.living_sqft:
            lower = subject_sqft * (1 - sqft_tol)
            upper = subject_sqft * (1 + sqft_tol)
            if comp.living_sqft < lower or comp.living_sqft > upper:
                continue
        # Beds/Baths minimum filter
        if beds_min is not None and comp.beds < beds_min:
            continue
        if baths_min is not None and comp.baths < baths_min:
            continue
        filtered.append(comp)
    return filtered


def compute_similarity(
    subject: Property,
    comp: Property,
    weights: Dict[str, float],
    market_index: Dict[Tuple[str, str], float],
    max_distance: float = 1.0,
) -> float:
    """Compute a similarity score between subject and a single comp.

    The score is a weighted sum of normalized differences; lower differences yield
    higher similarity. All weight keys should sum to 1.0 (approximately).

    Args:
        subject: the property for which comps are sought.
        comp: a candidate comparable property.
        weights: a mapping from factor name to weight. Supported keys are:
            'distance_miles', 'living_sqft_diff_pct', 'bed_equiv_diff',
            'bath_equiv_diff', 'year_built_diff', 'condition_diff',
            'features_jaccard', and 'time_adjustment'.
        market_index: preloaded market index data for time adjustments.
        max_distance: normalization constant for distance (miles). If comps
            are filtered within `radius_miles`, this should match that radius.

    Returns:
        A similarity score between 0 and 1 (higher is more similar).
    """
    # Compute distance (normalized by max_distance)
    dist = haversine_distance_miles(subject.lat, subject.lng, comp.lat, comp.lng)
    distance_score = 1.0 - min(dist / max_distance, 1.0)

    # Living area difference percentage
    sqft_diff_pct = abs(comp.living_sqft - subject.living_sqft) / subject.living_sqft if subject.living_sqft else 0.0
    size_score = 1.0 - min(sqft_diff_pct, 1.0)

    # Bed/bath equivalent differences
    bed_diff = abs(comp.bed_equiv() - subject.bed_equiv())
    bath_diff = abs(comp.bath_equiv() - subject.bath_equiv())
    # Normalize by plausible maximum difference (e.g., 5 bedrooms difference -> 5)
    bed_score = 1.0 - min(bed_diff / 5.0, 1.0)
    bath_score = 1.0 - min(bath_diff / 5.0, 1.0)

    # Year built difference (normalize by 100 years)
    year_score = 1.0
    if subject.year_built and comp.year_built:
        year_diff = abs(subject.year_built - comp.year_built)
        year_score = 1.0 - min(year_diff / 100.0, 1.0)

    # Condition rating difference (scale difference of 4 (1 to 5) to [0,1])
    condition_score = 1.0
    if subject.condition_rating is not None and comp.condition_rating is not None:
        cond_diff = abs(subject.condition_rating - comp.condition_rating)
        condition_score = 1.0 - min(cond_diff / 4.0, 1.0)

    # Features Jaccard similarity
    feature_score = jaccard_similarity(subject.features, comp.features)

    # Time adjustment factor: 1 for perfect match; else less if older comp
    adj_factor = time_adjustment_factor(comp.sale_date, comp.market_index_geo, market_index)
    time_score = adj_factor  # Already normalized around 1.0
    # Clip to [0, 1.5] to avoid extreme influences
    time_score = max(0.0, min(time_score, 1.5)) / 1.5  # normalize to [0,1]

    # Weighted sum of scores
    components = {
        'distance_miles': distance_score,
        'living_sqft_diff_pct': size_score,
        'bed_equiv_diff': bed_score,
        'bath_equiv_diff': bath_score,
        'year_built_diff': year_score,
        'condition_diff': condition_score,
        'features_jaccard': feature_score,
        'time_adjustment': time_score,
    }
    # Compute final similarity
    total = 0.0
    for key, weight in weights.items():
        total += weight * components.get(key, 0.0)
    return total


def find_comps(
    subject: Property,
    comps: Sequence[Property],
    filters: Dict[str, float],
    weights: Dict[str, float],
    market_index: Dict[Tuple[str, str], float],
    return_limit: int = 10,
) -> Tuple[List[Property], List[Tuple[Property, float]]]:
    """Filter and rank comparable properties relative to a subject.

    Args:
        subject: the property for which comps are sought.
        comps: candidate comparable properties.
        filters: hard filter criteria; see `filter_comps` for supported keys.
        weights: weights for similarity scoring; should sum to approximately 1.0.
        market_index: market index values for time adjustments.
        return_limit: number of top comps to return.

    Returns:
        A tuple `(filtered, scored)`. `filtered` is the list of comps that
        satisfied the hard filters. `scored` is a list of `(comp, score)` pairs
        sorted by descending score. Only the top `return_limit` comps are
        included in the scored list.
    """
    # Apply hard filters
    filtered = filter_comps(subject, comps, filters)

    # Compute similarity scores
    scored: List[Tuple[Property, float]] = []
    # Use radius in filters as normalization for distance
    max_dist = filters.get("radius_miles", 1.0) or 1.0
    for comp in filtered:
        score = compute_similarity(subject, comp, weights, market_index, max_dist)
        scored.append((comp, score))
    # Sort by descending similarity score
    scored.sort(key=lambda x: x[1], reverse=True)
    return filtered, scored[:return_limit]


# Example default weights; adjust as needed to emphasize different factors.
default_weights: Dict[str, float] = {
    'distance_miles': 0.22,
    'living_sqft_diff_pct': 0.20,
    'bed_equiv_diff': 0.12,
    'bath_equiv_diff': 0.10,
    'year_built_diff': 0.06,
    'condition_diff': 0.12,
    'features_jaccard': 0.10,
    'time_adjustment': 0.08,
}
