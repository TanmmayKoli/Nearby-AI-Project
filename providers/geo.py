"""Zip/city -> approximate coordinates, and distances. Offline (zipcodes package)."""

import math
from functools import lru_cache

import zipcodes


@lru_cache(maxsize=None)
def zip_to_latlng(zip_code: str) -> tuple[float, float] | None:
    """Centroid of a US zip code, or None if unknown."""
    try:
        matches = zipcodes.matching(zip_code)
    except (ValueError, TypeError):
        return None
    if not matches:
        return None
    return float(matches[0]["lat"]), float(matches[0]["long"])


@lru_cache(maxsize=None)
def city_to_latlng(city: str, state: str = "CA") -> tuple[float, float] | None:
    """Average of a city's standard zip centroids. Fallback when a listing has no zip."""
    matches = zipcodes.filter_by(city=city, state=state, zip_code_type="STANDARD")
    if not matches:
        return None
    lat = sum(float(m["lat"]) for m in matches) / len(matches)
    lng = sum(float(m["long"]) for m in matches) / len(matches)
    return lat, lng


def miles_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle (haversine) distance in miles."""
    lat1, lng1, lat2, lng2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))
