"""Provider record stored in providers.json."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

LocationSource = Literal["exact", "zip_centroid", "city_centroid"]


class Provider(BaseModel):
    place_id: str
    name: str
    phone: str | None = None
    address: str | None = None
    city: str | None = None
    zip: str | None = None
    # Exact Places location when available; otherwise the zip (or city)
    # centroid. location_source says which, so reviews can tell them apart.
    lat: float | None = None
    lng: float | None = None
    location_source: LocationSource | None = None
    business_status: str | None = None  # Places businessStatus, e.g. OPERATIONAL
    primary_type: str | None = None  # Places primaryType, e.g. plumber
    rating: float | None = None
    review_count: int = 0
    website: str | None = None
    category: str
    source_query: str
    fetched_at: datetime
