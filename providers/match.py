"""Match a lead to providers: filter by category, rank by distance + reputation.

score = bayesian_rating - penalty_per_mile(urgency) * miles

Bayesian rating pulls ratings with few reviews toward a prior, so a 5.0 with
3 reviews doesn't beat a 4.8 with 400. The distance penalty depends on urgency
(weights in agent/config.py): for an emergency, a close crew beats a slightly
better-rated one across town. All constants are tunable.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DISTANCE_PENALTY_PER_MILE, DISTANCE_PENALTY_PER_MILE
from providers.geo import miles_between, zip_to_latlng
from providers.models import Provider

PROVIDERS_PATH = Path(__file__).parent / "providers.json"

PRIOR_RATING = 4.0  # what we assume about a provider with no reviews
PRIOR_WEIGHT = 20  # reviews needed before a provider's own rating dominates
MAX_MATCH_DISTANCE_MILES = 25  # beyond this, don't recommend


# Service area: within this distance of one of the four cities we cover.
SERVICE_AREA_ANCHORS = {"Davis": "95616", "Woodland": "95695", "West Sacramento": "95691", "Sacramento": "95814"}
SERVICE_AREA_RADIUS_MILES = 25


def zip_service_status(zip_code: str) -> str:
    """'in_area', 'out_of_area' (a real zip outside the region) or 'unknown' (not a real zip)."""
    loc = zip_to_latlng(zip_code)
    if loc is None:
        return "unknown"
    nearest = min(miles_between(loc, zip_to_latlng(z)) for z in SERVICE_AREA_ANCHORS.values())
    return "in_area" if nearest <= SERVICE_AREA_RADIUS_MILES else "out_of_area"


def penalty_per_mile(urgency: str | None) -> float:
    return DISTANCE_PENALTY_PER_MILE.get(urgency, DEFAULT_DISTANCE_PENALTY_PER_MILE)


def bayesian_rating(rating: float | None, review_count: int) -> float:
    if not rating or review_count <= 0:
        return PRIOR_RATING
    return (review_count * rating + PRIOR_WEIGHT * PRIOR_RATING) / (review_count + PRIOR_WEIGHT)


@lru_cache(maxsize=1)
def load_providers(path: Path = PROVIDERS_PATH) -> tuple[Provider, ...]:
    data = json.loads(path.read_text())
    return tuple(Provider(**p) for p in data)


def match_providers(
    category: str,
    zip_code: str,
    providers: list[Provider] | tuple[Provider, ...] | None = None,
    n: int = 3,
    urgency: str | None = None,
) -> list[dict[str, Any]]:
    """Top-n providers for a category near a zip, as dicts (ready for LeadState).

    Ranked by score, with one guarantee: the nearest provider that passed every
    filter is always included (it replaces the last slot if the score left it
    out), so a highly rated company across town never crowds out the one down
    the street.

    Returns [] if the zip is unknown or nobody serves that category nearby,
    which makes the lead non-dispatchable.
    """
    if providers is None:
        providers = load_providers()
    user_loc = zip_to_latlng(zip_code)
    if user_loc is None:
        return []

    penalty = penalty_per_mile(urgency)
    scored = []
    for p in providers:
        if p.category != category or p.lat is None or p.lng is None or not p.phone:
            continue
        miles = miles_between(user_loc, (p.lat, p.lng))
        if miles > MAX_MATCH_DISTANCE_MILES:
            continue
        score = bayesian_rating(p.rating, p.review_count) - penalty * miles
        scored.append((score, miles, p))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:n]
    if scored and n > 0:
        nearest = min(scored, key=lambda t: t[1])
        if nearest not in top:
            top[-1] = nearest
    return [{**p.model_dump(mode="json"), "distance_miles": round(miles, 1)} for _, miles, p in top]
