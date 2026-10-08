"""One-time provider pull from Google Places API (New) Text Search.

Usage:
    python -m providers.fetch --dry-run   # print queries, request count, field mask. No API calls.
    python -m providers.fetch             # call the API, write providers_raw.json + providers.json
    python -m providers.fetch --reselect  # rebuild providers.json from providers_raw.json (no API calls)
    python -m providers.fetch --review    # print a table of providers.json (combine with --reselect)

Manual exclusions live in providers/overrides.json (applied after the type allowlist).

Two files:
- providers_raw.json: every result we paid for, deduped. Lets us re-tune the
  selection without calling the API again.
- providers.json: the curated set the agent actually uses (see `select`).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent.categories import CATEGORIES
from agent.config import GOOGLE_PLACES_API_KEY
from providers.geo import city_to_latlng, miles_between, zip_to_latlng
from providers.match import PROVIDERS_PATH, bayesian_rating
from providers.models import Provider

RAW_PATH = Path(__file__).parent / "providers_raw.json"
OVERRIDES_PATH = Path(__file__).parent / "overrides.json"
SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

CITIES = ["Davis", "Woodland", "West Sacramento", "Sacramento"]
ANCHOR_ZIPS = {"Davis": "95616", "Woodland": "95695", "West Sacramento": "95691", "Sacramento": "95814"}

# Fields listed in CLAUDE.md. City and zip are parsed from formattedAddress.
# Billing is per request at the highest tier of any requested field. Phone,
# rating, review count and website are Enterprise; everything else here is
# Pro or lower, so businessStatus/location/primaryType add no cost.
FIELD_MASK = ",".join(
    [
        "places.id",  # place_id (Essentials)
        "places.displayName",  # name (Pro)
        "places.formattedAddress",  # address, city, zip (Pro)
        "places.businessStatus",  # drop non-operational (Pro)
        "places.location",  # exact lat/lng for distance (Pro)
        "places.primaryType",  # drop retail stores (Pro)
        "places.nationalPhoneNumber",  # phone (Enterprise)
        "places.rating",  # (Enterprise)
        "places.userRatingCount",  # review count (Enterprise)
        "places.websiteUri",  # website (Enterprise)
    ]
)
PAGE_SIZE = 20  # max per request; we don't paginate

# Selection (providers_raw -> providers)
SERVICE_RADIUS_MILES = 15  # must be within this of at least one anchor city
MAX_PER_ANCHOR = 3  # cap per category per nearest anchor city. Not a quota: fewer is fine.
MIN_REVIEW_COUNT = 10  # thin listings (few reviews) are often placeholder/lead-gen
THIN_CELL_WARNING = 2  # --review flags category x city cells with fewer than this

# Which Places primaryType values count as doing the work, per category.
# Trades with a dedicated type allow it (+ general_contractor for multi-trade
# shops). Where Places has no specific type, generic types are allowed
# EXPLICITLY (marked GENERIC); overrides.json + MIN_REVIEW_COUNT catch bad ones.
CATEGORY_TYPE_ALLOWLIST: dict[str, frozenset[str]] = {
    "plumbing": frozenset({"plumber", "general_contractor"}),
    "electrical": frozenset({"electrician", "general_contractor"}),
    "roofing": frozenset({"roofing_contractor"}),
    # GENERIC: no HVAC-specific type; plumber-typed listings excluded (mostly plain plumbers).
    "hvac": frozenset({"general_contractor", "service"}),
    # GENERIC: no restoration-specific type.
    "water_damage": frozenset({"general_contractor", "service"}),
    # GENERIC: everything is typed "service".
    "pest_control": frozenset({"service"}),
    # GENERIC: everything is typed "service".
    "appliance_repair": frozenset({"service"}),
    # GENERIC: handymen are typed general_contractor or service.
    "handyman": frozenset({"general_contractor", "service"}),
}

# Stores that show up in service searches but don't do the work. Redundant with
# the allowlist today; kept as a safety net if an allowlist is widened.
# (Deliberately not including appliance-store types: some appliance repair shops
# are listed that way.)
EXCLUDED_PRIMARY_TYPES: frozenset[str] = frozenset(
    {
        "hardware_store",
        "home_goods_store",
        "home_improvement_store",
        "electronics_store",
        "department_store",
    }
)


def build_queries() -> list[tuple[str, str, str]]:
    """(category, search_term, text_query) for every category term x city."""
    return [
        (spec.key, term, f"{term} in {city}, CA")
        for spec in CATEGORIES.values()
        for term in spec.search_queries
        for city in CITIES
    ]


def search_text(client: httpx.Client, text_query: str, api_key: str) -> list[dict]:
    """One Text Search request. Returns the raw `places` list."""
    resp = client.post(
        SEARCH_URL,
        headers={"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": FIELD_MASK},
        json={"textQuery": text_query, "pageSize": PAGE_SIZE},
    )
    resp.raise_for_status()
    return resp.json().get("places", [])


_ADDRESS_RE = re.compile(r"(?:^|,)\s*([^,]+),\s*CA(?:\s+(\d{5}))?(?:,|$)")


def parse_city_zip(address: str | None) -> tuple[str | None, str | None]:
    """'123 Main St, Davis, CA 95616, USA' -> ('Davis', '95616')."""
    if not address:
        return None, None
    match = _ADDRESS_RE.search(address)
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2)


def resolve_location(
    location: dict | None, zip_code: str | None, city: str | None
) -> tuple[tuple[float, float] | None, str | None]:
    """Exact Places location first, then zip centroid, then city centroid."""
    if location and "latitude" in location and "longitude" in location:
        return (location["latitude"], location["longitude"]), "exact"
    if zip_code and (loc := zip_to_latlng(zip_code)):
        return loc, "zip_centroid"
    if city and (loc := city_to_latlng(city)):
        return loc, "city_centroid"
    return None, None


def to_provider(place: dict, category: str, source_query: str, fetched_at: datetime) -> Provider:
    address = place.get("formattedAddress")
    city, zip_code = parse_city_zip(address)
    loc, source = resolve_location(place.get("location"), zip_code, city)
    return Provider(
        place_id=place["id"],
        name=place.get("displayName", {}).get("text", ""),
        phone=place.get("nationalPhoneNumber"),
        address=address,
        city=city,
        zip=zip_code,
        lat=loc[0] if loc else None,
        lng=loc[1] if loc else None,
        location_source=source,
        business_status=place.get("businessStatus"),
        primary_type=place.get("primaryType"),
        rating=place.get("rating"),
        review_count=place.get("userRatingCount", 0),
        website=place.get("websiteUri"),
        category=category,
        source_query=source_query,
        fetched_at=fetched_at,
    )


def fetch_all(api_key: str) -> list[Provider]:
    """Run every query. Dedupe on (place_id, category): a business can legitimately
    appear in two categories (e.g. a plumber that also does water damage)."""
    fetched_at = datetime.now(timezone.utc)
    seen: dict[tuple[str, str], Provider] = {}
    queries = build_queries()
    with httpx.Client(timeout=20) as client:
        for i, (category, _, text_query) in enumerate(queries, 1):
            places = search_text(client, text_query, api_key)
            print(f"[{i}/{len(queries)}] {text_query}: {len(places)} results")
            for place in places:
                key = (place["id"], category)
                if key not in seen:
                    seen[key] = to_provider(place, category, text_query, fetched_at)
    return list(seen.values())


def nearest_anchor(p: Provider) -> tuple[str, float] | None:
    if p.lat is None or p.lng is None:
        return None
    distances = [(city, miles_between((p.lat, p.lng), zip_to_latlng(z))) for city, z in ANCHOR_ZIPS.items()]
    return min(distances, key=lambda d: d[1])


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, list[dict]]:
    """overrides.json -> {place_id: [entries]}. Missing file means no overrides."""
    if not path.exists():
        return {}
    by_place: dict[str, list[dict]] = defaultdict(list)
    for entry in json.loads(path.read_text()):
        if entry.get("action") != "exclude":
            raise ValueError(f"unsupported override action: {entry.get('action')}")
        by_place[entry["place_id"]].append(entry)
    return by_place


def override_reason(p: Provider, overrides: dict[str, list[dict]]) -> str | None:
    for entry in overrides.get(p.place_id, []):
        cats = entry["categories"]
        if cats == "all" or p.category in cats:
            return f"override: {entry['reason']}"
    return None


def exclusion_reason(p: Provider, overrides: dict[str, list[dict]]) -> str | None:
    """Why a provider shouldn't be used in its category, or None if it passes.

    Order matters only for which reason gets logged: allowlist, then overrides.
    """
    if p.business_status != "OPERATIONAL":
        return f"business_status={p.business_status}"
    if p.primary_type in EXCLUDED_PRIMARY_TYPES:
        return f"retail primary_type={p.primary_type}"
    if p.primary_type not in CATEGORY_TYPE_ALLOWLIST.get(p.category, frozenset()):
        return f"type {p.primary_type} not allowed for {p.category}"
    if reason := override_reason(p, overrides):
        return reason
    if not p.phone:
        return "no phone"
    if p.review_count < MIN_REVIEW_COUNT:
        return f"thin listing ({p.review_count} reviews < {MIN_REVIEW_COUNT})"
    anchor = nearest_anchor(p)
    if anchor is None:
        return "no location"
    if anchor[1] > SERVICE_RADIUS_MILES:
        return f"out of area ({anchor[1]:.0f} mi from {anchor[0]})"
    return None


def normalize_phone_digits(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits


def dedupe_by_phone(providers: list[Provider]) -> tuple[list[Provider], list[tuple[Provider, str]]]:
    """Within each category, one business per phone number (keep the most-reviewed).

    The same business can still appear in several categories, since each
    category entry has already passed that category's allowlist.
    """
    kept: dict[tuple[str, str], Provider] = {}
    dropped: list[tuple[Provider, str]] = []
    for p in sorted(providers, key=lambda p: p.review_count, reverse=True):
        key = (p.category, normalize_phone_digits(p.phone))
        if key in kept:
            dropped.append((p, f"duplicate phone of {kept[key].name}"))
        else:
            kept[key] = p
    return list(kept.values()), dropped


def select(
    raw: list[Provider], overrides: dict[str, list[dict]] | None = None
) -> tuple[list[Provider], list[tuple[Provider, str]]]:
    """Keep providers likely to be real, local, operational, reachable, and in-category.

    1. Hard filters (exclusion_reason): status, retail, type allowlist,
       overrides, phone, MIN_REVIEW_COUNT, location, service area.
    2. Dedupe by phone within each category.
    3. Keep up to MAX_PER_ANCHOR per category per nearest anchor city. Never
       backfills with providers that failed step 1; fewer is fine.

    Returns (selected, excluded) where excluded pairs each provider with a reason.
    """
    overrides = overrides or {}
    excluded: list[tuple[Provider, str]] = []
    passed: list[Provider] = []
    for p in raw:
        reason = exclusion_reason(p, overrides)
        if reason:
            excluded.append((p, reason))
        else:
            passed.append(p)

    passed, duplicates = dedupe_by_phone(passed)
    excluded.extend(duplicates)

    buckets: dict[tuple[str, str], list[Provider]] = defaultdict(list)
    for p in passed:
        buckets[(p.category, nearest_anchor(p)[0])].append(p)

    selected = []
    for (category, city), providers in buckets.items():
        providers.sort(key=lambda p: bayesian_rating(p.rating, p.review_count), reverse=True)
        selected.extend(providers[:MAX_PER_ANCHOR])
        for p in providers[MAX_PER_ANCHOR:]:
            excluded.append((p, f"below top {MAX_PER_ANCHOR} for {category} near {city}"))
    selected.sort(key=lambda p: (p.category, p.city or "", p.name))
    return selected, excluded


def print_exclusions(excluded: list[tuple[Provider, str]]) -> None:
    """Log every exclusion, grouped by reason type, so the cut is auditable."""
    print(f"\nExcluded {len(excluded)}:")
    for p, reason in sorted(excluded, key=lambda e: (e[1], e[0].category, e[0].name)):
        print(f"  {reason:<45} {p.category:<17} {p.name}")


def review(providers: list[Provider]) -> None:
    """Compact table of the selected providers, for manual review."""
    header = f"{'name':<38} {'category':<17} {'near':<15} {'primary_type':<24} {'rating':>6} {'reviews':>7}  location"
    print(header)
    print("-" * len(header))
    for p in providers:
        anchor = nearest_anchor(p)
        location = "exact" if p.location_source == "exact" else f"centroid ({p.location_source})"
        rating = f"{p.rating:.1f}" if p.rating is not None else "-"
        print(
            f"{p.name[:38]:<38} {p.category:<17} {(anchor[0] if anchor else '-'):<15} "
            f"{(p.primary_type or '-')[:24]:<24} {rating:>6} {p.review_count:>7}  {location}"
        )
    print_coverage(providers)


def print_coverage(providers: list[Provider]) -> None:
    """Count per category x nearest anchor city. '!' marks cells under THIN_CELL_WARNING."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for p in providers:
        anchor = nearest_anchor(p)
        if anchor:
            counts[(p.category, anchor[0])] += 1
    cities = list(ANCHOR_ZIPS)
    print(f"\nCoverage (! = fewer than {THIN_CELL_WARNING}):")
    print(f"  {'':<18}" + "".join(f"{c:>17}" for c in cities))
    for cat in CATEGORIES:
        cells = []
        for city in cities:
            n = counts.get((cat, city), 0)
            cells.append(f"{n}{'!' if n < THIN_CELL_WARNING else ' '}".rjust(17))
        print(f"  {cat:<18}" + "".join(cells))


def write(path: Path, providers: list[Provider]) -> None:
    path.write_text(json.dumps([p.model_dump(mode="json") for p in providers], indent=2))


def print_summary(providers: list[Provider]) -> None:
    by_cat: dict[str, int] = defaultdict(int)
    for p in providers:
        by_cat[p.category] += 1
    for cat in CATEGORIES:
        print(f"  {cat:<18} {by_cat.get(cat, 0)}")


def dry_run() -> None:
    queries = build_queries()
    print(f"{len(queries)} Text Search requests (pageSize={PAGE_SIZE}, no pagination):\n")
    for _, _, text_query in queries:
        print(f"  {text_query}")
    print(f"\nX-Goog-FieldMask: {FIELD_MASK}")


def load(path: Path) -> list[Provider]:
    return [Provider(**p) for p in json.loads(path.read_text())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, no API calls")
    parser.add_argument("--reselect", action="store_true", help="rebuild providers.json from providers_raw.json")
    parser.add_argument("--review", action="store_true", help="print a table of providers.json")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.review and not args.reselect:
        review(load(PROVIDERS_PATH))
        return

    if args.reselect:
        raw = load(RAW_PATH)
    else:
        if not GOOGLE_PLACES_API_KEY:
            sys.exit("GOOGLE_PLACES_API_KEY is not set (see .env.example).")
        raw = fetch_all(GOOGLE_PLACES_API_KEY)
        write(RAW_PATH, raw)
        print(f"\nWrote {len(raw)} raw providers to {RAW_PATH.name}")

    selected, excluded = select(raw, load_overrides())
    write(PROVIDERS_PATH, selected)
    print_exclusions(excluded)
    print(f"\nWrote {len(selected)} selected providers to {PROVIDERS_PATH.name}:")
    print_summary(selected)

    if args.review:
        print()
        review(selected)


if __name__ == "__main__":
    main()
