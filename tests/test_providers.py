"""Provider tests use made-up providers (test fixtures only, never written to providers.json)."""

from datetime import datetime, timezone

import itertools
import json

import pytest

from providers import fetch
from providers.fetch import build_queries, parse_city_zip, select, to_provider
from providers.geo import zip_to_latlng
from providers.match import bayesian_rating, match_providers
from providers.models import Provider

NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)
_phones = itertools.count(100)


def make(
    name: str, zip_code: str, rating: float, reviews: int, category: str = "plumbing",
    phone: str | None = "auto", status: str | None = "OPERATIONAL", primary_type: str = "plumber",
) -> Provider:
    """Each provider gets a unique phone unless one is given (so dedupe doesn't merge them)."""
    if phone == "auto":
        phone = f"(530) 555-{next(_phones):04d}"
    lat, lng = zip_to_latlng(zip_code)
    return Provider(
        place_id=f"test-{name}", name=name, phone=phone, zip=zip_code, lat=lat, lng=lng,
        location_source="zip_centroid", business_status=status, primary_type=primary_type,
        rating=rating, review_count=reviews, category=category, source_query="test", fetched_at=NOW,
    )


PROVIDERS = [
    make("Davis Good", "95616", 4.8, 300),
    make("Davis Few Reviews", "95616", 5.0, 2),
    make("Sacramento Good", "95814", 4.8, 300),
    make("Far Away", "95670", 5.0, 900),  # Rancho Cordova, ~25+ mi from Davis
    make("No Phone", "95616", 5.0, 900, phone=None),
    make("Davis Roofer", "95616", 4.9, 500, category="roofing"),
]


def test_queries_cover_every_term_and_city():
    queries = build_queries()
    assert len(queries) == 40
    assert ("hvac", "HVAC repair", "HVAC repair in West Sacramento, CA") in queries


def test_parse_city_zip():
    assert parse_city_zip("123 Main St, Davis, CA 95616, USA") == ("Davis", "95616")
    assert parse_city_zip("Sacramento, CA, USA") == ("Sacramento", None)
    assert parse_city_zip(None) == (None, None)


def test_to_provider_prefers_exact_location_then_centroids():
    base = {"id": "x", "displayName": {"text": "X"}, "businessStatus": "OPERATIONAL", "primaryType": "plumber"}
    exact = to_provider(
        {**base, "formattedAddress": "1 A St, Davis, CA 95616, USA", "location": {"latitude": 38.5, "longitude": -121.7}},
        "plumbing", "q", NOW,
    )
    assert (exact.lat, exact.lng, exact.location_source) == (38.5, -121.7, "exact")
    assert exact.business_status == "OPERATIONAL" and exact.primary_type == "plumber"

    by_zip = to_provider({**base, "formattedAddress": "1 A St, Davis, CA 95616, USA"}, "plumbing", "q", NOW)
    assert by_zip.location_source == "zip_centroid"

    by_city = to_provider({**base, "formattedAddress": "Sacramento, CA, USA"}, "plumbing", "q", NOW)
    assert by_city.zip is None and by_city.lat is not None and by_city.location_source == "city_centroid"


def test_bayesian_rating_discounts_few_reviews():
    assert bayesian_rating(4.8, 300) > bayesian_rating(5.0, 2)
    assert bayesian_rating(None, 0) == 4.0


def test_match_ranks_close_and_reputable_first():
    names = [m["name"] for m in match_providers("plumbing", "95616", PROVIDERS)]
    assert names == ["Davis Good", "Sacramento Good", "Davis Few Reviews"]


def test_match_filters_category_phone_and_distance():
    names = {m["name"] for m in match_providers("plumbing", "95616", PROVIDERS, n=10)}
    assert "Davis Roofer" not in names
    assert "No Phone" not in names
    assert "Far Away" not in names


def test_match_unknown_or_far_zip_returns_nothing():
    assert match_providers("plumbing", "00000", PROVIDERS) == []
    assert match_providers("plumbing", "90210", PROVIDERS) == []


def test_select_filters_and_caps_per_anchor_with_reasons():
    many = [make(f"Davis {i}", "95616", 4.5, 100 + i) for i in range(5)]
    bad = [
        make("No Phone", "95616", 5.0, 999, phone=None),
        make("Closed", "95616", 5.0, 999, status="CLOSED_PERMANENTLY"),
        make("No Status", "95616", 5.0, 999, status=None),
        make("Big Box", "95616", 4.9, 999, primary_type="home_improvement_store"),
        make("Too Far", "95991", 5.0, 999),  # Yuba City, > 15 mi from every anchor
    ]
    selected, excluded = select(many + bad)
    assert [p.name for p in selected] == ["Davis 2", "Davis 3", "Davis 4"]
    reasons = {p.name: r for p, r in excluded}
    assert reasons["No Phone"] == "no phone"
    assert reasons["Closed"] == "business_status=CLOSED_PERMANENTLY"
    assert reasons["No Status"] == "business_status=None"
    assert reasons["Big Box"] == "retail primary_type=home_improvement_store"
    assert reasons["Too Far"].startswith("out of area")
    assert reasons["Davis 0"].startswith("below top 3")


def test_reselect_and_review_use_files(tmp_path, monkeypatch, capsys):
    raw_path, out_path = tmp_path / "raw.json", tmp_path / "providers.json"
    raw_path.write_text(json.dumps([p.model_dump(mode="json") for p in PROVIDERS]))
    monkeypatch.setattr(fetch, "RAW_PATH", raw_path)
    monkeypatch.setattr(fetch, "PROVIDERS_PATH", out_path)
    monkeypatch.setattr(fetch, "OVERRIDES_PATH", tmp_path / "none.json")

    monkeypatch.setattr("sys.argv", ["fetch", "--reselect", "--review"])
    fetch.main()
    written = json.loads(out_path.read_text())
    assert "No Phone" not in {p["name"] for p in written}
    out = capsys.readouterr().out
    assert "Excluded" in out and "centroid (zip_centroid)" in out and "Coverage" in out

    monkeypatch.setattr("sys.argv", ["fetch", "--review"])
    fetch.main()
    assert "Davis Good" in capsys.readouterr().out


def test_allowlist_overrides_and_review_floor():
    providers = [
        make("Real Plumber", "95616", 4.8, 50),
        make("Handyman As Plumber", "95616", 4.9, 80, primary_type="service"),  # type not allowed
        make("Roofer Typed GC", "95616", 4.9, 80, category="roofing", primary_type="general_contractor"),
        make("Excluded Everywhere", "95616", 4.9, 80),
        make("Excluded In Plumbing Only", "95616", 4.9, 80),
        make("Excluded In Plumbing Only", "95616", 4.9, 80, category="handyman", primary_type="service"),
        make("Thin", "95616", 5.0, 9),
    ]
    by_name = {p.name: p.place_id for p in providers}
    overrides = {
        by_name["Excluded Everywhere"]: [{"categories": "all", "reason": "not hireable"}],
        by_name["Excluded In Plumbing Only"]: [{"categories": ["plumbing"], "reason": "wrong trade"}],
    }
    selected, excluded = select(providers, overrides)
    assert {(p.name, p.category) for p in selected} == {
        ("Real Plumber", "plumbing"),
        ("Excluded In Plumbing Only", "handyman"),
    }
    reasons = {(p.name, p.category): r for p, r in excluded}
    assert reasons[("Handyman As Plumber", "plumbing")] == "type service not allowed for plumbing"
    assert reasons[("Roofer Typed GC", "roofing")] == "type general_contractor not allowed for roofing"
    assert reasons[("Excluded Everywhere", "plumbing")] == "override: not hireable"
    assert reasons[("Excluded In Plumbing Only", "plumbing")] == "override: wrong trade"
    assert reasons[("Thin", "plumbing")].startswith("thin listing (9 reviews")


def test_no_backfill_below_threshold():
    """A city with one good provider and several thin ones keeps just the one."""
    providers = [make("Good", "95695", 4.8, 50)] + [make(f"Thin {i}", "95695", 5.0, 3) for i in range(3)]
    selected, _ = select(providers)
    assert [p.name for p in selected] == ["Good"]


def test_dedupe_by_phone_within_category_only():
    same = "(916) 588-9345"
    providers = [
        make("Pest Pros", "95814", 5.0, 1091, category="pest_control", primary_type="service", phone=same),
        make("PEST PROS", "95814", 5.0, 2281, category="pest_control", primary_type="service", phone="+1 916.588.9345"),
        make("Combo Shop", "95814", 4.9, 500, phone="(916) 111-2222"),
        make("Combo Shop", "95814", 4.9, 500, category="hvac", primary_type="general_contractor", phone="(916) 111-2222"),
    ]
    selected, excluded = select(providers)
    assert [p.name for p in selected if p.category == "pest_control"] == ["PEST PROS"]  # more reviews wins
    assert {p.category for p in selected if p.name == "Combo Shop"} == {"plumbing", "hvac"}
    assert ("Pest Pros", "duplicate phone of PEST PROS") in {(p.name, r) for p, r in excluded}


def test_overrides_file_is_valid():
    from agent.categories import CATEGORIES

    overrides = fetch.load_overrides()
    assert overrides
    for entries in overrides.values():
        for e in entries:
            assert e["reason"]
            assert e["categories"] == "all" or set(e["categories"]) <= set(CATEGORIES)


def test_coverage_flags_thin_cells(capsys):
    fetch.print_coverage([make("Only One", "95616", 4.8, 50)])
    out = capsys.readouterr().out
    plumbing_row = next(line for line in out.splitlines() if line.strip().startswith("plumbing"))
    assert "1!" in plumbing_row and "0!" in plumbing_row


def test_emergency_weights_distance_more_heavily():
    near_ok = make("Near OK", "95616", 4.6, 25)
    far_great = make("Far Great", "95814", 5.0, 900)  # ~15 mi from Davis
    assert [m["name"] for m in match_providers("plumbing", "95616", [near_ok, far_great], urgency="flexible")][0] == "Far Great"
    assert [m["name"] for m in match_providers("plumbing", "95616", [near_ok, far_great], urgency="emergency")][0] == "Near OK"


def test_real_data_emergency_in_davis_ranks_servpro_davis_above_far_providers():
    """On the real providers.json: SERVPRO of Davis (2.4 mi) must beat anything 15+ mi away."""
    matches = match_providers("water_damage", "95616", urgency="emergency")
    names = [m["name"] for m in matches]
    assert "SERVPRO of Davis" in names
    davis_rank = names.index("SERVPRO of Davis")
    for i, m in enumerate(matches):
        if m["distance_miles"] >= 15:
            assert davis_rank < i, names


def test_nearest_good_match_always_included():
    """Fixture: two strong far providers + a closer good one that loses on score."""
    far = [make(f"Far {i}", "95814", 5.0, 900) for i in range(3)]
    near = make("Near", "95616", 4.5, 15)
    names = [m["name"] for m in match_providers("plumbing", "95616", far + [near], urgency="flexible")]
    assert "Near" in names and names[-1] == "Near" and len(names) == 3


@pytest.mark.parametrize("urgency", ["emergency", "within_48h", "within_week", "flexible", None])
def test_real_data_roofing_95616_includes_lucas_roofing(urgency):
    """Lucas Roofing (1.8 mi) is the closest roofer to 95616; it must make the list."""
    names = [m["name"] for m in match_providers("roofing", "95616", urgency=urgency)]
    assert "Lucas Roofing" in names, names
