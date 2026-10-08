"""What a provider receives (agent/dispatch.py): built from the Lead only."""

import json

from agent.dispatch import lead_to_provider_message, provider_view, short_name
from agent.state import Lead

PROVIDER = {"name": "Test Restoration Co", "city": "Davis", "phone": "(530) 555-0199", "lat": 38.54, "lng": -121.74,
            "place_id": "abc", "source_query": "water damage restoration in Davis, CA", "distance_miles": 2.0}


def make_lead(**overrides) -> Lead:
    base = dict(
        lead_id="abc12345", category="water_damage", urgency="emergency", zip="95616",
        problem_description="Storm water is coming into the basement, about an inch deep.",
        facts=["Water coming into basement"], name="Jane Doe", contact_phone="(530) 555-0123",
        category_details={"still_active": "yes", "extent": "about an inch", "source": "unknown"},
        matched_providers=[PROVIDER, {**PROVIDER, "name": "B"}, {**PROVIDER, "name": "C"}], consent_to_share=True,
    )
    return Lead(**{**base, **overrides})


def test_provider_message_has_what_a_pro_needs():
    msg = lead_to_provider_message(make_lead())
    assert msg.splitlines()[0] == "New lead: Emergency · Water damage restoration · 95616"
    assert "Storm water is coming into the basement, about an inch deep." in msg
    assert "Still active: yes" in msg and "Extent: about an inch" in msg
    assert "Source" not in msg  # "unknown" answers are left out
    assert "Customer: Jane D." in msg and "Jane Doe" not in msg
    assert "Phone: (530) 555-0123" in msg
    assert msg.splitlines()[-1] == "Shared with 3 local pros · customer consented · ref abc12345"


def test_provider_message_never_includes_internal_fields():
    msg = lead_to_provider_message(make_lead())
    for internal in ["38.54", "-121.74", "place_id", "source_query", "Davis, CA", "distance", "lat", "lng"]:
        assert internal not in msg


def test_email_only_extras_and_one_provider():
    lead = make_lead(contact_phone=None, contact_email="jane@example.com", address="12 Elm St",
                     availability="weekday mornings", safety_flags=["water_near_electrical"],
                     matched_providers=[PROVIDER], urgency="within_week")
    v = provider_view(lead)
    assert v.header.startswith("Within a week") and v.contact == "Email: jane@example.com"
    assert v.extras == ["Address: 12 Elm St", "Availability: weekday mornings", "Safety: water near electrical"]
    assert v.footer.startswith("Shared with 1 local pro ·")


def test_short_name():
    assert short_name("Jane Doe") == "Jane D."
    assert short_name("Mary Ann van der berg") == "Mary B."
    assert short_name("Dana") == "Dana"


def test_message_matches_a_saved_lead_file(tmp_path):
    """Round trip: a lead as written to leads/*.json gives the same message."""
    lead = make_lead()
    reloaded = Lead(**json.loads(json.dumps(lead.model_dump(mode="json"))))
    assert lead_to_provider_message(reloaded) == lead_to_provider_message(lead)
