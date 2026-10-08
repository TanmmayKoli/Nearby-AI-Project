import pytest
from pydantic import ValidationError

from agent.categories import CATEGORIES, HARD_STOP_FLAGS, OUT_OF_SCOPE
from agent.state import (
    ExtractedFields,
    Lead,
    LeadState,
    merge_extracted,
    missing_required_fields,
    normalize_phone,
    normalize_zip,
)


def test_normalizers():
    assert normalize_zip("95616") == "95616"
    assert normalize_zip("95616-1234") == "95616"
    assert normalize_zip("9561") is None
    assert normalize_phone("530.555.0123") == "(530) 555-0123"
    assert normalize_phone("+1 (530) 555-0123") == "(530) 555-0123"
    assert normalize_phone("555-0123") is None


def test_null_never_overwrites():
    state = LeadState(zip="95616", name="Jane")
    updates = merge_extracted(state, ExtractedFields(urgency="emergency"))
    assert updates == {"urgency": "emergency"}


def test_correction_overwrites_and_details_merge():
    state = LeadState(zip="95616", category_details={"source": "storm"})
    extracted = ExtractedFields(zip="95618", category_details={"extent": "1 inch, whole basement"})
    updates = merge_extracted(state, extracted)
    assert updates["zip"] == "95618"
    assert updates["category_details"] == {"source": "storm", "extent": "1 inch, whole basement"}


def test_invalid_phone_is_dropped():
    updates = merge_extracted(LeadState(), ExtractedFields(contact_phone="call me maybe"))
    assert "contact_phone" not in updates


def test_safety_flags_accumulate_without_duplicates():
    state = LeadState(safety_flags=["active_flooding"])
    updates = merge_extracted(state, ExtractedFields(safety_flags=["active_flooding", "water_near_electrical"]))
    assert updates["safety_flags"] == ["active_flooding", "water_near_electrical"]


def test_missing_fields_order_and_low_confidence():
    state = LeadState(category="plumbing", category_confidence=0.4)
    assert missing_required_fields(state) == [
        "category", "facts", "urgency", "zip", "name", "contact",
    ]
    full = LeadState(
        category="plumbing", category_confidence=0.9, facts=["Leaking sink"],
        urgency="within_48h", zip="95616", name="Jane", contact_email="j@x.com",
    )
    assert missing_required_fields(full) == []


def test_lead_requires_consent_and_provider():
    base = dict(
        lead_id="x", category="plumbing", problem_description="Leak", facts=["Sink leaking"],
        urgency="flexible", zip="95616", name="Jane",
    )
    provider = [{"name": "A"}]
    with pytest.raises(ValidationError):  # no contact
        Lead(**base, matched_providers=provider, consent_to_share=True)
    base["contact_phone"] = "(530) 555-0123"
    with pytest.raises(ValidationError):
        Lead(**base, matched_providers=[{"name": "A"}], consent_to_share=False)
    with pytest.raises(ValidationError):
        Lead(**base, matched_providers=[], consent_to_share=True)
    Lead(**base, matched_providers=[{"name": "A"}], consent_to_share=True)


def test_category_table_is_consistent():
    for key, spec in CATEGORIES.items():
        assert spec.key == key
        assert 2 <= len(spec.detail_questions) <= 3
        assert spec.search_queries
        keys = [q.key for q in spec.detail_questions]
        assert len(keys) == len(set(keys))


def test_new_facts_append_and_skip_duplicates():
    state = LeadState(facts=["Water in basement"])
    updates = merge_extracted(
        state, ExtractedFields(new_facts=["Water in basement", "About 1 inch deep", "About 1 inch deep"])
    )
    assert updates["facts"] == ["Water in basement", "About 1 inch deep"]
    assert "facts" not in merge_extracted(state, ExtractedFields(new_facts=["Water in basement"]))
    assert "problem_description" not in updates


def test_category_requires_confidence():
    with pytest.raises(ValidationError):
        ExtractedFields(category="plumbing")
    ExtractedFields(category="plumbing", category_confidence=0.8)


def test_only_gas_is_hard_stop_and_out_of_scope_covers_all_reasons():
    assert HARD_STOP_FLAGS == {"gas_smell"}
    assert set(OUT_OF_SCOPE) == {"utility_outage", "wildlife_removal", "other"}


def test_out_of_scope_reason_carries_over():
    updates = merge_extracted(LeadState(), ExtractedFields(out_of_scope_reason="utility_outage"))
    assert updates["out_of_scope_reason"] == "utility_outage"
    assert "out_of_scope_reason" not in merge_extracted(
        LeadState(out_of_scope_reason="utility_outage"), ExtractedFields()
    )


def test_wire_schema_has_no_optional_fields_and_list_details():
    """Regression: optional fields made the API reject the schema ("Schema is too
    complex"), and a dict-typed category_details was sent as an object that
    could hold no keys. Check the schema LangChain actually sends."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    captured = {}

    class Capture(ChatAnthropic):
        def _create(self, payload):
            captured.update(payload)
            raise RuntimeError("stop before network")

    llm = Capture(model="claude-sonnet-5-5", max_tokens=10, api_key="test", max_retries=0)
    with pytest.raises(RuntimeError):
        llm.with_structured_output(ExtractedFields, method="json_schema").invoke([HumanMessage("hi")])
    schema = captured["output_config"]["format"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    unions = [k for k, v in schema["properties"].items() if "anyOf" in v or isinstance(v.get("type"), list)]
    assert len(unions) <= 16  # documented limit for union-typed params
    assert schema["properties"]["category_details"]["type"] == "array"


def test_category_details_accepts_dict_or_list_and_merges_to_dict():
    a = ExtractedFields(category_details={"source": "storm"})
    b = ExtractedFields(category_details=[{"key": "extent", "value": "1 inch"}])
    state = LeadState(category_details={"still_active": "yes"})
    assert merge_extracted(state, a)["category_details"] == {"still_active": "yes", "source": "storm"}
    assert merge_extracted(state, b)["category_details"] == {"still_active": "yes", "extent": "1 inch"}
