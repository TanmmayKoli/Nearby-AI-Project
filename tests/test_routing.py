"""Phase 4 routing tests (no API): safety, clarify, out-of-scope, edge cases.

Uses the scripted FakeLLM from test_graph. If a path is supposed to skip the
LLM, the fake is given no scripted outputs, so any unexpected call fails.
"""

import pytest

from agent import nodes
from agent.categories import (
    GAS_EMERGENCY_MESSAGE,
    OUT_OF_SCOPE,
    SAFETY_NOTICES,
)
from agent.faithfulness import unsupported_claims
from agent.prompts import SUMMARIZE_SYSTEM, URGENCY_GUIDE, after_end_prompt, ask_system_prompt
from agent.safety import detect_hazards
from agent.state import LeadState
from tests.test_graph import PROVIDER, FakeLLM, make_graph, turn

ELECTRICAL = {"category": "electrical", "category_confidence": 0.9, "new_facts": ["Kitchen outlet sparked"]}


# --- safety ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I smell gas in the kitchen", ["gas_smell"]),
        ("it smells like rotten eggs", ["gas_smell"]),
        ("I don't smell gas or anything", []),
        ("my gas stove won't light", []),
        ("one of the outlets sparked", ["sparks_or_smoke"]),
        ("no sparks, it just stopped working", []),
        ("my smoke detector keeps beeping", []),
        ("smells like something burning near the panel", ["burning_smell"]),
        ("water in the basement", []),
    ],
)
def test_detect_hazards(text, expected):
    assert detect_hazards(text) == expected


def test_gas_skips_llm_and_ends(tmp_path):
    llm = FakeLLM()  # no scripted outputs: any LLM call would fail
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "I smell gas near the water heater", tmp_path)
    assert s.status == "emergency" and s.messages[-1].content == GAS_EMERGENCY_MESSAGE
    assert "1-800-743-5000" in s.messages[-1].content and s.lead_id is None
    # After the end: the safety instruction is repeated first (by code), then the reply.
    llm.texts.append("Once PG&E says it's safe, tap Start a new request and I'll find you a plumber.")
    reply = turn(graph, tid, "ok what now", tmp_path).messages[-1].content
    assert reply.startswith(nodes.GAS_REMINDER) and reply.endswith("I'll find you a plumber.")


def test_gas_caught_by_extractor_when_keywords_miss(tmp_path):
    llm = FakeLLM(extractions=[{"new_facts": ["Odd odor by the furnace"], "safety_flags": ["gas_smell"]}])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "there's a weird sulfury odor by the furnace", tmp_path)
    assert s.status == "emergency" and s.last_route == "emergency_response"


def test_sparks_notice_first_raises_urgency_floor_and_continues(tmp_path):
    llm = FakeLLM(
        extractions=[{**ELECTRICAL, "urgency": "within_week"}, {"category_details": {"scope": "just that outlet"}}],
        texts=["Is it just that one outlet, or more of the house?", "Any burning smell or a tripping breaker?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "One of my outlets sparked when I plugged in the toaster", tmp_path)
    reply = s.messages[-1].content
    assert reply.startswith(SAFETY_NOTICES["sparks_or_smoke"]) and reply.endswith("more of the house?")
    # single spark, extractor said within_week -> floor of within_48h (not emergency)
    assert s.urgency == "within_48h" and s.status == "in_progress" and s.warned == ["sparks_or_smoke"]
    assert s.last_route.startswith("ask_next:detail")  # urgency not asked: the floor set it
    s = turn(graph, tid, "just that outlet", tmp_path)
    assert SAFETY_NOTICES["sparks_or_smoke"] not in s.messages[-1].content  # warned once


def test_burning_after_sparks_warning_is_not_repeated(tmp_path):
    llm = FakeLLM(
        extractions=[{**ELECTRICAL, "urgency": "emergency"}, {"safety_flags": ["burning_smell"]}],
        texts=["Is it just that one outlet?", "Did the breaker trip?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    turn(graph, tid, "my outlet sparked", tmp_path)
    s = turn(graph, tid, "just that one, and it smelled like burning", tmp_path)
    assert s.messages[-1].content == "Did the breaker trip?"
    assert set(s.warned) == {"sparks_or_smoke", "burning_smell"}


def test_water_near_electrical_gets_tip_without_forcing_urgency(tmp_path):
    llm = FakeLLM(
        extractions=[{"category": "water_damage", "category_confidence": 0.9, "new_facts": ["Water near the panel"],
                      "safety_flags": ["water_near_electrical"]}],
        texts=["How soon do you need someone?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "there's water on the floor near my breaker panel", tmp_path)
    assert s.messages[-1].content.startswith(SAFETY_NOTICES["water_near_electrical"])
    assert s.urgency is None and s.last_route == "ask_next:urgency"


# --- clarify + out of scope ----------------------------------------------------------


def test_clarify_once_then_go_with_best_guess(tmp_path):
    ambiguous = {"category": "plumbing", "category_confidence": 0.5, "new_facts": ["Water under the sink cabinet"]}
    llm = FakeLLM(
        extractions=[ambiguous, {"category": "plumbing", "category_confidence": 0.55}],
        texts=["Is a pipe leaking, or is there standing water to clean up?", "How soon do you need someone?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "there's water under my sink", tmp_path)
    assert s.last_route == "clarify"
    assert "plumbing" in llm.prompts[0] and "water damage restoration" in llm.prompts[0]
    s = turn(graph, tid, "hmm kind of both?", tmp_path)
    assert s.category_locked == "plumbing" and s.last_route == "ask_next:urgency"
    assert s.asked.count("clarify") == 1


@pytest.mark.parametrize("reason", ["utility_outage", "other"])
def test_out_of_scope_redirects_and_ends(tmp_path, reason):
    llm = FakeLLM(extractions=[{"new_facts": ["Something outside our scope"], "out_of_scope_reason": reason}])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "there's a raccoon in my attic", tmp_path)
    assert s.status == "out_of_scope" and s.out_of_scope_reason == reason
    assert s.messages[-1].content == OUT_OF_SCOPE[reason] and s.lead_id is None
    llm.texts.append("Hi! If there's something at home I can help with, tap Start a new request.")
    s = turn(graph, tid, "hello?", tmp_path)
    assert s.messages[-1].content.startswith("Hi!") and s.status == "out_of_scope"


# --- contact edge cases ---------------------------------------------------------------

READY = {"category": "plumbing", "category_confidence": 0.9, "urgency": "flexible", "new_facts": ["Sink dripping"],
         "category_details": {"fixture": "sink", "symptom": "leaking"}, "zip": "95616", "name": "Jane"}


def test_declined_phone_then_email_then_explain_then_stop(tmp_path):
    llm = FakeLLM(
        extractions=[READY, {"declined_contact": ["phone"]}, {"declined_contact": ["email"]}, {}],
        texts=["What's the best number?", "Would email work instead?", "A pro needs some way to reach you..."],
    )
    graph, tid = make_graph(llm, tmp_path)
    assert turn(graph, tid, "sink dripping, 95616, I'm Jane", tmp_path).last_route == "ask_next:contact"
    assert turn(graph, tid, "I'd rather not give my number", tmp_path).last_route == "ask_next:contact_email"
    assert turn(graph, tid, "no email either", tmp_path).last_route == "ask_next:contact_required"
    s = turn(graph, tid, "no thanks", tmp_path)
    assert s.status == "declined" and s.last_route == "no_contact" and s.lead_id is None
    assert "Test Restoration Co: (530) 555-0199" in s.messages[-1].content  # still gets pros' numbers


def test_ignored_email_offer_counts_as_declined():
    state = LeadState(**{k: v for k, v in READY.items() if k != "new_facts"}, facts=["Sink dripping"],
                      declined=["phone"], asked=["contact", "contact_email"])
    assert nodes.next_target(state) == "contact_required"


def test_contact_question_not_repeated_forever():
    state = LeadState(**{k: v for k, v in READY.items() if k != "new_facts"}, facts=["Sink dripping"],
                      asked=["contact", "contact"])
    assert nodes.next_target(state) == "contact_required"
    state.asked.append("contact_required")
    assert nodes.next_target(state) == "no_contact"


# --- idk, problem change, confirm edits ------------------------------------------------


def test_idk_detail_marked_unknown_and_never_repeated(tmp_path):
    llm = FakeLLM(
        extractions=[{"category": "plumbing", "category_confidence": 0.9, "urgency": "flexible", "new_facts": ["Drain is slow"]}, {}],
        texts=["What's affected?", "What's your zip?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    assert turn(graph, tid, "my drain is slow", tmp_path).last_route == "ask_next:detail:fixture"
    s = turn(graph, tid, "idk", tmp_path)
    assert s.category_details["fixture"] == "unknown"
    # after the first "not sure", no more detail questions for this lead
    assert s.last_route == "ask_next:zip" and s.asked.count("detail:fixture") == 1


def test_problem_change_updates_category_and_drops_stale_facts_and_details(tmp_path):
    llm = FakeLLM(
        extractions=[
            {"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Pipe leaking under sink", "Started yesterday"],
             "category_details": {"fixture": "pipe"}},
            {"category": "roofing", "category_confidence": 0.9, "retracted_facts": ["Pipe leaking under sink"],
             "new_facts": ["Roof leaking into the kitchen"]},
        ],
        texts=["How soon?", "How soon?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    turn(graph, tid, "pipe leaking under my sink since yesterday", tmp_path)
    s = turn(graph, tid, "actually it's the roof, it drips into the kitchen", tmp_path)
    assert s.category == "roofing"
    assert s.facts == ["Started yesterday", "Roof leaking into the kitchen"]
    assert "fixture" not in s.category_details


def test_confirm_change_use_email_instead_reshows_summary(tmp_path):
    ready = {**READY, "contact_phone": "5305550123"}
    llm = FakeLLM(
        extractions=[ready, {"contact_email": "jane@example.com", "remove_fields": ["contact_phone"]}],
        texts=["Sink is dripping.", "Sink is dripping."],
        consents=["changes", "yes"],
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "everything at once", tmp_path)
    assert s.status == "awaiting_confirm" and "(530) 555-0123" in s.messages[-1].content
    s = turn(graph, tid, "use my email instead: jane@example.com", tmp_path)
    assert s.status == "awaiting_confirm" and s.contact_phone is None
    assert "jane@example.com" in s.messages[-1].content and "555-0123" not in s.messages[-1].content
    assert turn(graph, tid, "yes", tmp_path).status == "converted"


# --- formatting + prompts ---------------------------------------------------------------


def test_confirm_description_is_its_own_paragraph():
    state = LeadState(category="plumbing", category_confidence=0.9, urgency="within_week", zip="95616",
                      name="Jane", contact_phone="(530) 555-0123", problem_description="Toilet keeps running.",
                      matched_providers=[PROVIDER])
    blocks = nodes.confirm_message(state).split("\n\n")
    assert "**Plumbing** · Within a week · 95616" in blocks
    assert "Toilet keeps running." in blocks  # not glued to the header line
    assert "2.0 mi" in nodes.confirm_message(state)


def test_urgency_guide_calibrates_routine_problems():
    assert "running toilet" in URGENCY_GUIDE and "within_week" in URGENCY_GUIDE


def test_ask_prompts_for_new_targets():
    state = LeadState(**{k: v for k, v in READY.items() if k != "new_facts"}, facts=["x"])
    assert "email instead" in ask_system_prompt(state, "contact_email", False)
    assert "some way to reach them" in ask_system_prompt(state, "contact_required", False)


# --- Phase 4 review fixes ----------------------------------------------------------

WILDLIFE_PRO = {**PROVIDER, "name": "Critter Co", "category": "wildlife_removal"}
RACCOON = {"category": "wildlife_removal", "category_confidence": 0.95, "urgency": "within_week",
           "new_facts": ["Raccoon living in the attic", "It tore up a vent on the roof"],
           "category_details": {"animal": "raccoon", "location": "attic", "damage": "torn roof vent"},
           "zip": "95616", "name": "Alex Kim", "contact_phone": "5305550188"}


def _raccoon_lead(tmp_path, extraction=RACCOON):
    llm = FakeLLM(extractions=[extraction], texts=["A raccoon is living in the attic."], consents=["yes"])
    graph, tid = make_graph(llm, tmp_path, matches=(WILDLIFE_PRO,))
    turn(graph, tid, "there's a raccoon in my attic and it tore up a roof vent", tmp_path)
    return llm, graph, tid, turn(graph, tid, "yes", tmp_path)


def test_raccoon_is_a_wildlife_lead_with_damage_as_a_fact_and_repair_offer(tmp_path):
    _, _, _, s = _raccoon_lead(tmp_path)
    assert s.status == "converted" and s.category == "wildlife_removal"
    assert "It tore up a vent on the roof" in s.facts  # damage stays in THIS lead
    assert (s.repair_offer, s.repair_category) == ("torn roof vent", "roofing")
    assert s.messages[-1].content.endswith("Want help finding a roofer for the torn roof vent too?")


@pytest.mark.parametrize("damage", ["none", "No damage noticed", "unknown", "not sure", "No, the vent looks fine"])
def test_no_repair_offer_without_damage(tmp_path, damage):
    details = {**RACCOON["category_details"], "damage": damage}
    no_damage = {**RACCOON, "category_details": details, "new_facts": ["Raccoon living in the attic"]}
    _, _, _, s = _raccoon_lead(tmp_path, no_damage)
    assert s.status == "converted" and s.repair_offer is None and "too?" not in s.messages[-1].content


@pytest.mark.parametrize(
    "facts, details, expected",
    [
        (["It tore up a vent on the roof"], {}, ("roofing", "a roofer", "torn vent")),
        ([], {"damage": "torn roof vent"}, ("roofing", "a roofer", "torn roof vent")),
        ([], {"damage": "some shingles pulled off"}, ("roofing", "a roofer", "shingles")),
        (["Raccoon chewed through the wires in the attic"], {}, ("electrical", "an electrician", "chewed wires")),
        ([], {"damage": "chewed wiring"}, ("electrical", "an electrician", "chewed wiring")),
        (["Squirrels chewed a cable"], {}, ("electrical", "an electrician", "chewed cable")),
        ([], {"damage": "soiled insulation"}, ("handyman", "a handyman", "soiled insulation")),
        (["There's a hole in the ceiling of the closet"], {}, ("handyman", "a handyman", "hole in the ceiling")),
        ([], {"damage": "ceiling hole"}, ("handyman", "a handyman", "ceiling hole")),
        ([], {"damage": "drywall is torn up"}, ("handyman", "a handyman", "drywall")),
        # damage detail is checked before facts; first rule wins within a text
        (["Wires chewed"], {"damage": "torn vent and chewed wiring"}, ("roofing", "a roofer", "torn vent")),
        (["Raccoon in the attic at night"], {"damage": "none"}, None),
        (["No damage to the vent or wiring"], {}, None),  # negated
        ([], {"location": "attic", "animal": "raccoon"}, None),
    ],
)
def test_repair_offer_mapping(facts, details, expected):
    state = LeadState(category="wildlife_removal", facts=facts, category_details=details)
    offer = nodes.repair_to_offer(state)
    assert (offer and (offer.category, offer.who, offer.damage)) == expected


def test_repair_offer_only_for_wildlife():
    state = LeadState(category="roofing", category_details={"damage": "torn vent"})
    assert nodes.repair_to_offer(state) is None


def test_wildlife_damage_question_asked_even_after_detail_budget():
    """The opener answered animal + location (the 2-question budget), but damage
    decides the repair offer, so it's still asked, once."""
    base = dict(category="wildlife_removal", category_confidence=0.95, facts=["Raccoon in attic"], urgency="within_week",
                category_details={"animal": "raccoon", "location": "attic"})
    assert nodes.next_target(LeadState(**base)) == "detail:damage"
    assert nodes.next_target(LeadState(**base, asked=["detail:damage"])) == "zip"  # asked once, not repeated
    answered = LeadState(**{**base, "category_details": {**base["category_details"], "damage": "none"}})
    assert nodes.next_target(answered) == "zip"


def test_after_end_never_changes_the_lead(tmp_path):
    """No extraction runs after the end (none is scripted: it would fail), so a
    'change my number' message can't touch the lead or state."""
    llm, graph, tid, s = _raccoon_lead(tmp_path)
    lead_before = (tmp_path / "leads" / f"{s.lead_id}.json").read_text()
    llm.texts.append("That request is already sent. Tap Start a new request to use a different number.")
    after = turn(graph, tid, "actually use 916-555-0100 instead", tmp_path)
    assert after.contact_phone == s.contact_phone == "(530) 555-0188" and after.status == "converted"
    assert (tmp_path / "leads" / f"{s.lead_id}.json").read_text() == lead_before
    assert "find a roofing pro for the torn roof vent" in llm.prompts[-1]  # repair offer is in the prompt


def test_after_end_llm_failure_apologizes_and_changes_nothing(tmp_path):
    llm, graph, tid, s = _raccoon_lead(tmp_path)
    llm.texts.append(TimeoutError("slow"))
    after = turn(graph, tid, "thanks", tmp_path)
    assert after.messages[-1].content == nodes.LLM_ERROR_REPLY and after.status == "converted"
    assert after.lead_id == s.lead_id


def test_after_end_prompt_describes_each_ending():
    assert "was created for these pros: Critter Co" in after_end_prompt(
        LeadState(status="converted", lead_id="abc", matched_providers=[WILDLIFE_PRO]))
    assert "gas smell" in after_end_prompt(LeadState(status="emergency"))
    assert "chose not to share contact" in after_end_prompt(LeadState(status="declined"))
    assert "outside the area" in after_end_prompt(LeadState(status="out_of_scope", out_of_scope_reason="out_of_area", zip="90012"))
    assert "Yes, find a repair pro" not in after_end_prompt(LeadState(status="declined"))


def test_carried_zip_and_contact_are_not_asked_again(tmp_path):
    from agent.graph import run_turn

    llm = FakeLLM(extractions=[{"category": "roofing", "category_confidence": 0.9, "urgency": "within_week",
                                "new_facts": ["Torn roof vent"], "category_details": {"active_leak": "no", "roof_type": "shingle"}}],
                  texts=["Torn roof vent, no leak inside."])
    graph, tid = make_graph(llm, tmp_path)
    s = run_turn(graph, tid, "I need the torn roof vent repaired", transcripts_dir=None,
                 initial={"zip": "95616", "name": "Alex Kim", "contact_phone": "(530) 555-0188"})
    assert s.zip == "95616" and s.name == "Alex Kim" and s.status == "awaiting_confirm"
    assert "zip" not in s.asked and "contact" not in s.asked


def test_animal_bite_gets_safety_notice_then_continues(tmp_path):
    bite = {**RACCOON, "safety_flags": ["animal_contact"], "zip": None, "name": None, "contact_phone": None}
    llm = FakeLLM(extractions=[bite], texts=["What's your zip code?"])
    graph, tid = make_graph(llm, tmp_path, matches=(WILDLIFE_PRO,))
    s = turn(graph, tid, "a raccoon scratched my son in the attic", tmp_path)
    assert s.messages[-1].content.startswith(SAFETY_NOTICES["animal_contact"])
    assert s.status == "in_progress" and s.last_route == "ask_next:zip"


def test_cause_questions_skipped_when_user_unsure_but_symptoms_still_asked():
    base = dict(category="plumbing", category_confidence=0.9, facts=["Ceiling stain under bathroom"], urgency="within_week")
    assert nodes.next_target(LeadState(**base)) == "detail:fixture"
    unsure = LeadState(**base, cause_unknown=True)
    assert nodes.next_target(unsure) == "detail:symptom"  # fixture (a cause question) skipped


def test_cause_unknown_is_sticky():
    from agent.state import ExtractedFields, merge_extracted

    assert merge_extracted(LeadState(), ExtractedFields(cause_unknown=True))["cause_unknown"] is True
    assert "cause_unknown" not in merge_extracted(LeadState(cause_unknown=True), ExtractedFields(cause_unknown=False))


def test_no_double_period_after_provider_name_ending_in_period():
    providers = [{**PROVIDER, "name": "Wisterman Electric"}, {**PROVIDER, "name": "Woodland Electrical Inc."}]
    state = LeadState(urgency="within_week", matched_providers=providers)
    msg = nodes.final_message(state, "abc")
    assert "Inc.." not in msg and "Woodland Electrical Inc. They'll" in msg
    assert "Wisterman Electric. They'll" in nodes.final_message(LeadState(matched_providers=providers[:1]), "abc")


def test_urgency_guide_is_conservative():
    for phrase in ["NEVER round up", "next few days", "this week", "ASAP", "today, tomorrow"]:
        assert phrase in URGENCY_GUIDE


def test_summarize_prompt_forbids_scope_claims():
    assert "limited to" in SUMMARIZE_SYSTEM and "inferred negatives" in SUMMARIZE_SYSTEM


@pytest.mark.parametrize(
    "description, facts, ok",
    [
        ("One outlet in the kitchen sparked when a toaster was plugged into it. The issue is limited to that single outlet.",
         ["One of the kitchen outlets sparked when the toaster was plugged in"], False),
        ("Water began coming into the basement last night after a storm and is still coming in.",
         ["Water started coming into the basement last night after the storm", "Water is still coming into the basement"], True),
        ("Water is coming into the basement. This is an emergency situation that needs prompt attention.",
         ["Water is coming into the basement"], False),
        ("Only the kitchen outlet sparked.", ["Only the kitchen outlet sparked"], True),  # user said "only"
    ],
)
def test_faithfulness_check(description, facts, ok):
    assert (unsupported_claims(description, facts) == []) == ok


# --- live-run review fixes (round 2) ---------------------------------------------------


def test_faithfulness_detail_keys_and_values_are_supported():
    facts = ["Water started coming into the basement last night after the storm"]
    assert unsupported_claims("The source is the storm.", facts, {"source": "storm"}) == []


def test_faithfulness_flags_ownership_words_only_when_unknown():
    facts = ["Thinks a raccoon is living in the attic"]
    assert unsupported_claims("The homeowner thinks a raccoon is living in the attic.", facts)
    assert unsupported_claims("The homeowner thinks a raccoon is living in the attic.", facts, owner_or_renter="owner") == []
    assert unsupported_claims("The customer thinks a raccoon is living in the attic.", facts) == []


def test_summary_prompt_and_input():
    import json as _json

    from agent.prompts import summarize_input

    assert '"the customer"' in SUMMARIZE_SYSTEM and "homeowner" in SUMMARIZE_SYSTEM
    state = LeadState(category="plumbing", facts=["Stain on ceiling"], category_details={"fixture": "unknown", "symptom": "stain"})
    data = _json.loads(summarize_input(state))
    assert data["details"] == {"symptom": "stain"} and "owner_or_renter" not in data
    state.owner_or_renter = "renter"
    assert _json.loads(summarize_input(state))["owner_or_renter"] == "renter"


def test_extractor_sees_unanswered_detail_keys():
    import json as _json

    from agent.prompts import known_state_json

    state = LeadState(category="roofing", category_confidence=0.9, category_details={"active_leak": "no"})
    unanswered = _json.loads(known_state_json(state))["unanswered_detail_keys"]
    assert set(unanswered) == {"cause", "roof_type"}


def test_unprompted_not_sure_also_stops_detail_questions():
    state = LeadState(category="plumbing", category_confidence=0.9, facts=["Drain slow"], urgency="flexible",
                      category_details={"symptom": "unknown"})
    assert nodes.next_target(state) == "zip"


def test_max_two_detail_questions():
    state = LeadState(category="water_damage", category_confidence=0.9, facts=["Wet floor"], urgency="flexible",
                      asked=["detail:still_active", "detail:extent"], category_details={"still_active": "no", "extent": "small"})
    assert nodes.next_target(state) == "zip"


@pytest.mark.parametrize(
    "flags, extracted_urgency, expected",
    [
        (["sparks_or_smoke"], None, "within_48h"),          # single spark, nothing else said
        (["sparks_or_smoke"], "flexible", "within_48h"),    # never below the floor
        (["sparks_or_smoke"], "emergency", "emergency"),    # extractor saw ongoing sparking/heat
        (["burning_smell"], "within_48h", "emergency"),     # burning smell = ongoing hazard
    ],
)
def test_electrical_hazard_urgency(flags, extracted_urgency, expected):
    state = LeadState(category="electrical", category_confidence=0.9, facts=["Outlet sparked"],
                      safety_flags=flags, urgency=extracted_urgency)
    assert nodes.route(state).get("urgency", state.urgency) == expected


# --- scenario-suite fixes ------------------------------------------------------------


def test_out_of_region_zip_ends_immediately_without_asking_contact(tmp_path):
    from agent.categories import OUT_OF_AREA_MESSAGE

    llm = FakeLLM(extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Water heater leaking"],
                                "zip": "90012"}])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "My water heater is leaking. I'm in 90012.", tmp_path)
    assert s.status == "out_of_scope" and s.out_of_scope_reason == "out_of_area"
    assert s.messages[-1].content == OUT_OF_AREA_MESSAGE.format(zip="90012")
    assert "contact" not in s.asked and s.lead_id is None
    llm.texts.append("Sorry I couldn't help this time.")
    assert turn(graph, tid, "oh ok", tmp_path).messages[-1].content == "Sorry I couldn't help this time."


def test_nonexistent_zip_is_dropped_and_asked_again(tmp_path):
    ready = {k: v for k, v in READY.items() if k not in ("zip", "name")}
    llm = FakeLLM(extractions=[{**ready, "zip": "00000"}], texts=["What's your zip code?"])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "sink dripping, zip 00000", tmp_path)
    assert s.zip is None and s.last_route == "ask_next:zip" and s.status == "in_progress"


def test_zip_given_first_is_never_asked(tmp_path):
    llm = FakeLLM(
        extractions=[
            {"category": "plumbing", "category_confidence": 0.9, "zip": "95618",
             "new_facts": ["Toilet won't stop running"], "category_details": {"fixture": "toilet"}},
            {"urgency": "within_week"},
            {"category_details": {"symptom": "running"}},
        ],
        texts=["How soon?", "Is it leaking, clogged, or no water?", "Name and number?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    for msg in ["95618, my toilet won't stop running", "sometime this week", "it just keeps running"]:
        s = turn(graph, tid, msg, tmp_path)
    assert s.zip == "95618" and "zip" not in s.asked and s.last_route == "ask_next:contact"


def test_refuses_phone_but_gives_email_converts(tmp_path):
    ready = {k: v for k, v in READY.items() if k != "name"}
    llm = FakeLLM(
        extractions=[ready, {"name": "Chris", "declined_contact": ["phone"], "contact_email": "chris@example.com"}],
        texts=["Name and number?", "Sink dripping."],
        consents=["yes"],
    )
    graph, tid = make_graph(llm, tmp_path)
    turn(graph, tid, "sink dripping, 95616", tmp_path)
    s = turn(graph, tid, "I'm Chris. No phone please, email me at chris@example.com", tmp_path)
    assert s.status == "awaiting_confirm" and "chris@example.com" in s.messages[-1].content
    s = turn(graph, tid, "yes", tmp_path)
    assert s.status == "converted" and s.contact_phone is None and s.contact_email == "chris@example.com"


def test_plumbing_vs_water_damage_rule_is_wired_everywhere():
    """The rule is prompt-level (the LLM applies it), so test that the single
    source of truth reaches every place that decides or asks about the category."""
    from agent.categories import CATEGORIES, PLUMBING_VS_WATER_DAMAGE
    from agent.prompts import clarify_instruction, extraction_system_prompt

    assert CATEGORIES["plumbing"].confusion_note == PLUMBING_VS_WATER_DAMAGE
    assert CATEGORIES["water_damage"].confusion_note == PLUMBING_VS_WATER_DAMAGE
    for phrase in ["NO pooled/standing water", "soaked", "emergency with standing water"]:
        assert phrase in PLUMBING_VS_WATER_DAMAGE
    prompt = extraction_system_prompt()
    assert "## Category rules" in prompt and "no water on the floor" in prompt
    for cat in ["plumbing", "water_damage"]:
        state = LeadState(category=cat, category_confidence=0.5)
        assert PLUMBING_VS_WATER_DAMAGE in clarify_instruction(state)


# --- ongoing electrical hazards ------------------------------------------------------


@pytest.mark.parametrize(
    "text, ongoing",
    [
        ("The outlet keeps sparking every time something is plugged in", True),
        ("it's still sparking", True),
        ("it sparks whenever I plug in the vacuum", True),
        ("it won't stop sparking", True),
        ("there's smoke coming out of the outlet", True),
        ("the cover plate feels warm", True),
        ("the plug gets hot", True),
        ("there are scorch marks on the outlet", True),
        ("the outlet looks melted", True),
        ("It sparked once, outlet isn't hot", False),
        ("no smoke, just one spark", False),
        ("one of the outlets sparked", False),
        ("my smoke detector keeps beeping", False),
        ("the hot water heater is leaking", False),
        ("the outlet is not warm and there's no smoke", False),
    ],
)
def test_ongoing_electrical_hazard_keywords(text, ongoing):
    from agent.safety import ongoing_electrical_hazard

    assert ongoing_electrical_hazard(text) is ongoing
    assert ("ongoing_electrical" in detect_hazards(text)) is ongoing


@pytest.mark.parametrize(
    "message, extracted_urgency, expected",
    [
        ("The outlet keeps sparking every time something is plugged in", "within_48h", "emergency"),
        ("The outlet keeps sparking every time something is plugged in", "within_week", "emergency"),
        ("It sparked once, outlet isn't hot", "within_48h", "within_48h"),
        ("no smoke, just one spark", "within_48h", "within_48h"),
    ],
)
def test_ongoing_sparks_force_emergency_in_code(tmp_path, message, extracted_urgency, expected):
    """Whatever the extractor says, an ongoing hazard is an emergency; a single spark isn't."""
    llm = FakeLLM(extractions=[{**ELECTRICAL, "urgency": extracted_urgency}], texts=["Is it just that one outlet?"])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, message, tmp_path)
    assert s.urgency == expected


def test_only_one_electrical_notice(tmp_path):
    from agent.categories import SAFETY_NOTICES as NOTICES

    llm = FakeLLM(extractions=[{**ELECTRICAL, "urgency": "emergency"}, {}],
                  texts=["Is it just that one outlet?", "What's your zip?"])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "the outlet keeps sparking and the plate is hot", tmp_path)
    assert s.messages[-1].content.count("Safety first") == 1
    assert s.messages[-1].content.startswith(NOTICES["sparks_or_smoke"])
    s = turn(graph, tid, "now there's smoke too", tmp_path)
    assert "Safety first" not in s.messages[-1].content  # already warned
