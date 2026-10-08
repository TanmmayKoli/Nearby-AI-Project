"""Offline graph tests with a scripted fake LLM (no API calls)."""

import json

import anthropic
import httpx

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agent import nodes
from agent.graph import build_graph, get_state, new_thread_id, run_turn
from agent.nodes import ConsentReply, next_target, salvage
from agent.state import ExtractedFields, LeadState

PROVIDER = {
    "name": "Test Restoration Co", "city": "Davis", "rating": 4.8, "review_count": 120,
    "distance_miles": 2.0, "phone": "(530) 555-0199", "category": "water_damage",
}


class Invalid:
    """Script item: the model returned these args but they fail validation."""

    def __init__(self, args: dict):
        self.args = args


class FakeStructured:
    def __init__(self, fake: "FakeLLM", schema, include_raw: bool):
        self.fake, self.schema, self.include_raw = fake, schema, include_raw

    def invoke(self, messages):
        if self.schema is ConsentReply:
            item = self.fake.consents.pop(0)
            if isinstance(item, Exception):
                raise item
            return ConsentReply(decision=item)
        item = self.fake.extractions.pop(0)
        if isinstance(item, Exception):
            raise item
        args = item.args if isinstance(item, Invalid) else item
        raw = AIMessage(content="", tool_calls=[{"name": "ExtractedFields", "args": args, "id": "t1"}])
        try:
            return {"raw": raw, "parsed": ExtractedFields.model_validate(args), "parsing_error": None}
        except ValidationError as e:
            return {"raw": raw, "parsed": None, "parsing_error": e}


class FakeLLM:
    def __init__(self, extractions=(), texts=(), consents=()):
        self.extractions, self.texts, self.consents = list(extractions), list(texts), list(consents)
        self.prompts: list[str] = []

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        return FakeStructured(self, schema, include_raw)

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        item = self.texts.pop(0)
        if isinstance(item, Exception):
            raise item
        return AIMessage(item)


def make_graph(llm, tmp_path, matches=(PROVIDER,)):
    graph = build_graph(llm=llm, leads_dir=tmp_path / "leads", matcher=lambda c, z, urgency=None: list(matches))
    return graph, new_thread_id()


def turn(graph, tid, text, tmp_path):
    return run_turn(graph, tid, text, transcripts_dir=tmp_path / "transcripts")


def test_basement_happy_path_writes_lead_and_transcript(tmp_path):
    llm = FakeLLM(
        extractions=[
            {"category": "water_damage", "category_confidence": 0.95, "urgency": "emergency",
             "new_facts": ["Water coming into basement since last night's storm"],
             "category_details": {"source": "storm"}},
            {"new_facts": ["About an inch of water across the basement"],
             "category_details": {"still_active": "yes", "extent": "about an inch, whole basement"}},
            {"zip": "95616"},
            {"name": "Jane Doe", "contact_phone": "530 555 0123"},
        ],
        texts=[
            "Is water still coming in right now?",
            "What's your zip code?",
            "What name and phone number should the pro use?",
            "Storm water entering the basement since last night, about an inch deep and still coming in.",
        ],
        consents=["yes"],
    )
    graph, tid = make_graph(llm, tmp_path)

    s = turn(graph, tid, "Water started coming into my basement last night after the storm.", tmp_path)
    assert s.last_route == "ask_next:detail:still_active" and s.status == "in_progress"
    s = turn(graph, tid, "Yes, still coming in. About an inch everywhere.", tmp_path)
    assert s.last_route == "ask_next:zip"  # 3 details known -> no more detail questions
    s = turn(graph, tid, "95616", tmp_path)
    assert s.last_route == "ask_next:contact"
    s = turn(graph, tid, "Jane Doe, 530 555 0123", tmp_path)
    assert s.status == "awaiting_confirm"
    assert "Test Restoration Co" in s.messages[-1].content and "Is it OK to share" in s.messages[-1].content
    s = turn(graph, tid, "yes go ahead", tmp_path)

    assert s.status == "converted" and s.lead_id
    lead = json.loads((tmp_path / "leads" / f"{s.lead_id}.json").read_text())
    assert lead["contact_phone"] == "(530) 555-0123"
    assert lead["facts"] == ["Water coming into basement since last night's storm", "About an inch of water across the basement"]
    assert lead["problem_description"].startswith("Storm water")
    assert lead["consent_to_share"] is True and lead["matched_providers"][0]["name"] == "Test Restoration Co"

    transcript = json.loads((tmp_path / "transcripts" / f"{tid}.json").read_text())
    assert transcript["state"]["status"] == "converted"
    assert transcript["messages"][0]["role"] == "user" and len(transcript["messages"]) == 10

    # After the end: a short LLM reply; the lead and state are never touched.
    llm.texts.append("You're welcome! Hope the basement dries out soon.")
    s = turn(graph, tid, "thanks!", tmp_path)
    assert s.messages[-1].content == "You're welcome! Hope the basement dries out soon."
    assert s.status == "converted" and json.loads((tmp_path / "leads" / f"{s.lead_id}.json").read_text()) == lead


def test_retry_once_then_success_is_logged(tmp_path):
    llm = FakeLLM(
        extractions=[Invalid({"category": "plumbing"}), {"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}],
        texts=["How soon do you need someone?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "my sink is leaking", tmp_path)
    assert s.category == "plumbing" and len(s.extraction_errors) == 1
    assert s.last_route == "ask_next:urgency"


def test_second_failure_drops_only_invalid_fields(tmp_path):
    bad = {"category": "plumbing", "zip": "95616", "new_facts": ["Something leaking"]}  # no confidence
    llm = FakeLLM(extractions=[Invalid(bad), Invalid(bad)], texts=["What kind of help do you need?"])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "something's leaking, 95616", tmp_path)
    assert s.category is None and s.zip == "95616" and s.facts == ["Something leaking"]
    assert len(s.extraction_errors) == 2
    assert s.last_route == "ask_next:category"


def timeout() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


FULL_EMERGENCY = {"category": "plumbing", "category_confidence": 0.9, "urgency": "emergency", "new_facts": ["Pipe burst"],
                  "category_details": {"fixture": "pipe", "symptom": "leaking"}, "zip": "95616",
                  "name": "Jane", "contact_phone": "5305550123"}
FULL = {"category": "plumbing", "category_confidence": 0.9, "urgency": "flexible", "new_facts": ["Sink leaking"],
        "category_details": {"fixture": "sink", "symptom": "leaking"}, "zip": "95616",
        "name": "Jane", "contact_phone": "5305550123"}


def test_extract_timeout_apologizes_and_leaves_state_unchanged(tmp_path):
    llm = FakeLLM(
        extractions=[timeout(), {"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}],
        texts=["How soon do you need someone?"],  # only used on the second turn
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "my sink is leaking", tmp_path)
    assert s.messages[-1].content == nodes.LLM_ERROR_REPLY
    assert s.last_route == "llm_error:extract" and len(s.llm_errors) == 1 and "APITimeoutError" in s.llm_errors[0]
    assert s.facts == [] and s.turn_count == 0 and s.status == "in_progress" and s.asked == []

    transcript = json.loads((tmp_path / "transcripts" / f"{tid}.json").read_text())
    assert transcript["state"]["llm_errors"] == s.llm_errors
    assert transcript["messages"][-1]["content"] == nodes.LLM_ERROR_REPLY

    s = turn(graph, tid, "my sink is leaking", tmp_path)  # stale llm_error route must not block this turn
    assert s.last_route == "ask_next:urgency" and s.facts == ["Sink leaking"] and s.turn_count == 1


def test_ask_next_failure_keeps_extracted_fields_and_reasks_next_turn(tmp_path):
    llm = FakeLLM(
        extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}, {}],
        texts=[anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "How soon do you need someone?"],
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "my sink is leaking", tmp_path)
    assert s.messages[-1].content == nodes.LLM_ERROR_REPLY and s.last_route == "llm_error:ask_next"
    assert s.facts == ["Sink leaking"] and s.asked == []  # question not marked as asked
    s = turn(graph, tid, "my sink is leaking", tmp_path)
    assert s.asked == ["urgency"] and s.messages[-1].content == "How soon do you need someone?"


def test_summarize_failure_skips_confirm_then_recovers(tmp_path):
    llm = FakeLLM(extractions=[FULL, {}], texts=[timeout(), "Sink leaking."], consents=["yes"])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "everything at once", tmp_path)
    assert s.last_route == "llm_error:summarize" and s.status == "in_progress" and s.problem_description is None
    s = turn(graph, tid, "hello?", tmp_path)
    assert s.status == "awaiting_confirm"
    assert turn(graph, tid, "yes", tmp_path).status == "converted"


def test_consent_failure_stays_awaiting_confirm(tmp_path):
    llm = FakeLLM(extractions=[FULL], texts=["Sink leaking."], consents=[timeout(), "yes"])
    graph, tid = make_graph(llm, tmp_path)
    turn(graph, tid, "everything at once", tmp_path)
    s = turn(graph, tid, "yes", tmp_path)
    assert s.status == "awaiting_confirm" and not s.consent_to_share
    assert s.messages[-1].content == nodes.LLM_ERROR_REPLY and len(s.llm_errors) == 1
    assert turn(graph, tid, "yes", tmp_path).status == "converted"


def test_code_bugs_still_raise(tmp_path):
    """Only LLM/API errors are swallowed; a bug in our own code must not be hidden."""
    import pytest

    llm = FakeLLM(extractions=[KeyError("bug")])
    graph, tid = make_graph(llm, tmp_path)
    with pytest.raises(KeyError):
        turn(graph, tid, "hi", tmp_path)


def test_get_llm_uses_configured_timeout_and_retries(monkeypatch):
    from agent import config
    from agent.graph import get_llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    llm = get_llm()
    assert llm.default_request_timeout == config.LLM_TIMEOUT_SECONDS == 45
    assert llm.max_retries == config.LLM_MAX_RETRIES == 2


def test_salvage_drops_bad_field_types():
    args = {"zip": "95616", "urgency": "asap", "new_facts": ["x"]}
    try:
        ExtractedFields.model_validate(args)
    except ValidationError as e:
        out = salvage(args, e)
    assert out.urgency is None and out.zip == "95616" and out.new_facts == ["x"]


def test_detail_questions_capped_when_user_dodges():
    state = LeadState(
        category="plumbing", category_confidence=0.9, facts=["Leak"], urgency="flexible",
        asked=["detail:fixture", "detail:symptom"],
    )
    assert next_target(state) == "zip"


def test_contact_comes_last():
    state = LeadState(
        category="plumbing", category_confidence=0.9, facts=["Leak"], urgency="flexible",
        category_details={"fixture": "sink", "symptom": "leaking"}, name="Jane",
    )
    assert next_target(state) == "zip"
    state.zip = "95616"
    assert next_target(state) == "contact"
    state.contact_email = "j@example.com"
    assert next_target(state) is None


def test_consent_changes_goes_back_through_extract(tmp_path):
    base = {"category": "plumbing", "category_confidence": 0.9, "urgency": "flexible", "new_facts": ["Sink leaking"],
            "category_details": {"fixture": "sink", "symptom": "leaking"}, "zip": "95616",
            "name": "Jane", "contact_phone": "5305550123"}
    llm = FakeLLM(
        extractions=[base, {"zip": "95618"}],
        texts=["Sink leaking.", "Sink leaking."],
        consents=["changes", "yes"],
    )
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "everything at once", tmp_path)
    assert s.status == "awaiting_confirm" and "95616" in s.messages[-1].content
    s = turn(graph, tid, "actually my zip is 95618", tmp_path)
    assert s.status == "awaiting_confirm" and s.zip == "95618" and "95618" in s.messages[-1].content
    s = turn(graph, tid, "yes", tmp_path)
    assert s.status == "converted"


def test_consent_no_creates_no_lead(tmp_path):
    base = {"category": "plumbing", "category_confidence": 0.9, "urgency": "flexible", "new_facts": ["Sink leaking"],
            "category_details": {"fixture": "sink", "symptom": "leaking"}, "zip": "95616",
            "name": "Jane", "contact_phone": "5305550123"}
    llm = FakeLLM(extractions=[base], texts=["Sink leaking."], consents=["no"])
    graph, tid = make_graph(llm, tmp_path)
    turn(graph, tid, "everything at once", tmp_path)
    s = turn(graph, tid, "no thanks", tmp_path)
    assert s.status == "declined" and s.lead_id is None
    assert not (tmp_path / "leads").exists()


def test_no_providers_ends_without_confirm(tmp_path):
    base = {"category": "plumbing", "category_confidence": 0.9, "urgency": "flexible", "new_facts": ["Leak"],
            "category_details": {"fixture": "sink", "symptom": "leaking"}, "zip": "95616",
            "name": "Jane", "contact_phone": "5305550123"}
    graph, tid = make_graph(FakeLLM(extractions=[base]), tmp_path, matches=())
    s = turn(graph, tid, "everything", tmp_path)
    assert s.last_route == "no_providers" and s.status == "in_progress"


def test_prompts_built_from_category_data():
    from agent.prompts import extraction_system_prompt

    text = extraction_system_prompt()
    for key in ["water_damage", "still_active", "utility_outage", "water_near_electrical"]:
        assert key in text


def test_raw_args_reads_tool_calls_or_json_text():
    from agent.nodes import _raw_args

    assert _raw_args(AIMessage(content="", tool_calls=[{"name": "x", "args": {"zip": "1"}, "id": "1"}])) == {"zip": "1"}
    assert _raw_args(AIMessage(content='{"zip": "95616"}')) == {"zip": "95616"}
    assert _raw_args(AIMessage(content="not json")) is None


def _ready_state(**overrides) -> LeadState:
    base = dict(
        category="water_damage", category_confidence=0.95, facts=["Water in basement"], urgency="emergency",
        category_details={"still_active": "yes", "source": "storm"}, zip="95616",
    )
    return LeadState(**{**base, **overrides})


def test_emergency_contact_question_offers_optional_address_once():
    from agent.prompts import EMERGENCY_ADDRESS_INSTRUCTION, ask_system_prompt

    state = _ready_state()
    assert next_target(state) == "contact"
    assert EMERGENCY_ADDRESS_INSTRUCTION in ask_system_prompt(state, "contact", already_asked=False)
    # user skipped the address and still owes contact: don't offer it again
    assert EMERGENCY_ADDRESS_INSTRUCTION not in ask_system_prompt(state, "contact", already_asked=True)
    # not an emergency, or address already given: never offered
    assert EMERGENCY_ADDRESS_INSTRUCTION not in ask_system_prompt(_ready_state(urgency="within_48h"), "contact", False)
    assert EMERGENCY_ADDRESS_INSTRUCTION not in ask_system_prompt(_ready_state(address="1 A St"), "contact", False)


def test_skipped_address_does_not_block_the_lead():
    state = _ready_state(name="Jane", contact_phone="(530) 555-0123")
    assert state.address is None and next_target(state) is None


def test_final_message_lists_phones_only_for_emergencies():
    providers = [{**PROVIDER, "phone": "(530) 555-0199"}, {**PROVIDER, "name": "Second Co", "phone": "(916) 555-0100"}]
    emergency = nodes.final_message(_ready_state(matched_providers=providers), "abc123")
    assert "call any of them now" in emergency
    assert "Test Restoration Co: (530) 555-0199" in emergency and "Second Co: (916) 555-0100" in emergency
    routine = nodes.final_message(_ready_state(urgency="within_week", matched_providers=providers), "abc123")
    assert "call any of them now" not in routine and "555-0199" not in routine


def test_confirm_shows_address_when_given():
    state = _ready_state(name="Jane", contact_phone="(530) 555-0123", address="1 A St",
                         problem_description="Water in basement.", matched_providers=[PROVIDER])
    assert "Address: 1 A St" in nodes.confirm_message(state)
    state.address = None
    assert "Address:" not in nodes.confirm_message(state)


def test_matcher_receives_urgency(tmp_path):
    seen = {}

    def matcher(category, zip_code, urgency=None):
        seen["urgency"] = urgency
        return [PROVIDER]

    graph = build_graph(llm=FakeLLM(extractions=[{**FULL_EMERGENCY}], texts=["Sink leaking."]), leads_dir=tmp_path, matcher=matcher)
    run_turn(graph, new_thread_id(), "everything", transcripts_dir=None)
    assert seen["urgency"] == "emergency"


def test_summary_input_excludes_urgency_and_prompt_forbids_commentary():
    from agent.prompts import SUMMARIZE_SYSTEM, summarize_input

    assert "urgency" not in json.loads(summarize_input(_ready_state()))
    assert "commentary" in SUMMARIZE_SYSTEM and "urgency is a separate field" in SUMMARIZE_SYSTEM


# --- streaming ---------------------------------------------------------------------

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402


class StreamingFake(GenericFakeChatModel):
    """A real LangChain chat model for free text (so LangGraph can stream its
    tokens) with scripted structured output from FakeLLM."""

    structured: object = None

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        return self.structured.with_structured_output(schema, include_raw=include_raw)


def streaming_llm(texts, **scripted) -> StreamingFake:
    return StreamingFake(messages=iter([AIMessage(t) for t in texts]), structured=FakeLLM(**scripted))


def test_stream_turn_yields_tokens_with_spaces_and_saves_transcript(tmp_path):
    from agent.graph import stream_turn

    llm = streaming_llm(
        ["How soon do you need someone?"],
        extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}],
    )
    graph, tid = make_graph(llm, tmp_path)
    streamed = "".join(stream_turn(graph, tid, "my sink is leaking", transcripts_dir=tmp_path / "t"))
    state = get_state(graph, tid)
    assert streamed == "How soon do you need someone?" == state.messages[-1].content
    assert state.last_route == "ask_next:urgency"
    assert (tmp_path / "t" / f"{tid}.json").exists()


def test_stream_turn_sends_safety_notice_first(tmp_path):
    from agent.categories import SAFETY_NOTICES
    from agent.graph import stream_turn

    llm = streaming_llm(
        ["Is it just that one outlet?"],
        extractions=[{"category": "electrical", "category_confidence": 0.9, "new_facts": ["Outlet sparked"]}],
    )
    graph, tid = make_graph(llm, tmp_path)
    streamed = "".join(stream_turn(graph, tid, "my outlet sparked", transcripts_dir=None))
    assert streamed.startswith(SAFETY_NOTICES["sparks_or_smoke"])
    assert streamed == get_state(graph, tid).messages[-1].content  # identical to what's stored


def test_stream_turn_streams_nothing_for_fixed_replies(tmp_path):
    from agent.graph import stream_turn

    graph, tid = make_graph(streaming_llm([]), tmp_path)
    assert "".join(stream_turn(graph, tid, "I smell gas", transcripts_dir=None)) == ""
    assert get_state(graph, tid).status == "emergency"  # shown from state after the turn


# --- extract model split + timing ---------------------------------------------------


def test_extract_uses_extract_llm_and_replies_use_main_llm(tmp_path):
    """Only the extract node talks to extract_llm; questions come from llm."""
    extractor = FakeLLM(extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}])
    llm = FakeLLM(texts=["How soon do you need someone?"])  # no extractions: extract must not use it
    graph = build_graph(llm=llm, extract_llm=extractor, leads_dir=tmp_path / "leads",
                        matcher=lambda c, z, urgency=None: [PROVIDER])
    tid = new_thread_id()
    state = turn(graph, tid, "my sink is leaking", tmp_path)
    assert state.messages[-1].content == "How soon do you need someone?"
    assert extractor.extractions == [] and extractor.prompts == []  # extracted once, wrote no replies
    assert len(llm.prompts) == 1


def test_build_graph_defaults_to_extract_model_for_extraction(monkeypatch):
    import agent.graph as graph_module
    from agent.config import EXTRACT_MODEL

    models = []
    monkeypatch.setattr(graph_module, "get_llm", lambda model="AGENT_MODEL": models.append(model) or FakeLLM())
    build_graph()
    assert models == [EXTRACT_MODEL, "AGENT_MODEL"]


def test_final_message_demo_says_not_sent_and_keeps_phones():
    providers = [{**PROVIDER, "phone": "(530) 555-0199"}, {**PROVIDER, "name": "Second Co", "phone": "(916) 555-0100"}]
    for urgency in ("emergency", "within_week"):
        msg = nodes.final_message(_ready_state(urgency=urgency, matched_providers=providers), "abc123", demo=True)
        assert "is ready to go to" not in msg and "demo" in msg and "won't contact you" in msg
        assert "Test Restoration Co: (530) 555-0199" in msg and "Second Co: (916) 555-0100" in msg


def test_node_timer_records_each_node(tmp_path):
    from agent.timing import NodeTimer

    llm = FakeLLM(extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}],
                  texts=["How soon do you need someone?"])
    graph, tid = make_graph(llm, tmp_path)
    timer = NodeTimer()
    run_turn(graph, tid, "my sink is leaking", transcripts_dir=None, callbacks=[timer])
    times = timer.totals()
    assert set(times) == {"safety_check", "extract", "route", "ask_next"}
    assert all(len(v) == 1 and v[0] >= 0 for v in times.values())


def test_turn_config_tags_trace_with_thread_id(tmp_path):
    """LangSmith groups traces by metadata thread_id; source goes in tags."""
    from langchain_core.callbacks import BaseCallbackHandler

    class RootRun(BaseCallbackHandler):
        def __init__(self):
            self.root = None

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kw):
            if parent_run_id is None:
                self.root = {"name": kw.get("name"), "tags": tags, "metadata": metadata}

    llm = FakeLLM(extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}],
                  texts=["How soon do you need someone?"])
    graph, tid = make_graph(llm, tmp_path)
    handler = RootRun()
    run_turn(graph, tid, "my sink is leaking", transcripts_dir=None, source="test", callbacks=[handler])
    assert handler.root["name"] == "turn"
    assert handler.root["metadata"]["thread_id"] == tid and "test" in handler.root["tags"]
