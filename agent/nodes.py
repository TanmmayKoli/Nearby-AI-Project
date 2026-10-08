"""Graph nodes. One function per node; each takes the state and returns updates.

LLM-backed nodes take the chat model as a second argument (bound in graph.py),
so tests can pass a scripted fake. Everything that decides *what happens next*
is plain Python: `entry_route`, `after_safety`, `route`, `next_target`.
"""

import json
from dataclasses import dataclass
import re
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import anthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from agent import prompts
from agent.categories import (
    CATEGORIES,
    EMERGENCY_SAFETY_FLAGS,
    GAS_EMERGENCY_MESSAGE,
    OUT_OF_AREA_MESSAGE,
    OUT_OF_SCOPE,
    PGE_EMERGENCY_PHONE,
    SAFETY_NOTICES,
    URGENT_SAFETY_FLAGS,
)
from agent.safety import ELECTRICAL_NOTICE_ORDER, detect_hazards, safety_notice
from providers.match import zip_service_status
from agent.state import (
    ExtractedFields,
    Lead,
    LeadState,
    category_is_confident,
    merge_extracted,
    missing_required_fields,
)

log = logging.getLogger(__name__)

# json_schema (native structured output), not forced tool calling: newer models
# like claude-sonnet-5-5 don't support forced tool calls.
STRUCTURED_METHOD = "json_schema"
MAX_DETAIL_QUESTIONS = 2  # ask at most this many category detail questions
MAX_CONTACT_ASKS = 2  # after this many unanswered contact questions, explain once, then stop

# Errors that mean "the LLM call failed" (after the SDK's own retries), as
# opposed to bugs in our code, which should still raise.
LLM_ERRORS = (anthropic.APIError, OutputParserException, TimeoutError, ConnectionError)
LLM_ERROR_REPLY = "Sorry, I had trouble on my end. Could you say that again?"
CLOSED_STATUSES = {"converted", "abandoned", "emergency", "out_of_scope", "declined"}


def content_text(content: Any) -> str:
    """The user-visible text in a message's content. Not stripped.

    Anthropic content can be a str, a list of blocks ({"type": "text"},
    {"type": "thinking"}, ...), or, after LangChain merges streamed chunks, a
    MIXED list like ['', {thinking block}, 'How soon...'] where text arrives as
    plain strings. Keep plain strings and text blocks; drop everything else
    (thinking, tool calls). Missing the plain strings is what made streamed
    replies come out empty.
    """
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts)


def _text(message: Any) -> str:
    """Stripped text of a message (or of raw content)."""
    return content_text(getattr(message, "content", message)).strip()


def _raw_args(raw: Any) -> dict | None:
    """The model's raw output as a dict (tool-call args or JSON text), for salvage."""
    tool_calls = getattr(raw, "tool_calls", None) or []
    if tool_calls:
        return tool_calls[0]["args"]
    try:
        parsed = json.loads(_text(raw)) if raw is not None else None
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _reply(state: LeadState, text: str) -> dict[str, Any]:
    """The agent's message for this turn, with any pending safety notice first."""
    if state.pending_notice:
        text = f"{state.pending_notice}\n\n{text}"
    return {"messages": [AIMessage(text)], "pending_notice": None}


def llm_failure(state: LeadState, node: str, error: Exception) -> dict[str, Any]:
    """The only updates a node makes when its LLM call fails: apologize and log.

    Everything else in state is left as it was, so the user can just repeat
    themselves. (Updates from EARLIER nodes in the same turn are kept, e.g. if
    extract succeeded and ask_next timed out, the extracted facts stay.) A
    pending safety notice is still shown.
    """
    entry = f"turn {state.turn_count + 1} {node}: {type(error).__name__}: {error}"
    log.error("LLM call failed: %s", entry)
    return {
        **_reply(state, LLM_ERROR_REPLY),
        "llm_errors": state.llm_errors + [entry],
        "last_route": f"llm_error:{node}",
    }


def after_llm_node(next_node: str) -> Callable[[LeadState], str]:
    """Conditional edge: continue to next_node unless this node's LLM call failed."""

    def decide(state: LeadState) -> str:
        return "end" if (state.last_route or "").startswith("llm_error") else next_node

    return decide


# --- entry + safety ---------------------------------------------------------------


def entry_route(state: LeadState) -> str:
    """Where a new user message goes. Plain code."""
    return "after_end" if state.status in CLOSED_STATUSES else "safety_check"


def _last_user_text(state: LeadState) -> str:
    return next((_text(m) for m in reversed(state.messages) if m.type == "human"), "")


def safety_check(state: LeadState) -> dict[str, Any]:
    """Keyword safety gate on the latest user message, before any LLM call.

    Adds detected flags to state. A gas smell goes straight to
    emergency_response; other hazards get a notice from `route` after extract.
    """
    found = [f for f in detect_hazards(_last_user_text(state)) if f not in state.safety_flags]
    return {"safety_flags": state.safety_flags + found} if found else {}


def after_safety(state: LeadState) -> str:
    if "gas_smell" in state.safety_flags:
        return "emergency_response"
    if state.status == "awaiting_confirm":
        return "consent_reply"
    return "extract"


def emergency_response(state: LeadState) -> dict[str, Any]:
    """Gas smell: safety instructions only. No lead collection."""
    return {
        "messages": [AIMessage(GAS_EMERGENCY_MESSAGE)],
        "status": "emergency",
        "last_route": "emergency_response",
        "pending_notice": None,
        "warned": state.warned + ["gas_smell"],
    }


# --- extract ---------------------------------------------------------------------


def salvage(args: dict, error: ValidationError) -> ExtractedFields:
    """Drop the fields a ValidationError points at and keep the rest.

    A model-level error (category without confidence) drops category, so route
    treats it as unknown. Repeats until it validates; worst case returns empty.
    """
    args = dict(args)
    for _ in range(len(args) + 1):
        bad = {str(e["loc"][0]) for e in error.errors() if e["loc"]}
        if any(not e["loc"] for e in error.errors()):
            bad |= {"category", "category_confidence"}
        if not bad & args.keys():
            break
        for key in bad:
            args.pop(key, None)
        try:
            return ExtractedFields.model_validate(args)
        except ValidationError as e:
            error = e
    return ExtractedFields()


def extract(state: LeadState, llm: BaseChatModel) -> dict[str, Any]:
    """LLM -> ExtractedFields -> merged into state.

    Failure handling (never ends the conversation):
    1. If the output fails validation, retry ONCE with the validation error in the prompt.
    2. If it fails again, drop just the invalid fields (e.g. category without a
       confidence -> category stays unknown, so route asks about it) and continue.
    Validation failures are logged and appended to state.extraction_errors.

    If the API call itself fails after retries (timeout, API error), the turn
    ends with an apology and state unchanged (see llm_failure).
    """
    structured = llm.with_structured_output(ExtractedFields, include_raw=True, method=STRUCTURED_METHOD)
    errors: list[str] = []
    extracted: ExtractedFields | None = None
    retry_error: str | None = None
    raw_args: dict | None = None

    for attempt in (1, 2):
        system = prompts.extraction_system_prompt() + "\n\n" + prompts.extraction_state_prompt(state, retry_error)
        try:
            result = structured.invoke([SystemMessage(system), *state.messages])
        except LLM_ERRORS as e:
            return llm_failure(state, "extract", e)
        if result.get("parsed") is not None and result.get("parsing_error") is None:
            extracted = result["parsed"]
            break
        retry_error = str(result.get("parsing_error"))
        errors.append(f"turn {state.turn_count + 1} attempt {attempt}: {retry_error}")
        raw_args = _raw_args(result.get("raw"))

    if extracted is None and raw_args is not None:
        try:
            extracted = ExtractedFields.model_validate(raw_args)
        except ValidationError as e:
            extracted = salvage(raw_args, e)
    if extracted is None:
        extracted = ExtractedFields()

    for err in errors:
        log.warning("extraction failure: %s", err)

    updates = merge_extracted(state, extracted)
    updates["turn_count"] = state.turn_count + 1
    updates["last_route"] = "extract"  # clears any llm_error from a previous turn
    if errors:
        updates["extraction_errors"] = state.extraction_errors + errors
    return updates


# --- route -----------------------------------------------------------------------


def _contact_target(state: LeadState) -> str | None:
    """Name + a way to reach them. Phone first; if declined, offer email once;
    if both are declined (or ignored), explain once; then give up (no_contact)."""
    has_contact = bool(state.contact_phone or state.contact_email)
    if has_contact:
        return None if state.name else "contact"
    phone_out = "phone" in state.declined
    email_out = "email" in state.declined or "contact_email" in state.asked
    stuck = state.asked.count("contact") >= MAX_CONTACT_ASKS
    if (phone_out and email_out) or stuck:
        return "no_contact" if "contact_required" in state.asked else "contact_required"
    if phone_out:
        return "contact_email"
    return "contact"


def next_target(state: LeadState) -> str | None:
    """What to ask next, or None if we have enough to match providers.

    Priority: what's wrong -> category (clarify once if ambiguous) -> urgency
    -> at most 2 category details (none after a "not sure"; details the user
    already answered are filled by extract, so never asked) -> zip -> name + contact (last, once the user is
    invested). A question is never repeated except required fields the user
    hasn't given yet (which get rephrased).
    """
    if not state.facts:
        return "facts"
    if state.category is None:
        return "category"
    if not category_is_confident(state):
        return "clarify"  # route locks the category instead if clarify was already asked
    if not state.urgency:
        return "urgency"

    spec = CATEGORIES[state.category]
    known = [q.key for q in spec.detail_questions if q.key in state.category_details]
    asked = [t for t in state.asked if t.startswith("detail:")]
    # After the first "not sure" on any detail, stop asking details for this lead.
    unsure = any(state.category_details.get(q.key) == "unknown" for q in spec.detail_questions)
    if not unsure and len(known) < MAX_DETAIL_QUESTIONS and len(asked) < MAX_DETAIL_QUESTIONS:
        for q in spec.detail_questions:
            if q.about_cause and state.cause_unknown:
                continue  # they already said they don't know; the pro will diagnose
            if q.key not in state.category_details and f"detail:{q.key}" not in state.asked:
                return f"detail:{q.key}"
    for q in spec.detail_questions:  # outside the budget (e.g. wildlife damage), still asked only once
        if q.always_ask and q.key not in state.category_details and f"detail:{q.key}" not in state.asked:
            return f"detail:{q.key}"

    if not state.zip:
        return "zip"
    return _contact_target(state)


def route(state: LeadState) -> dict[str, Any]:
    """Deterministic: decide where this turn goes. No LLM.

    Order: gas (extractor caught it) -> out of scope -> zip outside the service area -> safety notices + urgency
    floor -> lock category after one clarify -> mark dodged detail questions
    "unknown" -> next question, or match providers.
    """
    if "gas_smell" in state.safety_flags:
        return {"last_route": "emergency_response"}
    if state.out_of_scope_reason:
        return {"last_route": "out_of_scope"}

    updates: dict[str, Any] = {}

    # Check the zip as soon as we have it, before asking for anything else: a
    # real zip outside the region ends the conversation; a zip that doesn't
    # exist (typo) is dropped so it gets asked again.
    if state.zip:
        zip_status = zip_service_status(state.zip)
        if zip_status == "out_of_area":
            return {"out_of_scope_reason": "out_of_area", "last_route": "out_of_scope"}
        if zip_status == "unknown":
            updates["zip"] = None

    # Electrical hazards: urgency is at least within_48h. An ONGOING hazard
    # (burning smell, or keywords like "keeps sparking", smoke, a hot outlet,
    # scorch marks: see safety.py) is forced to emergency in code, whatever the
    # extractor said. A single spark that stopped stays within_48h. Any unwarned
    # hazard gets a notice either way.
    hazards = set(state.safety_flags) & URGENT_SAFETY_FLAGS
    if hazards & EMERGENCY_SAFETY_FLAGS and state.urgency != "emergency":
        updates["urgency"] = "emergency"
    elif hazards and state.urgency not in ("emergency", "within_48h"):
        updates["urgency"] = "within_48h"
    warned = list(state.warned)
    unwarned = [f for f in state.safety_flags if f in SAFETY_NOTICES and f not in warned]
    if set(warned) & set(ELECTRICAL_NOTICE_ORDER):  # one electrical notice per conversation
        covered = [f for f in unwarned if f in ELECTRICAL_NOTICE_ORDER]
        unwarned = [f for f in unwarned if f not in covered]
        warned += covered
    if unwarned:
        updates["pending_notice"] = safety_notice(unwarned)
        warned += unwarned
    if warned != state.warned:
        updates["warned"] = warned

    # Never clarify twice: on a second ambiguous answer, go with the best guess.
    if state.category and not category_is_confident(state) and "clarify" in state.asked:
        updates["category_locked"] = state.category

    # A detail question that was asked but not answered ("idk", or ignored) is
    # recorded as unknown so it's never asked again.
    if state.category:
        dodged = {
            q.key: "unknown"
            for q in CATEGORIES[state.category].detail_questions
            if f"detail:{q.key}" in state.asked and q.key not in state.category_details
        }
        if dodged:
            updates["category_details"] = {**state.category_details, **dodged}

    s = state.model_copy(update=updates)
    target = next_target(s)
    if target is None:
        decision = "match_providers"
    elif target in ("clarify", "no_contact"):
        decision = target
    else:
        decision = f"ask_next:{target}"
    return {**updates, "missing_fields": missing_required_fields(s), "last_route": decision}


def after_route(state: LeadState) -> str:
    decision = state.last_route or ""
    if decision.startswith("ask_next"):
        return "ask_next"
    return decision  # emergency_response | out_of_scope | clarify | no_contact | match_providers


# --- asking ----------------------------------------------------------------------


def _ask(state: LeadState, llm: BaseChatModel, target: str, node: str) -> dict[str, Any]:
    """Code picks the question; the LLM words it naturally. One question per turn."""
    system = prompts.ask_system_prompt(state, target, already_asked=target in state.asked)
    try:
        reply = llm.invoke([SystemMessage(system), *state.messages])
    except LLM_ERRORS as e:
        return llm_failure(state, node, e)
    text = _text(reply)
    if not text:  # never store (or show) an empty bubble
        return llm_failure(state, node, ValueError(f"empty reply, content={reply.content!r:.200}"))
    return {**_reply(state, text), "asked": state.asked + [target]}


def ask_next(state: LeadState, llm: BaseChatModel) -> dict[str, Any]:
    return _ask(state, llm, next_target(state), "ask_next")


def clarify(state: LeadState, llm: BaseChatModel) -> dict[str, Any]:
    """Category is ambiguous: ask ONE question offering the 2 likely categories
    (from categories.py confusion data). Only ever asked once; see route."""
    return _ask(state, llm, "clarify", "clarify")


# --- terminal redirects ------------------------------------------------------------


def out_of_scope(state: LeadState) -> dict[str, Any]:
    """Not a home service we cover: redirect and end. Excluded from conversion in evals."""
    if state.out_of_scope_reason == "out_of_area":
        text = OUT_OF_AREA_MESSAGE.format(zip=state.zip)
    else:
        text = OUT_OF_SCOPE[state.out_of_scope_reason]
    return {**_reply(state, text), "status": "out_of_scope"}


def _provider_phone_lines(providers: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {p['name']}: {p['phone']}" for p in providers if p.get("phone"))


def no_contact(state: LeadState, matcher: Callable[..., list[dict[str, Any]]]) -> dict[str, Any]:
    """User won't share any contact method: no lead, but still give them the
    matched pros' numbers so the conversation isn't a dead end."""
    matches = matcher(state.category, state.zip, urgency=state.urgency)
    text = "Understood. I can't send a request without a way for the pro to reach you."
    if matches:
        text += f" You can still contact these pros directly:\n\n{_provider_phone_lines(matches)}"
    return {**_reply(state, text), "status": "declined", "matched_providers": matches}


# --- match + summarize + confirm -------------------------------------------------


def match_providers_node(
    state: LeadState, matcher: Callable[..., list[dict[str, Any]]]
) -> dict[str, Any]:
    """Urgency is passed through so emergencies favor nearby providers."""
    matches = matcher(state.category, state.zip, urgency=state.urgency)
    if matches:
        return {"matched_providers": matches}
    return {
        **_reply(
            state,
            f"I'm sorry, I couldn't find a {CATEGORIES[state.category].label.lower()} pro "
            f"near {state.zip}. We currently cover {prompts.SERVICE_AREA}.",
        ),
        "matched_providers": [],
        "last_route": "no_providers",
    }


def after_match(state: LeadState) -> str:
    return "summarize" if state.matched_providers else "end"


def summarize(state: LeadState, llm: BaseChatModel) -> dict[str, Any]:
    """Generate problem_description from facts, right before confirm (regenerated
    if the user changes something at confirm)."""
    try:
        reply = llm.invoke([SystemMessage(prompts.SUMMARIZE_SYSTEM), HumanMessage(prompts.summarize_input(state))])
    except LLM_ERRORS as e:
        return llm_failure(state, "summarize", e)
    description = _text(reply)
    if not description:
        return llm_failure(state, "summarize", ValueError(f"empty summary, content={reply.content!r:.200}"))
    return {"problem_description": description, "last_route": "summarize"}


PROVIDERS_HEADING = "Pros near you:"  # the UI finds this block to render provider cards

URGENCY_LABELS = {
    "emergency": "Emergency",
    "within_48h": "Within 48 hours",
    "within_week": "Within a week",
    "flexible": "Flexible",
}


def confirm_message(state: LeadState) -> str:
    """Deterministic template, so provider names/numbers can't be hallucinated.

    Markdown: blank lines between blocks so the header, description and contact
    render as separate paragraphs; two trailing spaces = line break.
    """
    contact = ", ".join(v for v in [state.contact_phone, state.contact_email] if v)
    contact_block = f"Contact: {state.name}, {contact}"
    if state.address:
        contact_block += f"  \nAddress: {state.address}"
    providers = "\n".join(
        f"{i}. {p['name']}, {p['city']} · "
        + (f"{p['rating']:.1f}★ ({p['review_count']} reviews)" if p.get("rating") else "no rating")
        + f" · {p['distance_miles']:.1f} mi"
        for i, p in enumerate(state.matched_providers, 1)
    )
    return "\n\n".join(
        [
            "Here's the request I'll send:",
            f"**{CATEGORIES[state.category].label}** · {URGENCY_LABELS[state.urgency]} · {state.zip}",
            state.problem_description,
            contact_block,
            f"{PROVIDERS_HEADING}\n\n{providers}",
            "Is it OK to share your name and contact info with these pros so they can reach you?",
        ]
    )


def confirm(state: LeadState) -> dict[str, Any]:
    """Summary + providers + consent question."""
    return {**_reply(state, confirm_message(state)), "status": "awaiting_confirm"}


class ConsentReply(BaseModel):
    decision: Literal["yes", "no", "changes"]


def consent_reply(state: LeadState, llm: BaseChatModel) -> dict[str, Any]:
    """Classify the reply to the consent question.

    yes -> create_lead. changes ("use my email instead") -> back through
    extract/route, which updates the field and re-shows the summary.
    no -> stop without a lead (status declined).
    """
    try:
        decision = llm.with_structured_output(ConsentReply, method=STRUCTURED_METHOD).invoke(
            [SystemMessage(prompts.CONSENT_SYSTEM), *state.messages[-2:]]
        ).decision
    except LLM_ERRORS as e:
        return llm_failure(state, "consent_reply", e)  # status stays awaiting_confirm

    turn = state.turn_count + 1
    if decision == "yes":
        return {"consent_to_share": True, "last_route": "consent:yes", "turn_count": turn}
    if decision == "changes":
        # extract will count this turn
        return {"status": "in_progress", "last_route": "consent:changes"}
    return {
        "status": "declined",
        "last_route": "consent:no",
        "turn_count": turn,
        "messages": [AIMessage("No problem, I won't share your information. If you change your mind, just start a new conversation.")],
    }


def after_consent(state: LeadState) -> str:
    return {"consent:yes": "create_lead", "consent:changes": "extract"}.get(state.last_route, "end")


# --- create_lead + closed --------------------------------------------------------


def create_lead(state: LeadState, leads_dir: Path | None, demo: bool = False) -> dict[str, Any]:
    """Validate the Lead (schema enforces consent, contact, >=1 provider) and write it.
    leads_dir=None (deployed demo) validates but doesn't write to disk."""
    lead_id = uuid.uuid4().hex[:8]
    lead = Lead(
        lead_id=lead_id,
        category=state.category,
        problem_description=state.problem_description,
        facts=state.facts,
        urgency=state.urgency,
        zip=state.zip,
        name=state.name,
        contact_phone=state.contact_phone,
        contact_email=state.contact_email,
        address=state.address,
        property_type=state.property_type,
        owner_or_renter=state.owner_or_renter,
        availability=state.availability,
        category_details=state.category_details,
        safety_flags=state.safety_flags,
        matched_providers=state.matched_providers,
        consent_to_share=state.consent_to_share,
    )
    if leads_dir is not None:
        leads_dir.mkdir(parents=True, exist_ok=True)
        (leads_dir / f"{lead_id}.json").write_text(json.dumps(lead.model_dump(mode="json"), indent=2))

    repair = repair_to_offer(state)
    return {
        "lead_id": lead_id,
        "status": "converted",
        "repair_offer": repair.damage if repair else None,
        "repair_category": repair.category if repair else None,
        "messages": [AIMessage(final_message(state, lead_id, demo, repair))],
    }


# --- repair offer after a wildlife lead ------------------------------------------
# Plain code: keywords in what the user said -> the trade that repairs it. Rules
# in order; within a rule, keyword patterns in priority order ("vent" before
# "roof", so "torn roof vent" names the vent).

@dataclass(frozen=True)
class RepairOffer:
    category: str  # roofing / electrical / handyman
    who: str  # "a roofer"
    damage: str  # "torn vent"


REPAIR_TRADES: list[tuple[str, str, list[str]]] = [
    ("roofing", "a roofer",
     [r"vents?", r"shingles?", r"soffits?", r"fascia", r"gutters?", r"eaves", r"holes?\s+in\s+(?:the\s+)?roof", r"roof"]),
    ("electrical", "an electrician", [r"wiring", r"wires?", r"cables?"]),
    ("handyman", "a handyman",
     [r"insulation", r"drywall", r"holes?\s+in\s+(?:the\s+)?(?:ceiling|wall)s?", r"ceiling\s+holes?", r"screens?"]),
]
REPAIR_WHO = {category: who for category, who, _ in REPAIR_TRADES}  # "roofing" -> "a roofer"
_DAMAGE_WORD = re.compile(
    r"\b(torn|tore|ripped|chewed|damaged|broken|broke|bent|pulled|loose|soiled|missing|wrecked)"
    r"(?:\s+(?:up|off|out|through|apart|into))?(?:\s+(?:a|an|the|my|some|our|its))?\s+(?:(\w+)\s+)?$",
    re.IGNORECASE,
)
_PARTICIPLE = {"tore": "torn", "broke": "broken"}


def _damage_phrase(text: str, match: re.Match) -> str:
    """'it tore up a vent on the roof' -> 'torn vent'; 'torn roof vent' -> 'torn roof vent'."""
    noun = match.group(0).lower()
    adj = _DAMAGE_WORD.search(text[: match.start()])
    if not adj:
        return noun
    word = _PARTICIPLE.get(adj.group(1).lower(), adj.group(1).lower())
    return " ".join(w for w in (word, (adj.group(2) or "").lower(), noun) if w)


def repair_to_offer(state: LeadState) -> RepairOffer | None:
    """Wildlife leads only. Scans the damage detail, the other details, then the
    facts; the first trade keyword found (not negated) decides. No match -> no offer.
    The damage stays a fact in THIS lead; the repair is a separate request."""
    if state.category != "wildlife_removal":
        return None
    details = state.category_details
    sources = [str(details.get("damage") or "")] + [str(v) for k, v in details.items() if k != "damage"] + state.facts
    for text in sources:
        for category, who, patterns in REPAIR_TRADES:
            for pattern in patterns:
                for m in re.finditer(rf"\b{pattern}\b", text, re.IGNORECASE):
                    if not _NEGATION_BEFORE.search(text[max(0, m.start() - 25): m.start()]):
                        return RepairOffer(category, who, _damage_phrase(text, m))
    return None


_NEGATION_BEFORE = re.compile(r"\b(no|not|don'?t|doesn'?t|didn'?t|never|without|isn'?t|aren'?t|haven'?t)\b[^.!?]*$", re.IGNORECASE)


def final_message(state: LeadState, lead_id: str, demo: bool = False, repair: RepairOffer | None = None) -> str:
    """After consent. Emergencies also get provider phone numbers so the user
    doesn't have to wait for a callback. Demo (deployed): nothing is sent, so
    say so plainly and always list the phone numbers. Wildlife with damage:
    offer a repair request too."""
    names = ", ".join(p["name"] for p in state.matched_providers)
    period = "" if names.endswith(".") else "."  # "Woodland Electrical Inc." -> no ".."
    if demo:
        message = (
            f"You're all set (ref {lead_id}). This is a demo, so your request was not sent "
            f"and {names} won't contact you. If you need help, you can call them directly:"
            f"\n\n{_provider_phone_lines(state.matched_providers)}"
        )
    else:
        message = (
            f"You're all set. Your request (ref {lead_id}) is ready to go to {names}{period} "
            f"They'll have your number and the details above, so you won't need to explain it again."
        )
        if state.urgency == "emergency":
            message += f"\n\nIf you'd rather not wait, you can call any of them now:\n\n{_provider_phone_lines(state.matched_providers)}"
    if repair:
        message += f"\n\nWant help finding {repair.who} for the {repair.damage} too?"
    return message


# --- after the conversation ended ----------------------------------------------------

GAS_REMINDER = f"If you still smell gas, stay outside and call 911 or PG&E at {PGE_EMERGENCY_PHONE}."


def after_end(state: LeadState, llm: BaseChatModel) -> dict[str, Any]:
    """A message after any ending (converted, out_of_scope, declined, emergency).

    A short reply to what they actually said. Only adds a message: the lead (and
    everything else in state) is never reopened or changed. Anything new goes
    through the app's "Start a new request" button. Gas: the safety instruction
    comes first, from code, so it's always there.
    """
    try:
        reply = llm.invoke([SystemMessage(prompts.after_end_prompt(state)), *state.messages])
    except LLM_ERRORS as e:
        return llm_failure(state, "after_end", e)
    text = _text(reply).strip()
    if not text:
        return llm_failure(state, "after_end", ValueError(f"empty reply, content={reply.content!r:.200}"))
    if state.status == "emergency":
        text = f"{GAS_REMINDER}\n\n{text}"
    return {"messages": [AIMessage(text)]}
