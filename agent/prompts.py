"""Prompts. Category knowledge comes from categories.py, never hardcoded here."""

import json
from functools import lru_cache

from agent.categories import CATEGORIES, OUT_OF_SCOPE, PLUMBING_VS_WATER_DAMAGE, SAFETY_FLAG_DESCRIPTIONS
from agent.state import LeadState

SERVICE_AREA = "Davis, Woodland, West Sacramento, and Sacramento, CA"

URGENCY_GUIDE = """\
Map conservatively and NEVER round up. If it's between two levels, pick the calmer one, \
or leave urgency null so the assistant asks.
- emergency: damage or a safety risk happening right now (water actively coming in, \
ongoing sparking, smoke, a burning smell, an outlet that's hot or scorched, sewage, \
can't shut off water, no heat in freezing weather), or the user says it can't wait.
- Electrical: a single spark that stopped, with no heat, smell, smoke or scorching, is \
within_48h, not emergency. If later answers show that, correct an earlier emergency.
- within_48h: ONLY if the user says today, tomorrow, ASAP, "as soon as possible", \
"right away", or there's a clear urgency signal like the ones above.
- within_week: "next few days", "this week", "soon", "in a few days", or a real \
problem that isn't getting worse quickly (a running toilet, a slow drain, a dripping \
faucet, one dead outlet).
- flexible: no time pressure ("whenever", "no rush", planning ahead, small jobs)."""


# --- Extraction -----------------------------------------------------------------


@lru_cache(maxsize=1)
def extraction_system_prompt() -> str:
    """Static part of the extraction prompt, built from the category table."""
    category_lines = []
    for spec in CATEGORIES.values():
        details = "; ".join(f'"{q.key}": {q.question}' for q in spec.detail_questions)
        category_lines.append(
            f"### {spec.key} ({spec.label})\n"
            f"{spec.description}\n"
            f"Emergency signs: {', '.join(spec.emergency_triggers)}\n"
            f"Often confused with {', '.join(spec.confused_with)}: {spec.confusion_note}\n"
            f"category_details keys: {details}"
        )
    safety_lines = "\n".join(f"- {k}: {v}" for k, v in SAFETY_FLAG_DESCRIPTIONS.items())
    oos_lines = "\n".join(f"- {k}" for k in OUT_OF_SCOPE)

    return f"""\
You extract structured information from a conversation between a homeowner and an \
intake assistant for a service that matches people in {SERVICE_AREA} with local \
home service pros.

## How this works (extraction contract)
- You receive the FULL conversation and the CURRENT STATE (what is already known).
- Report only what is NEW or CHANGED in the user's messages. Leave a field null if \
there is nothing new. Null means "nothing new", never "clear this field".
- Never invent values. If the user didn't say it (or clearly imply it), leave it null.
- If the user corrects something (e.g. a different zip), return the corrected value.
- Judge remarks in context. An offhand comment ("no rush I guess") does not \
downgrade an urgency the user already made clear.
- Only extract from what the USER said, not from the assistant's questions.

## Fields
- category + category_confidence (0-1): best-fit category below. Use < 0.7 when \
the problem could reasonably be two categories or is too vague. Always set both or neither.
- new_facts: short factual statements about the problem the user made that are NOT \
already in state.facts. One fact per item, in the user's terms, keep specifics \
(amounts, locations, timing, what they tried). No speculation, no advice, and no \
contact info or zip (those have their own fields).
- retracted_facts: if the user corrects or changes the problem ("actually it's the \
roof, not a pipe"), list the exact state.facts entries that no longer apply. Keep \
facts that still apply. Also update category if it changed.
- urgency: one of the levels below, only if stated or clearly implied.
- category_details: answers to the chosen category's detail questions, as a list of \
{{key, value}} using the keys listed for that category. Only record a key when the \
user's words (in ANY message so far) directly answer that question. Check the \
"unanswered_detail_keys" in CURRENT STATE every turn: if something the user already \
said answers one (e.g. cause = "raccoon" after "the raccoon tore up a vent"; symptom = \
"spreading water stain on ceiling" after they described one), fill it so it isn't asked. Do NOT infer answers: "one of the outlets \
sparked" does not answer "is it one outlet or the whole house?". No inferred negatives \
or scope ("only", "just", "limited to") unless the user said it. Short values in the \
user's terms. If the user answers a detail question with "I don't know" / "not sure", \
record that key with value "unknown". Empty list if nothing new.
- cause_unknown: true if the user says they don't know what's causing it ("not sure \
what's causing it", "no idea where it's coming from"). A guess with "probably" is not \
knowing.
- zip: 5-digit zip. name, contact_phone, contact_email, address: only if given.
- property_type (home/business), owner_or_renter: only if stated.
- availability: when they're available, in their words.
- safety_flags: any that apply (list below).
- out_of_scope_reason: set only if the request is not one of the categories below \
(e.g. a neighborhood power outage, wildlife like raccoons, car repair).
- Wildlife damage: if the user describes damage an animal caused (torn roof vent -> \
roofing, chewed wiring -> electrical, soiled insulation or a broken screen -> handyman), \
set that repair category with confidence. Removing the animal itself stays out of scope.
- declined_contact: "phone" and/or "email" if the user says they won't share that.
- remove_fields: fields the user asks you to stop using, e.g. "use my email instead" \
-> ["contact_phone"] (and set contact_email if they gave one).

## Urgency levels
{URGENCY_GUIDE}

## Categories
{chr(10).join(category_lines)}

## Category rules (apply these before choosing)
- {PLUMBING_VS_WATER_DAMAGE}
- Example: "brown stain spreading on the ceiling, no water on the floor, probably \
leaking from the bathroom" -> plumbing, high confidence.
- After the assistant asks a clarifying question about the category, the user's \
answer usually settles it: apply these rules to it and return the category with a \
confidence that reflects the answer (>= 0.7 when the rule clearly decides).

## Safety flags
{safety_lines}

## Out-of-scope reasons
{oos_lines}
"""


def known_state_json(state: LeadState) -> str:
    """The CURRENT STATE block the extractor sees (no messages, no routing internals)."""
    known = {
        "category": state.category,
        "category_confidence": state.category_confidence if state.category else None,
        "facts": state.facts,
        "urgency": state.urgency,
        "zip": state.zip,
        "name": state.name,
        "contact_phone": state.contact_phone,
        "contact_email": state.contact_email,
        "address": state.address,
        "property_type": state.property_type,
        "owner_or_renter": state.owner_or_renter,
        "availability": state.availability,
        "category_details": state.category_details,
        "unanswered_detail_keys": _unanswered_detail_keys(state),
        "safety_flags": state.safety_flags,
    }
    return json.dumps(known, indent=2)


def _unanswered_detail_keys(state: LeadState) -> dict[str, str]:
    """Detail questions for the current category not answered yet, so the
    extractor can fill any that earlier messages already answer."""
    if state.category is None:
        return {}
    return {
        q.key: q.question
        for q in CATEGORIES[state.category].detail_questions
        if q.key not in state.category_details
    }


def extraction_state_prompt(state: LeadState, retry_error: str | None = None) -> str:
    text = f"## CURRENT STATE\n{known_state_json(state)}"
    if retry_error:
        text += (
            "\n\n## Your previous output failed validation\n"
            f"{retry_error}\nReturn corrected output. Remember: if you set category, "
            "also set category_confidence."
        )
    return text


# --- Asking the next question ---------------------------------------------------

ASK_SYSTEM = f"""\
You are the intake assistant for a service that connects people in {SERVICE_AREA} \
with local home service pros. Write your next chat message to the user.

Rules:
- Ask exactly ONE question: the one described under "Next question". Don't ask \
for anything else.
- 1-3 short sentences. Plain, warm, human. Not a form, not salesy.
- Briefly acknowledge what they just said only when it helps (no "Great!" every turn).
- Don't ask for anything already known (see "Known so far").
- Don't promise prices, arrival times, or that a pro will definitely come.
- No repair instructions. Don't add safety warnings yourself; any needed safety notice \
is shown separately right before your message.
- If this question was already asked, rephrase it more simply instead of repeating it."""

_TARGET_INSTRUCTIONS = {
    "facts": "Ask them to describe what's going on at their home.",
    "category": "Their problem is still unclear. Ask a short question that would tell "
    "you what kind of help they need.",
    "urgency": "Ask how soon they need someone: right away, in the next day or two, "
    "this week, or whenever works.",
    "zip": "Ask for their zip code so you can find pros nearby.",
    "contact": "Ask for their name and the best phone number for a pro to reach them "
    "(email is fine if they'd rather). Mention it's so the pro can contact them.",
    "contact_email": "They'd rather not share a phone number, and that's fine. Offer email "
    "instead, once, briefly and with no pressure.",
    "contact_required": "They've declined both phone and email. Explain briefly and kindly "
    "that a pro needs some way to reach them to take the job, and ask if they'd be willing "
    "to share either one. No pressure.",
}


def clarify_instruction(state: LeadState) -> str:
    """One question offering the 2 likely categories, using categories.py confusion data."""
    spec = CATEGORIES[state.category]
    other_key = next((k for k in spec.confused_with if k in CATEGORIES), None)
    if other_key is None:
        return f"Confirm in one short question whether they need help with {spec.label.lower()}."
    other = CATEGORIES[other_key]
    return (
        f"It's unclear whether they need {spec.label.lower()} or {other.label.lower()}. "
        f"Ask ONE short question that would settle it under this rule, in plain words "
        f"(not category names like '{spec.key}'). The rule: {spec.confusion_note}"
    )


EMERGENCY_ADDRESS_INSTRUCTION = (
    "In the same message, also invite them to share their street address, clearly "
    "as optional, e.g. \"If you're comfortable sharing your address, the crew can head "
    "straight there.\" Don't make it sound required."
)


def offer_address(state: LeadState, already_asked: bool) -> bool:
    """Emergencies: offer the (optional) address once, alongside the first contact
    question. If the user skips it, it's never asked again."""
    return state.urgency == "emergency" and not state.address and not already_asked


def ask_system_prompt(state: LeadState, target: str, already_asked: bool) -> str:
    """System prompt for a question. If a safety notice is pending, it is shown
    separately (prepended by code), so the question shouldn't repeat it."""
    if target.startswith("detail:"):
        key = target.split(":", 1)[1]
        question = next(q.question for q in CATEGORIES[state.category].detail_questions if q.key == key)
        instruction = f"Ask this, in your own words: {question}"
    elif target == "clarify":
        instruction = clarify_instruction(state)
    elif target == "contact":
        missing = [f for f, v in [("name", state.name), ("phone or email", state.contact_phone or state.contact_email)] if not v]
        instruction = _TARGET_INSTRUCTIONS["contact"] + f" Still needed: {', '.join(missing)}."
        if offer_address(state, already_asked):
            instruction += " " + EMERGENCY_ADDRESS_INSTRUCTION
    else:
        instruction = _TARGET_INSTRUCTIONS[target]
    if already_asked:
        instruction += " (You already asked this once; rephrase it.)"
    return f"{ASK_SYSTEM}\n\n## Known so far\n{known_state_json(state)}\n\n## Next question\n{instruction}"


# --- Summary + consent ---------------------------------------------------------

SUMMARIZE_SYSTEM = """\
Write a problem description for a home service pro who will read it cold, without \
the chat. 1-2 tight sentences, plain and specific.
Only restate the facts given, combined into readable sentences. Use details only to \
add something the facts don't already say; never restate a detail as its own \
sentence when the facts cover it ("The source is the storm" after "...after the \
storm"). Every claim must come from the input. If something isn't stated, leave it out.
Refer to the person as "the customer" (or avoid referring to them). Never say \
"homeowner", "owner", "renter" or "tenant" unless owner_or_renter is given.
Do NOT add:
- inferred negatives or scope claims ("only", "just", "limited to", "single", "no \
other", "the rest works") unless those words' meaning is in the facts;
- commentary, assessments, or severity/urgency language ("this is an emergency", \
"needs prompt attention", "serious") - urgency is a separate field;
- causes, diagnoses, or recommendations the user didn't state;
- names, phone numbers, or addresses."""


def summarize_input(state: LeadState) -> str:
    return json.dumps(
        {
            "category": state.category,
            # urgency deliberately left out: it's its own field, and including it
            # invites "this is an emergency..." commentary in the description.
            "facts": state.facts,
            "details": {k: v for k, v in state.category_details.items() if v != "unknown"},
            **({"owner_or_renter": state.owner_or_renter} if state.owner_or_renter else {}),
        },
        indent=2,
    )


CONSENT_SYSTEM = """\
The assistant just showed the user a summary of their request and asked whether it's \
OK to share their name and contact info with the listed pros. Classify the user's reply:
- yes: they agree (e.g. "yes", "sure", "go ahead").
- no: they decline to share or want to stop.
- changes: they want to correct or add something before sending."""
