"""State models.

Three models, three jobs:
- ExtractedFields: what the LLM returns from one user turn (everything optional).
- LeadState:       what the graph carries across the conversation.
- Lead:            the final dispatchable record written to leads/. Required
                   fields are non-optional, so a Lead can't exist half-filled.

Rule: the LLM extracts, code decides. `merge_extracted` and
`missing_required_fields` are plain Python so routing stays testable.

Problem details are kept as an append-only list of `facts`, not an
overwritable summary, so nothing the user said gets lost between turns.
`problem_description` is generated ONCE from `facts`, right before confirm
(Phase 3/4 node), and stays None during the conversation.
"""

import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.categories import CATEGORIES, CategoryKey, OutOfScopeReason, SafetyFlag

Urgency = Literal["emergency", "within_48h", "within_week", "flexible"]
PropertyType = Literal["home", "business"]
OwnerOrRenter = Literal["owner", "renter"]
# out_of_scope: redirected (utility outage, not a home service, out of area). Excluded from the
# conversion-rate denominator in evals.
# declined: the user said no to sharing contact info (at consent, or refused both
# phone and email). Counts as a non-conversion, distinct from walking away.
Status = Literal[
    "in_progress", "emergency", "awaiting_confirm", "converted", "abandoned", "out_of_scope", "declined"
]
ContactChannel = Literal["phone", "email"]
RemovableField = Literal["contact_phone", "contact_email", "address"]

# Below this, route sends the user to `clarify` instead of continuing.
CATEGORY_CONFIDENCE_THRESHOLD = 0.7


# --- LLM output ---------------------------------------------------------------


class DetailAnswer(BaseModel):
    """One answer to a category detail question (a list of these, not a dict,
    because strict structured output can't express free-form dict keys)."""

    key: str = Field(description="The detail question key, e.g. still_active.")
    value: str = Field(description="The user's answer, in their terms.")


def _all_fields_required(schema: dict) -> None:
    """Mark every property required in the JSON schema sent to the API.

    Values can still be null, so "nothing new" is expressible. Optional
    properties are what blow up the structured-output grammar ("Schema is too
    complex"); required-but-nullable is the documented workaround. Python code
    can still construct ExtractedFields with defaults.
    """
    schema["required"] = list(schema.get("properties", {}))


class ExtractedFields(BaseModel):
    """Fields pulled from the conversation. Leave anything the user didn't say as null.

    Extraction contract: the extractor receives the FULL message history plus
    the current state. That lets it (a) report only what's new this turn
    instead of re-reporting known facts, and (b) judge remarks in context,
    so an offhand "no rush I guess" doesn't downgrade an urgency the user
    already made clear. A null means "nothing new", never "clear this field".
    """

    model_config = ConfigDict(json_schema_extra=_all_fields_required)

    category: CategoryKey | None = Field(None, description="Best-fit service category.")
    category_confidence: float | None = Field(
        None, ge=0, le=1, description="0-1 confidence in `category`. Below 0.7 means ambiguous."
    )
    new_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Short factual statements about the problem the user made THIS turn that aren't "
            "already in state.facts. User's own terms, no speculation."
        ),
    )
    retracted_facts: list[str] = Field(
        default_factory=list,
        description="Exact entries from state.facts that no longer apply because the user "
        "corrected or changed the problem. Empty if none.",
    )
    out_of_scope_reason: OutOfScopeReason | None = Field(
        None, description="Set only if the request isn't a home service we cover."
    )
    cause_unknown: bool = Field(
        False, description="True if the user has said they don't know what's causing the problem."
    )
    declined_contact: list[ContactChannel] = Field(
        default_factory=list, description="Contact methods the user said they won't share."
    )
    remove_fields: list[RemovableField] = Field(
        default_factory=list,
        description="Fields the user asked to stop using (e.g. 'use my email instead' -> contact_phone).",
    )
    urgency: Urgency | None = Field(None, description="Only if stated or clearly implied.")
    zip: str | None = Field(None, description="5-digit US zip code.")
    name: str | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    address: str | None = None
    property_type: PropertyType | None = None
    owner_or_renter: OwnerOrRenter | None = None
    availability: str | None = Field(None, description="When the user is available, in their words.")
    category_details: list[DetailAnswer] = Field(
        default_factory=list,
        description="Answers to the chosen category's detail questions, one per key. Empty if none.",
    )
    safety_flags: list[SafetyFlag] = Field(default_factory=list)

    @field_validator("category_details", mode="before")
    @classmethod
    def _details_from_dict(cls, value: Any) -> Any:
        """Accept {"source": "storm"} too (handy in code and tests)."""
        if isinstance(value, dict):
            return [{"key": k, "value": v} for k, v in value.items()]
        return value

    @model_validator(mode="after")
    def _category_needs_confidence(self) -> "ExtractedFields":
        if self.category is not None and self.category_confidence is None:
            raise ValueError("category_confidence is required when category is set")
        return self


# --- Graph state --------------------------------------------------------------


class LeadState(BaseModel):
    """Everything the graph knows about one conversation."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # Lead fields
    category: CategoryKey | None = None
    facts: list[str] = Field(default_factory=list)  # append-only, from new_facts
    problem_description: str | None = None  # None until generated from facts before confirm
    urgency: Urgency | None = None
    zip: str | None = None
    name: str | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    address: str | None = None
    property_type: PropertyType | None = None
    owner_or_renter: OwnerOrRenter | None = None
    availability: str | None = None
    category_details: dict[str, str] = Field(default_factory=dict)
    safety_flags: list[SafetyFlag] = Field(default_factory=list)
    matched_providers: list[dict[str, Any]] = Field(default_factory=list)
    consent_to_share: bool = False
    # evals break down out-of-scope endings by this; "out_of_area" is set by route, not the extractor
    out_of_scope_reason: OutOfScopeReason | Literal["out_of_area"] | None = None
    declined: list[ContactChannel] = Field(default_factory=list)  # contact methods the user refused
    cause_unknown: bool = False  # user said they don't know the cause: skip cause questions
    # Wildlife lead with damage another trade repairs (torn vent, chewed wiring):
    # offered as a separate request after this lead is created.
    repair_offer: str | None = None  # the damage, e.g. "torn vent"
    repair_category: str | None = None  # the trade that repairs it, e.g. "roofing"

    # Routing metadata
    category_confidence: float = 0.0
    missing_fields: list[str] = Field(default_factory=list)
    turn_count: int = 0
    status: Status = "in_progress"
    last_route: str | None = None  # route's last decision, for the "agent brain" panel + evals
    asked: list[str] = Field(default_factory=list)  # ask_next targets so far (caps detail questions, avoids loops)
    extraction_errors: list[str] = Field(default_factory=list)  # validation failures, surfaced in evals
    llm_errors: list[str] = Field(default_factory=list)  # timeouts/API errors after retries, surfaced in evals
    lead_id: str | None = None  # set by create_lead
    category_locked: CategoryKey | None = None  # accepted after one clarify, even if confidence stayed low
    warned: list[SafetyFlag] = Field(default_factory=list)  # safety notices already shown
    pending_notice: str | None = None  # safety notice to prepend to this turn's reply


# --- Final record ---------------------------------------------------------------


class Lead(BaseModel):
    """A dispatchable lead. A provider should be able to act on this without the chat."""

    lead_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    category: CategoryKey
    problem_description: str  # generated once from facts, before confirm
    facts: list[str]  # raw details, so the provider sees exactly what the user said
    urgency: Urgency
    zip: str
    name: str
    contact_phone: str | None = None
    contact_email: str | None = None
    address: str | None = None
    property_type: PropertyType | None = None
    owner_or_renter: OwnerOrRenter | None = None
    availability: str | None = None
    category_details: dict[str, str] = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    matched_providers: list[dict[str, Any]] = Field(min_length=1)
    consent_to_share: Literal[True]  # no consent, no lead

    @model_validator(mode="after")
    def _needs_contact(self) -> "Lead":
        if not (self.contact_phone or self.contact_email):
            raise ValueError("a lead needs contact_phone or contact_email")
        return self


# --- Validation ---------------------------------------------------------------
# Invalid values are dropped (return None) so the field stays missing and gets
# re-asked, instead of a bad value silently ending up in a lead.


def normalize_zip(value: str | None) -> str | None:
    """'95616' or '95616-1234' -> '95616'. Anything else -> None."""
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d{5})(?:-\d{4})?\s*", value)
    return match.group(1) if match else None


def normalize_phone(value: str | None) -> str | None:
    """Any 10-digit US number (optionally with leading 1) -> '(530) 555-0123'."""
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def normalize_email(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    return value if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) else None


# --- Merge & completeness -------------------------------------------------------

_SCALAR_FIELDS = [
    "category",
    "out_of_scope_reason",
    "urgency",
    "name",
    "address",
    "property_type",
    "owner_or_renter",
    "availability",
]


def merge_extracted(state: LeadState, extracted: ExtractedFields) -> dict[str, Any]:
    """Return state updates from one extraction.

    - A null never overwrites a known value.
    - A new non-null value does overwrite (the user corrected themselves).
    - new_facts append to facts (exact duplicates skipped); retracted_facts are removed.
    - A category change drops category_details that belong to the old category.
    - remove_fields clears those fields; declined_contact accumulates.
    - category_details merge per key; safety_flags only accumulate.
    - zip/phone/email are validated; invalid ones are dropped.
    """
    updates: dict[str, Any] = {}

    for name in _SCALAR_FIELDS:
        value = getattr(extracted, name)
        if value is not None:
            updates[name] = value

    if extracted.category is not None and extracted.category_confidence is not None:
        updates["category_confidence"] = extracted.category_confidence

    for name, normalizer in [
        ("zip", normalize_zip),
        ("contact_phone", normalize_phone),
        ("contact_email", normalize_email),
    ]:
        value = normalizer(getattr(extracted, name))
        if value is not None:
            updates[name] = value

    facts = [f for f in state.facts if f not in set(extracted.retracted_facts)]
    for fact in extracted.new_facts:
        if fact not in facts:
            facts.append(fact)
    if facts != state.facts:
        updates["facts"] = facts

    details = dict(state.category_details)
    new_category = updates.get("category")
    if new_category and new_category != state.category:
        valid = {q.key for q in CATEGORIES[new_category].detail_questions}
        details = {k: v for k, v in details.items() if k in valid}
    details.update({d.key: d.value for d in extracted.category_details})
    if details != state.category_details:
        updates["category_details"] = details

    for name in extracted.remove_fields:
        if name not in updates:  # "use my email instead: x@y.com" sets email, clears phone
            updates[name] = None

    if extracted.cause_unknown and not state.cause_unknown:
        updates["cause_unknown"] = True  # sticky: a later "probably X" doesn't un-ask anything

    new_declined = [c for c in extracted.declined_contact if c not in state.declined]
    if new_declined:
        updates["declined"] = state.declined + new_declined

    new_flags = [f for f in extracted.safety_flags if f not in state.safety_flags]
    if new_flags:
        updates["safety_flags"] = state.safety_flags + new_flags

    return updates


def category_is_confident(state: LeadState) -> bool:
    """Confident, or accepted after one clarify question (category_locked)."""
    if state.category is None:
        return False
    return state.category_confidence >= CATEGORY_CONFIDENCE_THRESHOLD or state.category == state.category_locked


def missing_required_fields(state: LeadState) -> list[str]:
    """Required fields still missing, in the order they should be asked.

    'facts' means we know nothing about the problem yet. 'contact' means
    neither phone nor email. Consent is handled at confirm, not here.
    """
    missing = []
    if not category_is_confident(state):
        missing.append("category")
    if not state.facts:
        missing.append("facts")
    if not state.urgency:
        missing.append("urgency")
    if not state.zip:
        missing.append("zip")
    if not state.name:
        missing.append("name")
    if not (state.contact_phone or state.contact_email):
        missing.append("contact")
    return missing
