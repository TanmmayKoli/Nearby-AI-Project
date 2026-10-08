"""Service categories as data.

Everything category-specific lives here (detail questions, emergency triggers,
common confusions, provider search terms) so prompts and nodes can read it
instead of hardcoding it. Adding a category = adding one entry to CATEGORIES.
"""

from dataclasses import dataclass, field
from typing import Literal

CategoryKey = Literal[
    "water_damage",
    "plumbing",
    "electrical",
    "hvac",
    "roofing",
    "pest_control",
    "appliance_repair",
    "handyman",
    "wildlife_removal",
]


@dataclass(frozen=True)
class DetailQuestion:
    """One category-specific question. `key` is where the answer goes in category_details.

    about_cause: the question asks what's CAUSING the problem. Skipped once the
    user has said they don't know the cause (the pro will diagnose it).
    always_ask: asked even after the usual detail-question budget is used up
    (wildlife damage: it decides whether we offer a separate repair request).
    """

    key: str
    question: str
    about_cause: bool = False
    always_ask: bool = False


@dataclass(frozen=True)
class CategorySpec:
    key: CategoryKey
    label: str
    description: str  # one line, helps the extractor pick the right category
    detail_questions: list[DetailQuestion]
    emergency_triggers: list[str]  # situations that make urgency = emergency
    confused_with: list[str]  # other category keys (or out-of-scope services) users mix up
    confusion_note: str  # how to tell them apart
    search_queries: list[str] = field(default_factory=list)  # Places Text Search terms (Phase 2)


# One rule, used as the confusion_note for BOTH categories so the extractor and
# the clarify question always apply the same tie-break.
PLUMBING_VS_WATER_DAMAGE = (
    "Plumbing vs water damage: an active leak or a stain with NO pooled/standing water "
    "and no soaked floors is plumbing (fix the source first). Water already pooled, soaked "
    "flooring or walls, or flooding is water_damage. If both apply (an active leak AND "
    "standing water), choose water_damage when it's an emergency with standing water; "
    "otherwise plumbing."
)

# Same rule for pest_control and wildlife_removal (confusion_note for both).
PEST_VS_WILDLIFE = (
    "Pest control vs wildlife removal: insects, termites, rats and mice are pest_control. "
    "Wild animals (raccoons, possums, skunks, squirrels, bats, birds) are wildlife_removal."
)

CATEGORIES: dict[str, CategorySpec] = {
    spec.key: spec
    for spec in [
        CategorySpec(
            key="water_damage",
            label="Water damage restoration",
            description="Water already in the home: flooding, soaked floors/walls, drying and cleanup, mold risk.",
            detail_questions=[
                DetailQuestion("still_active", "Is water still actively coming in?"),
                DetailQuestion("extent", "Roughly how much water, and how big an area?"),
                DetailQuestion("source", "Do you know the source (storm, pipe, appliance, or unknown)?", about_cause=True),
            ],
            emergency_triggers=["active flooding", "water near electrical", "sewage"],
            confused_with=["plumbing"],
            confusion_note=PLUMBING_VS_WATER_DAMAGE,
            search_queries=["water damage restoration"],
        ),
        CategorySpec(
            key="plumbing",
            label="Plumbing",
            description="Pipes, drains, toilets, sinks, faucets, water heaters, sewer lines.",
            detail_questions=[
                DetailQuestion("fixture", "What's affected (toilet, sink, water heater, pipe, drain)?", about_cause=True),
                DetailQuestion("symptom", "Is it leaking, clogged, or no water at all?"),
                DetailQuestion("can_shut_off", "Are you able to shut off the water?"),
            ],
            emergency_triggers=["burst pipe", "can't shut off water", "sewage backup", "no water"],
            confused_with=["water_damage"],
            confusion_note=PLUMBING_VS_WATER_DAMAGE,
            search_queries=["plumber"],
        ),
        CategorySpec(
            key="electrical",
            label="Electrical",
            description="Outlets, wiring, breakers, panels, lighting, power in part of the home.",
            detail_questions=[
                DetailQuestion("scope", "Is it one room or outlet, or the whole house?"),
                DetailQuestion("hazard_signs", "Any burning smell, sparks, or hot outlets?"),
                DetailQuestion("breaker", "Is a breaker tripping?"),
            ],
            emergency_triggers=["burning smell", "sparks", "smoke", "hot outlets", "water near electrical"],
            confused_with=["utility_outage"],
            confusion_note="If the whole street/neighborhood is out, it's a utility outage (PG&E or SMUD), not an electrician.",
            search_queries=["electrician"],
        ),
        CategorySpec(
            key="hvac",
            label="Heating & cooling (HVAC)",
            description="Furnace, AC, heat pump, ducts, thermostat problems.",
            detail_questions=[
                DetailQuestion("heat_or_cool", "Is it heating or cooling?"),
                DetailQuestion("symptom", "Is it completely dead, or weak, noisy, or leaking?"),
                DetailQuestion("system", "Do you know the system type or roughly how old it is?"),
            ],
            emergency_triggers=[
                "no heat in freezing weather",
                "no AC in extreme heat with vulnerable people (elderly, infants, medical needs)",
                "gas smell",
            ],
            confused_with=["electrical"],
            confusion_note="A dead thermostat or tripped breaker can look like an HVAC failure; still route to hvac unless it's clearly a wiring problem.",
            search_queries=["HVAC repair", "heating and air conditioning"],
        ),
        CategorySpec(
            key="roofing",
            label="Roofing",
            description="Roof leaks, missing shingles, storm or tree damage to the roof.",
            detail_questions=[
                DetailQuestion("active_leak", "Is water actively leaking inside right now?"),
                DetailQuestion("cause", "What caused it (storm, age, tree)?", about_cause=True),
                DetailQuestion("roof_type", "Do you know the roof type (shingle, tile, flat)?"),
            ],
            emergency_triggers=["active leak during rain", "large hole", "tree on roof"],
            confused_with=["water_damage"],
            confusion_note="Roofing fixes the roof; interior soaking/cleanup is water_damage. A leaking roof with a wet ceiling is still roofing first.",
            search_queries=["roofing contractor", "roof repair"],
        ),
        CategorySpec(
            key="pest_control",
            label="Pest control",
            description="Insects, rodents, termites, wasps/bees.",
            detail_questions=[
                DetailQuestion("pest", "What kind of pest is it?"),
                DetailQuestion("location", "Where are you seeing them, and how widespread?"),
                DetailQuestion("duration", "How long has this been going on?"),
            ],
            emergency_triggers=["wasp or bee nest near an entry", "rodents in living areas with kids"],
            confused_with=["wildlife_removal"],
            confusion_note=PEST_VS_WILDLIFE,
            search_queries=["pest control"],
        ),
        CategorySpec(
            key="appliance_repair",
            label="Appliance repair",
            description="Washer, dryer, dishwasher, fridge, oven/range, microwave.",
            detail_questions=[
                DetailQuestion("appliance", "Which appliance, and what brand?"),
                DetailQuestion("symptom", "What is it doing (or not doing)?"),
                DetailQuestion("fuel", "Is it gas or electric?"),
            ],
            emergency_triggers=["gas smell", "burning smell", "heavy leaking"],
            confused_with=["plumbing", "hvac"],
            confusion_note="Washer/dishwasher supply-line leaks may be plumbing; water heaters usually go to plumbing or hvac, not appliance repair.",
            search_queries=["appliance repair"],
        ),
        CategorySpec(
            key="handyman",
            label="Handyman",
            description="Small repairs, installs, assembly, painting, doors, drywall patches.",
            detail_questions=[
                DetailQuestion("task", "What's the task (repair, install, assemble, paint)?"),
                DetailQuestion("area", "Which part of the home?"),
                DetailQuestion("scope", "Is it one task or several?"),
            ],
            emergency_triggers=["broken lock", "exterior door that won't close"],
            confused_with=["electrical", "plumbing"],
            confusion_note="Anything beyond minor fixes in electrical or plumbing needs a licensed pro, not a handyman.",
            search_queries=["handyman"],
        ),
        CategorySpec(
            key="wildlife_removal",
            label="Wildlife removal",
            description=(
                "Raccoons, possums, skunks, squirrels, bats or birds in the attic, crawlspace, "
                "chimney or walls. Humane removal and sealing entry points."
            ),
            detail_questions=[
                DetailQuestion("animal", "What kind of animal is it?"),
                DetailQuestion("location", "Where is it: attic, crawlspace, chimney, or walls?"),
                DetailQuestion("damage", "Have you noticed any damage or places where it's getting in?", always_ask=True),
            ],
            emergency_triggers=[
                "an animal loose in the living space",
                "a bat in a living area",
                "a sick or aggressive animal",
            ],
            confused_with=["pest_control"],
            confusion_note=PEST_VS_WILDLIFE,
            search_queries=["wildlife removal", "animal removal"],
        ),
    ]
}


# --- Safety -----------------------------------------------------------------
# Flags the extractor may set. Two tiers (behavior lands in Phase 4):
#   HARD_STOP_FLAGS:  give safety instructions and END. No lead collection.
#   URGENT_SAFETY_FLAGS: give safety instructions FIRST, raise urgency (see
#                     route), then keep collecting the lead.

SafetyFlag = Literal[
    "gas_smell",
    "sparks_or_smoke",
    "burning_smell",
    "water_near_electrical",
    "sewage",
    "active_flooding",
    "animal_contact",
    "ongoing_electrical",
]

SAFETY_FLAG_DESCRIPTIONS: dict[str, str] = {
    "gas_smell": "User smells gas (rotten eggs / sulfur).",
    "sparks_or_smoke": "Sparks, smoke, or visible fire from wiring, outlets, or appliances.",
    "burning_smell": "Burning or electrical smell.",
    "water_near_electrical": "Water in contact with or near outlets, panels, or wiring.",
    "sewage": "Sewage or black water present.",
    "active_flooding": "Water is still actively coming in.",
    "animal_contact": "Someone was bitten or scratched by a wild animal, or touched or was near a bat.",
    "ongoing_electrical": "An electrical hazard that's still happening: repeated sparking, smoke, a hot or warm "
    "outlet/plug/cord, or scorch or melt marks. NOT a single spark that stopped.",
}

HARD_STOP_FLAGS: frozenset[str] = frozenset({"gas_smell"})
URGENT_SAFETY_FLAGS: frozenset[str] = frozenset({"sparks_or_smoke", "burning_smell", "ongoing_electrical"})
# These force urgency = emergency in route, whatever the extractor said.
EMERGENCY_SAFETY_FLAGS: frozenset[str] = frozenset({"burning_smell", "ongoing_electrical"})

PGE_EMERGENCY_PHONE = "1-800-743-5000"

# Shown for a gas smell. Lead collection stops (status "emergency").
GAS_EMERGENCY_MESSAGE = (
    "If you smell gas, please act now:\n\n"
    "1. Leave the house right away and get everyone out, pets too.\n"
    "2. Don't flip light switches, use appliances or phones inside, or light anything.\n"
    f"3. Once you're outside and away from the house, call 911 or PG&E at {PGE_EMERGENCY_PHONE}.\n\n"
    "I'll stop here so you can focus on getting safe. Once PG&E says it's safe, start a new "
    "conversation and I can help you find a pro for any repairs."
)

# Shown once, at the start of the agent's reply, then lead collection continues.
# URGENT_SAFETY_FLAGS also set an urgency floor: burning smell -> emergency;
# sparks -> at least within_48h (emergency if ongoing, decided by the extractor).
SAFETY_NOTICES: dict[str, str] = {
    "sparks_or_smoke": (
        "Safety first: stay away from it and don't touch the outlet or anything plugged in. "
        "If you can reach your breaker panel safely (not standing in water), switch off that "
        "circuit. If you see flames or the smoke gets worse, get out and call 911."
    ),
    "burning_smell": (
        "Safety first: stop using that outlet or appliance. If you can reach your breaker "
        "panel safely, switch off that circuit. If you see smoke or flames, get out and call 911."
    ),
    "ongoing_electrical": (
        "Safety first: stop using that outlet or appliance and don't touch it. If you can reach your "
        "breaker panel safely, switch off that circuit. If you see flames or the smoke gets worse, "
        "get out and call 911."
    ),
    "water_near_electrical": (
        "Quick safety tip: stay out of any water that's near outlets, cords, or your electrical panel."
    ),
    "animal_contact": (
        "Safety first: don't try to handle the animal. Because of the bite, scratch or bat contact, "
        "please call a doctor or your county public health office today, before anything else."
    ),
}


# --- Out of scope -------------------------------------------------------------
# Requests we redirect instead of turning into a lead. These conversations get
# status "out_of_scope" and are EXCLUDED from the conversion-rate denominator
# in evals (there was never a lead to win).

OutOfScopeReason = Literal["utility_outage", "other"]

# Set by code (route), never by the extractor: the zip is real but outside the
# region. Counted with out_of_scope in evals (we can't serve them).
OUT_OF_AREA_MESSAGE = (
    "Sorry, we only cover the Davis–Sacramento area right now (Davis, Woodland, West "
    "Sacramento and Sacramento), so I can't match you with a pro near {zip}."
)

OUT_OF_SCOPE: dict[str, str] = {
    "utility_outage": (
        "That sounds like a utility outage rather than a problem in your home, so an electrician "
        "won't be able to fix it. Contact your utility: PG&E serves Davis, Woodland, and West "
        "Sacramento, and SMUD serves Sacramento. Both have outage maps and reporting on their websites."
    ),
    "other": (
        "That's outside what I can help with. I connect people with local pros for water damage, "
        "plumbing, electrical, heating and cooling, roofing, pest control, appliance repair, "
        "handyman work, and wildlife removal."
    ),
}


def get_category(key: str | None) -> CategorySpec | None:
    """Return the spec for a category key, or None if unknown/missing."""
    if key is None:
        return None
    return CATEGORIES.get(key)
