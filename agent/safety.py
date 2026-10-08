"""Deterministic safety gate. Runs on every user message BEFORE the LLM.

Why keywords and not the LLM: a gas smell must get safety instructions even if
the API is slow or down, and the check should be instant and explainable. The
extractor's safety_flags are a second net for phrasings these patterns miss
(route checks them after extract).

Deliberately conservative: a false alarm costs one extra safety message; a miss
could cost much more. Simple negation ("I don't smell gas", "no sparks") is
skipped so routine conversations aren't derailed.
"""

import re

from agent.categories import SAFETY_NOTICES

PATTERNS: dict[str, re.Pattern] = {
    "gas_smell": re.compile(
        r"\b(smell(s|ed|ing)?\s+(of\s+|like\s+)?(natural\s+)?gas|gas\s+(smell|leak|odou?r)s?|"
        r"rotten\s+eggs?|smells?\s+like\s+(sulfur|sulphur))\b",
        re.IGNORECASE,
    ),
    "sparks_or_smoke": re.compile(
        r"\b(spark(s|ed|ing|y)?|smok(e|ing|ey|y)(?!\s+(detector|alarm)s?))\b", re.IGNORECASE
    ),
    "burning_smell": re.compile(
        r"\b(burning\s+(smell|plastic|odou?r)|smell(s|ed|ing)?\s+(like\s+)?(something\s+)?burn(ing|t|ed)|"
        r"melt(ed|ing))\b",
        re.IGNORECASE,
    ),
}

_NEGATION = re.compile(r"\b(no|not|don'?t|doesn'?t|didn'?t|never|without|isn'?t|aren'?t)\b[^.!?]*$", re.IGNORECASE)
_NEGATION_WINDOW = 25  # characters before the match to look for a negation

# Ongoing electrical hazard -> urgency is forced to emergency in route (a single
# spark that stopped is NOT this; it stays within_48h). Repeated-sparking phrases
# are affirmative by construction ("won't stop sparking" contains a negation
# word), so they skip the negation check; smoke / hot / scorched use it.
ONGOING_SPARKING = re.compile(
    r"\b((keeps?|kept|still|constantly|repeatedly)\s+spark(ing|s)?|won'?t\s+stop\s+spark(ing)?|"
    r"spark(s|ing)?\s+(every\s*time|each\s+time|whenever|constantly|repeatedly|over\s+and\s+over))\b",
    re.IGNORECASE,
)
SMOKE = re.compile(r"\bsmok(e|ing|ey|y)\b(?!\s+(detector|alarm)s?)", re.IGNORECASE)
SCORCHED = re.compile(r"\b(scorch(ed|ing|es)?|charred|melt(ed|ing))\b", re.IGNORECASE)
HOT = re.compile(r"\b(hot|warm|heat(s|ing)?\s+up)\b", re.IGNORECASE)
ELECTRICAL_THING = re.compile(
    r"\b(outlets?|plugs?|sockets?|switch(es)?|cords?|wires?|wiring|breakers?|(cover|face)\s*plates?|"
    r"(electrical|breaker)\s+(panel|box))\b",
    re.IGNORECASE,
)
_NEARBY = 40  # a hot/warm word counts only within this many characters of an electrical thing


def _affirmed(pattern: re.Pattern, text: str) -> list[re.Match]:
    """Matches of pattern that aren't preceded by a negation ("no smoke", "isn't hot")."""
    return [
        m for m in pattern.finditer(text)
        if not _NEGATION.search(text[max(0, m.start() - _NEGATION_WINDOW): m.start()])
    ]


def ongoing_electrical_hazard(text: str) -> bool:
    """Repeated sparking, smoke, a hot/warm outlet or plug, or scorch/melt marks."""
    if ONGOING_SPARKING.search(text) or _affirmed(SMOKE, text) or _affirmed(SCORCHED, text):
        return True
    return any(
        ELECTRICAL_THING.search(text[max(0, m.start() - _NEARBY): m.end() + _NEARBY]) for m in _affirmed(HOT, text)
    )


def detect_hazards(text: str) -> list[str]:
    """Safety flags whose pattern appears in text, unless clearly negated."""
    found = [flag for flag, pattern in PATTERNS.items() if _affirmed(pattern, text)]
    if ongoing_electrical_hazard(text):
        found.append("ongoing_electrical")
    return found


# One electrical notice at most: the first of these present covers the others.
ELECTRICAL_NOTICE_ORDER = ["sparks_or_smoke", "burning_smell", "ongoing_electrical"]


def safety_notice(new_flags: list[str]) -> str | None:
    """Combined notice for flags not yet warned about (one electrical notice max)."""
    electrical = [f for f in ELECTRICAL_NOTICE_ORDER if f in new_flags][:1]
    flags = [f for f in new_flags if f not in ELECTRICAL_NOTICE_ORDER] + electrical
    texts = [SAFETY_NOTICES[f] for f in flags if f in SAFETY_NOTICES]
    return "\n\n".join(texts) if texts else None
