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


def detect_hazards(text: str) -> list[str]:
    """Safety flags whose pattern appears in text, unless clearly negated."""
    found = []
    for flag, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            before = text[max(0, match.start() - _NEGATION_WINDOW): match.start()]
            if not _NEGATION.search(before):
                found.append(flag)
                break
    return found


def safety_notice(new_flags: list[str]) -> str | None:
    """Combined notice for flags not yet warned about. Sparks covers burning."""
    flags = list(new_flags)
    if "sparks_or_smoke" in flags and "burning_smell" in flags:
        flags.remove("burning_smell")
    texts = [SAFETY_NOTICES[f] for f in flags if f in SAFETY_NOTICES]
    return "\n\n".join(texts) if texts else None
