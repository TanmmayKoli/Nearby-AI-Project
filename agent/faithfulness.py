"""Cheap, deterministic faithfulness check for problem_description.

Not a proof of faithfulness, a tripwire for the two failure modes seen in
testing:
1. Scope / negative claims the user never made ("limited to that single
   outlet", "no other outlets affected"). Flagged when a guard word appears
   in a sentence but not anywhere in the source (facts + details).
2. Sentences mostly made of content the source doesn't mention.

3. Ownership words ("homeowner", "renter", ...) when owner_or_renter is unknown:
   the summary should say "the customer".

Source = facts + category_details keys AND values ("The source is the storm"
is supported by source=storm). Used by the scripted scenario suite
(tests/test_live.py) on every converted lead.
"""

import re

OWNERSHIP_WORDS = {"homeowner", "homeowners", "owner", "owners", "renter", "renters", "tenant", "tenants", "landlord"}
GUARD_WORDS = {"only", "just", "limited", "single", "solely", "entire", "whole", "other", "none", "nothing", "isolated"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "that", "this", "there", "their", "they",
    "has", "have", "had", "as", "into", "when", "after", "since", "also", "which", "while", "about",
    "some", "any", "not", "no", "so", "than", "then", "up", "out", "over", "under", "what", "who",
    # reporting words a summary may legitimately add ("homeowner" is NOT here: see OWNERSHIP_WORDS)
    "user", "customer", "reports", "reported", "says", "said", "states", "suspects",
    "believes", "notes", "describes", "mentioned",
}
MIN_SUPPORTED_RATIO = 0.6  # share of a sentence's content words that must appear in the source


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def unsupported_claims(
    description: str,
    facts: list[str],
    details: dict[str, str] | None = None,
    owner_or_renter: str | None = None,
) -> list[str]:
    """Reasons a description goes beyond its source. Empty list = looks faithful."""
    details = details or {}
    detail_keys = [k.replace("_", " ") for k in details]
    source_text = " ".join(facts + detail_keys + list(details.values()))
    source_words = set(_words(source_text))
    source_stems = {_stem(w) for w in source_words}

    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", description.strip()):
        words = _words(sentence)
        if owner_or_renter is None:
            ownership = [w for w in words if w in OWNERSHIP_WORDS]
            if ownership:
                problems.append(f"ownership word(s) {ownership} but owner_or_renter is unknown: {sentence!r}")
                continue
        guards = [w for w in words if w in GUARD_WORDS and w not in source_words]
        if guards:
            problems.append(f"scope/negative word(s) {guards} not in source: {sentence!r}")
            continue
        content = [w for w in words if w not in STOPWORDS and len(w) > 2]
        if content:
            supported = sum(1 for w in content if _stem(w) in source_stems)
            if supported / len(content) < MIN_SUPPORTED_RATIO:
                problems.append(f"mostly unsupported ({supported}/{len(content)} words in source): {sentence!r}")
    return problems
