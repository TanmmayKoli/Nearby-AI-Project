"""Read saved transcripts for the app's "Past conversations" list (read-only)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.categories import CATEGORIES

OPENER_CHARS = 40


def _opener(messages: list[dict]) -> str:
    first = next((m["content"] for m in messages if m.get("role") == "user"), "")
    first = " ".join(first.split())
    return first if len(first) <= OPENER_CHARS else first[: OPENER_CHARS - 1].rstrip() + "…"


def list_conversations(transcripts_dir: Path, source: str = "app", limit: int | None = None) -> list[dict[str, Any]]:
    """Summaries of saved conversations from `source`, newest first.

    Each: thread_id, saved_at (datetime), category (label or "Unknown"),
    status, opener (first ~40 chars of the user's first message). Transcripts
    from other sources (tests) or untagged/old/unreadable files are skipped.
    """
    if not transcripts_dir.exists():
        return []
    items = []
    for path in transcripts_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text())
            if record.get("source") != source:
                continue
            state = record.get("state", {})
            category = state.get("category")
            items.append(
                {
                    "thread_id": record["thread_id"],
                    "saved_at": datetime.fromisoformat(record["saved_at"]),
                    "category": CATEGORIES[category].label if category in CATEGORIES else "Unknown",
                    "status": state.get("status", "unknown"),
                    "opener": _opener(record.get("messages", [])),
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue  # unreadable or malformed transcript: just don't list it
    items.sort(key=lambda c: c["saved_at"], reverse=True)
    return items[:limit] if limit else items


def load_conversation(transcripts_dir: Path, thread_id: str) -> dict[str, Any] | None:
    """The full saved transcript (messages + final state), or None if missing/unreadable."""
    path = transcripts_dir / f"{thread_id}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
