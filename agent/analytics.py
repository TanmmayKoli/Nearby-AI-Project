"""Anonymous usage analytics in Supabase, for measuring real conversion.

One row per conversation in the `conversations` table, upserted after every
turn. A row holds outcome data only (ALLOWED_KEYS): never message text, names,
phone numbers, emails, zips, addresses or facts. Its id is a random uuid made
just for analytics (not the chat's thread id), and no IP or user identifier is
sent.

Fire-and-forget: rows go out from ONE background thread (in order, so an older
turn can never overwrite a newer one) with a 2-second timeout. Every error is
caught and logged as a warning, so the chat never waits on, or fails because
of, analytics. Disabled when SUPABASE_URL / SUPABASE_SERVICE_KEY aren't set,
and always disabled under pytest (tests/conftest.py sets ANALYTICS_DISABLED).

Talks to Supabase's REST API (PostgREST) with httpx, which is already a
dependency; the full supabase client isn't needed for one upsert and one select.
"""

import logging
import os
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent import config
from agent.state import LeadState

log = logging.getLogger(__name__)

TABLE = "conversations"
WRITE_TIMEOUT_SECONDS = 2.0
READ_TIMEOUT_SECONDS = 10.0
ALLOWED_KEYS = frozenset(
    {
        "id", "started_at", "updated_at", "source", "turns", "status",
        "category", "urgency", "providers_matched", "avg_turn_seconds",
    }
)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analytics")


class AnalyticsNotConfigured(RuntimeError):
    pass


def _settings() -> tuple[str, str] | None:
    """(url, key), or None when analytics is off (not configured, or under tests)."""
    if os.getenv("ANALYTICS_DISABLED"):
        return None
    url = config.get_secret("SUPABASE_URL")
    key = config.get_secret("SUPABASE_SERVICE_KEY")
    return (url.rstrip("/"), key) if url and key else None


def enabled() -> bool:
    return _settings() is not None


def _headers(key: str) -> dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


# --- writing -------------------------------------------------------------------


def new_conversation_id() -> str:
    return str(uuid.uuid4())


def outcome(state: LeadState) -> str:
    """The row's status. "no_match" (no provider near that zip) is the agent's
    `no_providers` route; everything else is the conversation status."""
    return "no_match" if state.last_route == "no_providers" else state.status


def build_payload(
    conversation_id: str, state: LeadState, started_at: datetime, turn_seconds: list[float], source: str
) -> dict[str, Any]:
    """The only data that leaves the app. Built field by field from outcome data;
    nothing the user typed is ever included."""
    payload = {
        "id": conversation_id,
        "started_at": started_at.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "turns": sum(1 for m in state.messages if m.type == "human"),
        "status": outcome(state),
        "category": state.category,
        "urgency": state.urgency,
        "providers_matched": len(state.matched_providers),
        "avg_turn_seconds": round(sum(turn_seconds) / len(turn_seconds), 2) if turn_seconds else None,
    }
    assert set(payload) == ALLOWED_KEYS
    return payload


def _post(settings: tuple[str, str], payload: dict[str, Any]) -> None:
    url, key = settings
    response = httpx.post(
        f"{url}/rest/v1/{TABLE}",
        params={"on_conflict": "id"},
        json=payload,
        headers={**_headers(key), "Prefer": "resolution=merge-duplicates,return=minimal"},
        timeout=WRITE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def _send(settings: tuple[str, str], payload: dict[str, Any]) -> None:
    try:
        _post(settings, payload)
    except Exception as e:  # never raise into the app
        log.warning("analytics upsert failed: %s: %s", type(e).__name__, e)


def record_turn(
    conversation_id: str, state: LeadState, started_at: datetime, turn_seconds: list[float], source: str
) -> Future | None:
    """Queue an upsert of this conversation's row. Returns at once (None when
    analytics is off). Never raises."""
    try:
        settings = _settings()
        if settings is None:
            return None
        return _executor.submit(_send, settings, build_payload(conversation_id, state, started_at, turn_seconds, source))
    except Exception as e:
        log.warning("analytics skipped: %s: %s", type(e).__name__, e)
        return None


# --- reading (stats page) ---------------------------------------------------------


def fetch_rows(page_size: int = 1000, max_rows: int = 50_000) -> list[dict[str, Any]]:
    """Every row, paged (Supabase returns at most 1000 per request)."""
    settings = _settings()
    if settings is None:
        raise AnalyticsNotConfigured("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set")
    url, key = settings
    rows: list[dict[str, Any]] = []
    while len(rows) < max_rows:
        response = httpx.get(
            f"{url}/rest/v1/{TABLE}",
            params={"select": ",".join(sorted(ALLOWED_KEYS)), "order": "started_at.asc",
                    "limit": page_size, "offset": len(rows)},
            headers=_headers(key),
            timeout=READ_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        batch = response.json()
        rows.extend(batch)
        if len(batch) < page_size:
            break
    return rows


# --- stats ---------------------------------------------------------------------------

ABANDON_AFTER = timedelta(minutes=30)
OPEN_STATUSES = {"in_progress", "awaiting_confirm"}
# Not counted in the conversion denominator: there was never a lead to win.
INELIGIBLE = {"out_of_scope", "no_match", "emergency"}


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def effective_status(row: dict[str, Any], now: datetime) -> str:
    """An open conversation with no update for 30+ minutes counts as abandoned."""
    if row["status"] in OPEN_STATUSES and now - _time(row["updated_at"]) >= ABANDON_AFTER:
        return "abandoned"
    return row["status"]


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, -(-9 * len(ordered) // 10) - 1)]  # nearest-rank 90th percentile


def summarize(rows: list[dict[str, Any]], now: datetime, source: str = "all") -> dict[str, Any]:
    """Everything the stats page shows, from raw rows.

    Conversion = converted / eligible, where eligible leaves out conversations
    that could never produce a lead (out of scope, no provider match, gas
    emergency) and ones still in progress (outcome not known yet). Raw
    conversion = converted / all conversations.
    """
    if source != "all":
        rows = [r for r in rows if r.get("source") == source]
    statuses = [effective_status(r, now) for r in rows]
    converted = statuses.count("converted")
    eligible = [s for s in statuses if s not in INELIGIBLE and s not in OPEN_STATUSES]
    converted_turns = [r["turns"] for r, s in zip(rows, statuses) if s == "converted" and r.get("turns")]
    timed = [(r["avg_turn_seconds"], r.get("turns") or 0) for r in rows if r.get("avg_turn_seconds") is not None]
    total_turns = sum(t for _, t in timed)
    return {
        "total": len(rows),
        "converted": converted,
        "eligible": len(eligible),
        "conversion_rate": converted / len(eligible) if eligible else None,
        "raw_conversion_rate": converted / len(rows) if rows else None,
        "outcomes": dict(Counter(statuses).most_common()),
        "by_category": dict(Counter(r.get("category") or "unknown" for r in rows).most_common()),
        "avg_turns_to_conversion": sum(converted_turns) / len(converted_turns) if converted_turns else None,
        # turn-weighted mean of each conversation's average; p90 across conversations
        "avg_turn_seconds": sum(a * t for a, t in timed) / total_turns if total_turns else None,
        "p90_turn_seconds": _p90([a for a, _ in timed]),
        "per_day": dict(sorted(Counter(_time(r["started_at"]).date().isoformat() for r in rows).items())),
    }
