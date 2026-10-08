"""Anonymous usage analytics (agent/analytics.py) and the stats page. All offline."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from streamlit.testing.v1 import AppTest

from agent import analytics, config
from agent.state import LeadState

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc)
PII = ["Jane", "Doe", "555", "0123", "jane@example.com", "95616", "12 Elm St", "basement", "storm"]


def full_state(**overrides) -> LeadState:
    """A converted conversation full of personal details that must never be sent."""
    base = dict(
        messages=[HumanMessage("Water in my basement after the storm"), AIMessage("Is it still coming in?"),
                  HumanMessage("Jane Doe, 530-555-0123, jane@example.com, 12 Elm St, 95616"), AIMessage("All set")],
        category="water_damage", urgency="emergency", status="converted", zip="95616", name="Jane Doe",
        contact_phone="(530) 555-0123", contact_email="jane@example.com", address="12 Elm St",
        facts=["Water in basement after a storm"], problem_description="Storm water in the basement.",
        matched_providers=[{"name": "A"}, {"name": "B"}, {"name": "C"}], lead_id="abc12345",
    )
    return LeadState(**{**base, **overrides})


@pytest.fixture
def configured(monkeypatch):
    """Analytics switched on with fake settings (conftest normally turns it off)."""
    monkeypatch.delenv("ANALYTICS_DISABLED", raising=False)
    fake = {"SUPABASE_URL": "https://example.supabase.co/", "SUPABASE_SERVICE_KEY": "service-key"}
    monkeypatch.setattr(config, "get_secret", lambda name, default=None: fake.get(name, default))


# --- the payload ------------------------------------------------------------------


def test_payload_has_only_allowed_keys_and_no_personal_data():
    started = NOW - timedelta(minutes=2)
    payload = analytics.build_payload("conv-1", full_state(), started, [3.0, 4.0], "deployed")
    assert set(payload) == analytics.ALLOWED_KEYS == {
        "id", "started_at", "updated_at", "source", "turns", "status",
        "category", "urgency", "providers_matched", "avg_turn_seconds",
    }
    assert payload["turns"] == 2 and payload["providers_matched"] == 3 and payload["avg_turn_seconds"] == 3.5
    assert (payload["status"], payload["category"], payload["source"]) == ("converted", "water_damage", "deployed")
    dumped = json.dumps(payload)
    for secret in PII:
        assert secret not in dumped


def test_no_match_outcome():
    state = full_state(status="in_progress", last_route="no_providers", matched_providers=[])
    assert analytics.build_payload("c", state, NOW, [], "local")["status"] == "no_match"


# --- on / off ----------------------------------------------------------------------


def test_disabled_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ANALYTICS_DISABLED", raising=False)
    monkeypatch.setattr(config, "get_secret", lambda name, default=None: default)
    monkeypatch.setattr(analytics, "_post", lambda *a: pytest.fail("must not send"))
    assert not analytics.enabled()
    assert analytics.record_turn("c", full_state(), NOW, [1.0], "local") is None


def test_always_disabled_under_pytest(monkeypatch):
    fake = {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_SERVICE_KEY": "k"}
    monkeypatch.setattr(config, "get_secret", lambda name, default=None: fake.get(name, default))
    assert analytics.enabled() is False  # conftest sets ANALYTICS_DISABLED, even with keys present


def test_upsert_request_shape(configured, monkeypatch):
    sent = {}

    def fake_post(url, **kwargs):
        sent.update(url=url, **kwargs)
        return httpx.Response(201, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    analytics.record_turn("conv-1", full_state(), NOW, [2.0], "local").result(timeout=5)
    assert sent["url"] == "https://example.supabase.co/rest/v1/conversations"
    assert sent["params"] == {"on_conflict": "id"} and sent["timeout"] == 2.0
    assert "merge-duplicates" in sent["headers"]["Prefer"] and set(sent["json"]) == analytics.ALLOWED_KEYS


# --- failure isolation --------------------------------------------------------------


@pytest.mark.parametrize("error", [httpx.ConnectTimeout("slow"), httpx.ConnectError("down"), ValueError("bad")])
def test_supabase_errors_are_logged_never_raised(configured, monkeypatch, caplog, error):
    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(httpx, "post", broken)
    with caplog.at_level(logging.WARNING, logger="agent.analytics"):
        future = analytics.record_turn("conv-1", full_state(), NOW, [1.0], "local")
        future.result(timeout=5)  # finished without raising
    assert "analytics upsert failed" in caplog.text and "Jane" not in caplog.text


def test_http_error_status_is_swallowed(configured, monkeypatch, caplog):
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(500, request=httpx.Request("POST", url)))
    with caplog.at_level(logging.WARNING, logger="agent.analytics"):
        analytics.record_turn("c", full_state(), NOW, [1.0], "local").result(timeout=5)
    assert "500" in caplog.text


def test_record_turn_never_raises_even_if_payload_fails(configured, monkeypatch):
    monkeypatch.setattr(analytics, "build_payload", lambda *a: 1 / 0)
    assert analytics.record_turn("c", full_state(), NOW, [1.0], "local") is None


def test_chat_still_works_when_supabase_is_down(configured, monkeypatch, tmp_path):
    """Analytics on, every Supabase call fails: the reply still shows, no exception."""
    import agent.graph as graph_module
    import streamlit as st
    from tests.test_graph import streaming_llm

    calls = []

    def down(*args, **kwargs):
        calls.append(1)
        raise httpx.ConnectError("supabase is down")

    monkeypatch.setattr(httpx, "post", down)
    llm = streaming_llm(["How soon do you need someone?"],
                        extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Sink leaking"]}])
    monkeypatch.setattr(graph_module, "get_llm", lambda model=None: llm)
    monkeypatch.setattr(graph_module, "TRANSCRIPTS_DIR", tmp_path / "t")
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "t")
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    at.chat_input[0].set_value("my sink is leaking").run()
    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == "How soon do you need someone?"
    analytics._executor.submit(lambda: None).result(timeout=5)  # let the queued upsert run
    assert calls  # it really tried, and failed quietly


# --- stats ------------------------------------------------------------------------------


def row(status, minutes_ago=60, source="deployed", category="plumbing", turns=5, avg=3.0, day="2026-10-08", now=NOW):
    updated = now - timedelta(minutes=minutes_ago)
    return {"id": f"{status}-{minutes_ago}-{source}", "started_at": f"{day}T12:00:00+00:00",
            "updated_at": updated.isoformat(), "source": source, "turns": turns, "status": status,
            "category": category, "urgency": None, "providers_matched": 3, "avg_turn_seconds": avg}


def rows_at(now: datetime) -> list[dict]:
    return [
        row("converted", turns=5, avg=3.0, now=now),
        row("converted", turns=7, avg=5.0, category="electrical", now=now),
        row("declined", turns=8, avg=4.0, now=now),
        row("in_progress", minutes_ago=45, turns=2, avg=2.0, now=now),  # stale: abandoned
        row("in_progress", minutes_ago=5, turns=1, avg=10.0, now=now),  # still active: not eligible yet
        row("out_of_scope", turns=1, avg=2.0, category=None, now=now),
        row("no_match", turns=3, avg=3.0, day="2026-10-07", now=now),
        row("emergency", turns=1, avg=None, category=None, day="2026-10-07", now=now),
        row("converted", source="local", turns=4, avg=6.0, now=now),
    ]


ROWS = rows_at(NOW)


def test_stats_calculations_all_sources():
    s = analytics.summarize(ROWS, NOW)
    assert s["total"] == 9 and s["converted"] == 3
    # eligible: 3 converted + declined + abandoned (the active one, out_of_scope, no_match, emergency are out)
    assert s["eligible"] == 5 and s["conversion_rate"] == pytest.approx(3 / 5)
    assert s["raw_conversion_rate"] == pytest.approx(3 / 9)
    assert s["outcomes"] == {"converted": 3, "declined": 1, "abandoned": 1, "in_progress": 1,
                             "out_of_scope": 1, "no_match": 1, "emergency": 1}
    assert s["by_category"] == {"plumbing": 6, "electrical": 1, "unknown": 2}
    assert s["avg_turns_to_conversion"] == pytest.approx((5 + 7 + 4) / 3)
    weighted = (3*5 + 5*7 + 4*8 + 2*2 + 10*1 + 2*1 + 3*3 + 6*4) / (5 + 7 + 8 + 2 + 1 + 1 + 3 + 4)
    assert s["avg_turn_seconds"] == pytest.approx(weighted)
    assert s["p90_turn_seconds"] == 10.0  # nearest rank: ceil(0.9 * 8) = 8th of [2,2,3,3,4,5,6,10]
    assert s["per_day"] == {"2026-10-07": 2, "2026-10-08": 7}


def test_stats_source_filter_and_empty():
    deployed = analytics.summarize(ROWS, NOW, "deployed")
    assert deployed["total"] == 8 and deployed["converted"] == 2
    local = analytics.summarize(ROWS, NOW, "local")
    assert local["total"] == 1 and local["conversion_rate"] == 1.0
    empty = analytics.summarize([], NOW)
    assert empty["total"] == 0 and empty["conversion_rate"] is None and empty["p90_turn_seconds"] is None


def test_abandoned_after_30_minutes_only():
    assert analytics.effective_status(row("in_progress", minutes_ago=30), NOW) == "abandoned"
    assert analytics.effective_status(row("awaiting_confirm", minutes_ago=31), NOW) == "abandoned"
    assert analytics.effective_status(row("in_progress", minutes_ago=29), NOW) == "in_progress"
    assert analytics.effective_status(row("converted", minutes_ago=600), NOW) == "converted"


# --- stats page ---------------------------------------------------------------------------

STATS = str(ROOT / "pages" / "stats.py")


@pytest.fixture
def stats_page(monkeypatch):
    import streamlit as st

    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)  # only the test's secrets count, not a developer's .env

    live_rows = rows_at(datetime.now(timezone.utc))  # the page uses the real clock
    monkeypatch.setattr(analytics, "fetch_rows", lambda **kw: live_rows)
    st.cache_data.clear()

    def open_page(**secrets):
        at = AppTest.from_file(STATS, default_timeout=30)
        for k, v in secrets.items():
            at.secrets[k] = v
        return at.run()

    return open_page


def test_stats_page_is_behind_admin_password(stats_page):
    at = stats_page(ADMIN_PASSWORD="admin-pw")
    assert not at.exception
    assert at.text_input[0].label == "Admin password" and not at.metric  # nothing shown yet
    at.text_input[0].set_value("nope")
    at.button[0].click().run()
    assert at.error[0].value == "Wrong password." and not at.metric
    at.text_input[0].set_value("admin-pw")
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Conversations"] == "9" and metrics["Conversion rate"] == "60%"
    assert metrics["Raw conversion rate"] == "33%"
    assert "Anonymous outcome data only. No messages or contact details are stored." in [c.value for c in at.caption]
    at.radio[0].set_value("local").run()
    assert {m.label: m.value for m in at.metric}["Conversations"] == "1"


def test_stats_page_off_without_admin_password(stats_page):
    at = stats_page()
    assert not at.exception and "set ADMIN_PASSWORD" in at.error[0].value and not at.text_input


def test_stats_page_when_supabase_not_configured(monkeypatch):
    import streamlit as st

    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    st.cache_data.clear()  # real fetch_rows: analytics is off under pytest
    at = AppTest.from_file(STATS, default_timeout=30)
    at.secrets["ADMIN_PASSWORD"] = "pw"
    at.run()
    at.text_input[0].set_value("pw")
    at.button[0].click().run()
    assert not at.exception and "Analytics isn't configured" in at.info[0].value


def test_stats_page_hidden_from_app_navigation():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    settings = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert settings["client"]["showSidebarNavigation"] is False


def test_walking_away_at_consent_counts_as_abandoned_not_in_progress():
    rows = [row("awaiting_confirm", minutes_ago=45), row("awaiting_confirm", minutes_ago=5), row("converted")]
    s = analytics.summarize(rows, NOW)
    assert s["outcomes"] == {"abandoned": 1, "awaiting_confirm": 1, "converted": 1}
    assert s["eligible"] == 2  # the stale one counts against conversion; the active one isn't decided yet
    assert s["conversion_rate"] == pytest.approx(1 / 2)
