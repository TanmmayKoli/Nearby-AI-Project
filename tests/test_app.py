"""Headless smoke tests for the Streamlit page (AppTest), with a fake LLM."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent.graph as graph_module
from agent import config
from tests.test_graph import streaming_llm

APP = str(Path(__file__).resolve().parent.parent / "app.py")

FULL = {"category": "water_damage", "category_confidence": 0.95, "urgency": "within_week",
        "new_facts": ["Water in basement"], "category_details": {"still_active": "no", "extent": "small"},
        "zip": "95616", "name": "Jane Doe", "contact_phone": "5305550123"}


@pytest.fixture
def app(monkeypatch, tmp_path):
    for name in ("APP_PASSWORD", "DEPLOYED"):  # a developer's .env shouldn't change these tests
        monkeypatch.delenv(name, raising=False)

    def run(texts, secrets=None, **scripted):
        llm = streaming_llm(texts, **scripted)
        monkeypatch.setattr(graph_module, "get_llm", lambda model=None: llm)
        monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
        monkeypatch.setattr(graph_module, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
        monkeypatch.setattr(graph_module, "LEADS_DIR", tmp_path / "leads")
        import streamlit as st

        st.cache_resource.clear()
        at = AppTest.from_file(APP, default_timeout=30)
        for key, value in (secrets or {}).items():
            at.secrets[key] = value
        return at.run()

    return run


def test_empty_state_header_and_sidebar(app):
    at = app([])
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Describe it" in text and "A few questions" in text and "Get matched" in text
    assert [b.label for b in at.sidebar.button] == [
        "New conversation", "Basement flooding", "Sparking outlet", "Raccoon in attic", "Something's wrong",
    ]


def test_opener_streams_reply_and_brain_panel_works(app):
    at = app(["Is water still coming in right now?"],
             extractions=[{"category": "water_damage", "category_confidence": 0.95, "urgency": "emergency",
                           "new_facts": ["Water in basement after storm"]}])
    at.sidebar.button[1].click().run()  # Basement flooding
    assert not at.exception
    chat = [(m.name, m.markdown[0].value) for m in at.chat_message]
    assert chat[0][0] == "user" and chat[-1] == ("assistant", "Is water still coming in right now?")
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert ("Category", "Water damage restoration") in [(m.label, m.value) for m in at.metric]


def test_confirm_shows_provider_cards_with_call_button(app):
    """Uses the real providers.json (water damage near 95616)."""
    at = app(["Water in the basement."], extractions=[FULL])
    at.chat_input[0].set_value("everything at once").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Here's the request I'll send" in text and "Is it OK to share" in text
    assert "SERVPRO of Davis" in text and "\n1. " not in text  # providers as cards, not a text list
    calls = [b.proto.url for b in at.get("link_button") if b.proto.label == "Call"]
    assert len(calls) >= 3 and all(u.startswith("tel:+1") and len(u) == 16 for u in calls)


def test_assets_and_theme():
    """Every image the app references exists and is valid SVG; theme is Option B."""
    import re as _re
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    import xml.dom.minidom

    root = Path(APP).parent
    referenced = set(_re.findall(r'"([a-z_]+\.svg)"', (root / "app.py").read_text()))
    assert {"logo.svg", "logo_icon.svg", "hero.svg", "step_describe.svg"} <= referenced
    for name in referenced:
        xml.dom.minidom.parse(str(root / "assets" / name))
    theme = tomllib.loads((root / ".streamlit" / "config.toml").read_text())["theme"]
    assert theme["primaryColor"] == "#1E3A8A" and theme["backgroundColor"] == "#FBFAF7"


def test_header_images_render(app):
    at = app([])
    assert not at.exception
    assert len(at.get("image")) == 4  # hero + 3 step icons


def test_code_built_reply_is_shown_gas(app):
    """Gas reply is built by code (nothing streams): the bubble must show it."""
    from agent.categories import GAS_EMERGENCY_MESSAGE

    at = app([])
    at.chat_input[0].set_value("I smell gas in my kitchen").run()
    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == GAS_EMERGENCY_MESSAGE


def test_provider_cards_have_website_and_maps_links_and_more_pros(app):
    """Real providers.json, water damage near 95616."""
    at = app(["Water in the basement."], extractions=[FULL])
    at.chat_input[0].set_value("everything at once").run()
    assert not at.exception
    links = [(b.proto.label, b.proto.url) for b in at.get("link_button")]
    maps = [u for label, u in links if label == "Google Maps"]
    assert len(maps) >= 3 and all(u.startswith("https://www.google.com/maps/place/?q=place_id:") for u in maps)
    assert any(label == "Website" and u.startswith("http") for label, u in links)
    exp = at.expander[0]
    assert exp.label.startswith("See more pros near you")
    assert "won't receive your request" in " ".join(c.value for c in exp.caption)
    calls = [u for label, u in links if label == "Call"]
    assert len(calls) > 3  # 3 main cards + extras inside the expander


def test_past_conversation_is_read_only_with_back_button(app):
    at = app(["Is water still coming in right now?"],
             extractions=[{"category": "water_damage", "category_confidence": 0.95, "urgency": "emergency",
                           "new_facts": ["Water in basement after storm"]}])
    at.sidebar.button[1].click().run()  # Basement flooding -> saved with source="app"
    at.sidebar.button[0].click().run()  # New conversation: the first one is now "past"
    assert not at.exception
    past = [b for b in at.sidebar.button if b.key and b.key.startswith("past-")]
    assert len(past) == 1 and "Water damage restoration" in past[0].label and "in_progress" in past[0].label
    past[0].click().run()
    assert not at.exception
    assert "Viewing a past conversation" in at.info[0].value
    assert at.chat_input[0].proto.disabled
    assert at.chat_message[-1].markdown[0].value == "Is water still coming in right now?"
    next(b for b in at.button if b.label == "Back to current conversation").click().run()
    assert not at.exception and not at.chat_input[0].proto.disabled and len(at.info) == 0


def test_password_gate_blocks_until_correct_password(app):
    at = app([], secrets={"APP_PASSWORD": "letmein"})
    assert not at.exception
    assert len(at.chat_input) == 0 and len(at.sidebar.button) == 0  # nothing behind the gate renders
    at.text_input[0].set_value("wrong")
    at.button[0].click().run()
    assert at.error[0].value == "Wrong password." and len(at.chat_input) == 0
    at.text_input[0].set_value("letmein")
    at.button[0].click().run()
    assert not at.exception and len(at.chat_input) == 1 and len(at.text_input) == 0


def test_no_password_set_skips_gate(app):
    at = app([])
    assert len(at.text_input) == 0 and len(at.chat_input) == 1


def test_deployed_hides_history_and_writes_nothing(app, tmp_path):
    at = app(["Water in the basement."], secrets={"DEPLOYED": "true"}, extractions=[FULL, {}], consents=["yes"])
    assert "Past conversations" not in [h.value for h in at.sidebar.subheader]
    at.chat_input[0].set_value("everything at once").run()
    at.chat_input[0].set_value("yes, send it").run()
    assert not at.exception
    assert "demo: not saved" in at.success[0].value
    assert not (tmp_path / "leads").exists() and not (tmp_path / "transcripts").exists()
    at.sidebar.button[0].click().run()  # New conversation: still no history list
    assert not [b for b in at.sidebar.button if b.key and b.key.startswith("past-")]


def test_local_mode_shows_history_heading(app):
    at = app([])
    assert "Past conversations" in [h.value for h in at.sidebar.subheader]
