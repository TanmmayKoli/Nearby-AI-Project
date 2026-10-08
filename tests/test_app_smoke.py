"""Startup smoke tests: app.py must load the way Streamlit Cloud loads it.

Runs offline in CI: no API keys, no network (any socket connection fails the
test), no LLM calls. Loading the page builds the graph but never calls the
model; the network guard proves it. If startup ever needs the LLM or the
internet, these tests fail instead of the deployed app.

Also imports every module directly, so an ImportError in a module the page
doesn't load (e.g. providers.fetch) is caught too.
"""

import importlib
import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
OPENERS = ["Basement flooding", "Sparking outlet", "Something's wrong"]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No keys, no network, nothing written to the real leads/transcripts folders."""
    for name in ("ANTHROPIC_API_KEY", "APP_PASSWORD", "DEPLOYED", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def no_network(*args, **kwargs):
        raise AssertionError("app startup tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    import agent.config as config
    import agent.graph as graph

    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(graph, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(graph, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(graph, "LEADS_DIR", tmp_path / "leads")

    import streamlit as st

    st.cache_resource.clear()  # build the graph fresh, the way a new deploy does


def test_app_starts_offline(offline):
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception, at.exception
    assert len(at.chat_input) == 1 and not at.chat_input[0].proto.disabled
    labels = [b.label for b in at.sidebar.button]
    assert labels[0] == "New conversation" and all(o in labels for o in OPENERS)


def test_deployed_app_shows_password_screen_then_app(offline):
    at = AppTest.from_file(APP)
    at.secrets["DEPLOYED"] = "true"
    at.secrets["APP_PASSWORD"] = "smoke-test"
    at.run(timeout=30)
    assert not at.exception, at.exception
    assert at.text_input[0].label == "Password" and len(at.chat_input) == 0  # nothing behind the gate

    at.text_input[0].set_value("smoke-test")
    at.button[0].click().run(timeout=30)
    assert not at.exception, at.exception
    assert len(at.chat_input) == 1 and "not actually sent" in at.info[0].value


@pytest.mark.parametrize(
    "module",
    sorted(
        ".".join(p.relative_to(ROOT).with_suffix("").parts)
        for folder in ("agent", "providers")
        for p in (ROOT / folder).glob("*.py")
    ),
)
def test_every_module_imports(module):
    importlib.import_module(module)
