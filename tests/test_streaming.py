"""Regression tests for the empty-bubble bug (thread 31a4ec76322e).

Cause: when claude-sonnet-5-5 streams a reply with a thinking block, LangChain
merges the chunks into a MIXED list ['', {thinking}, 'text...']. The old text
helper only read dict blocks, so ask_next stored '' and the app showed empty
bubbles. Only the streaming path was affected; run_turn (no streaming) was fine.
"""

import pytest
from langchain_core.messages import AIMessage

from agent import nodes
from agent.nodes import content_text
from tests import fake_anthropic
from tests.test_graph import FakeLLM, make_graph, turn


@pytest.mark.parametrize(
    "content, expected",
    [
        ("plain string", "plain string"),
        ([{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}], "Hello there"),
        (["", {"type": "thinking", "thinking": "hmm", "signature": "x"}, "How soon?"], "How soon?"),  # merged stream
        ([{"type": "thinking", "thinking": "hmm"}], ""),  # thinking only: no user-visible text
        ([{"type": "tool_use", "id": "t", "name": "x", "input": {}}, {"type": "text", "text": "ok"}], "ok"),
        ([" leading", " spaces "], " leading spaces "),  # not stripped: tokens keep their spaces
        ([], ""),
    ],
)
def test_content_text(content, expected):
    assert content_text(content) == expected


def test_empty_llm_reply_becomes_apology_not_empty_bubble(tmp_path):
    llm = FakeLLM(extractions=[{"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Leak"]}], texts=[""])
    graph, tid = make_graph(llm, tmp_path)
    s = turn(graph, tid, "my sink leaks", tmp_path)
    assert s.messages[-1].content == nodes.LLM_ERROR_REPLY
    assert "empty reply" in s.llm_errors[-1]


@pytest.fixture
def real_client_llm(monkeypatch):
    """The real ChatAnthropic client pointed at the local fake API."""
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = fake_anthropic.start()
    yield lambda: ChatAnthropic(
        model="claude-sonnet-5-5", api_key="test", max_retries=0, max_tokens=200,
        anthropic_api_url=f"http://127.0.0.1:{srv.server_port}",
    )
    srv.shutdown()


@pytest.mark.parametrize("with_thinking", [False, True])
def test_streamed_reply_is_stored_with_real_client(real_client_llm, monkeypatch, tmp_path, with_thinking):
    from agent.graph import build_graph, get_state, new_thread_id, run_turn, stream_turn

    monkeypatch.setattr(fake_anthropic, "WITH_THINKING", with_thinking)
    graph = build_graph(llm=real_client_llm(), leads_dir=tmp_path / "leads")

    tid = new_thread_id()
    streamed = "".join(stream_turn(graph, tid, "my water heater is dripping", transcripts_dir=None))
    stored = get_state(graph, tid).messages[-1].content
    assert stored == fake_anthropic.QUESTION
    assert streamed.strip() == stored  # what streamed is what's saved
    assert "{" not in streamed  # extract's JSON never streams

    plain = run_turn(graph, new_thread_id(), "my water heater is dripping", transcripts_dir=None)
    assert plain.messages[-1].content == stored  # same result with or without streaming
