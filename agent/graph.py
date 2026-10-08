"""Graph wiring. One invocation per user turn; MemorySaver keeps each thread.

Per turn:

    START ─(entry_route)─┬─ closed ─► END                       (conversation already ended)
                         └─ safety_check ─(after_safety)─┬─ emergency_response ─► END   (gas: no lead)
                                                         ├─ consent_reply ─┬─ create_lead ─► END
                                                         │                 ├─ extract (user wants changes)
                                                         │                 └─ END (declined)
                                                         └─ extract ─ route ─┬─ ask_next ─► END
                                                                             ├─ clarify ─► END
                                                                             ├─ out_of_scope ─► END
                                                                             ├─ wildlife_pivot ─► END  (redirect + "any damage?")
                                                                             ├─ emergency_response ─► END
                                                                             ├─ no_contact ─► END
                                                                             └─ match_providers ─ summarize ─ confirm ─► END

safety_check is keyword-based (no LLM) so a gas smell is handled even if the API
is down; the extractor's safety_flags are a second net, checked in route.

If an LLM call fails after retries (timeout, API error), that node replies with
a short apology, changes nothing else, and the turn ends (`llm_error:*`).
"""

import json
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.config import AGENT_MODEL, ANTHROPIC_API_KEY, LEADS_DIR, LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS, TRANSCRIPTS_DIR
from agent.state import LeadState
from providers.match import match_providers


def get_llm(model: str = AGENT_MODEL) -> ChatAnthropic:
    # No temperature: newer models (e.g. claude-sonnet-5-5) only accept the default.
    # Pass the key explicitly when it came from st.secrets (not in the env);
    # otherwise ChatAnthropic reads ANTHROPIC_API_KEY from the env itself.
    key = {"api_key": ANTHROPIC_API_KEY} if ANTHROPIC_API_KEY else {}
    return ChatAnthropic(
        model=model, max_tokens=1024, timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES, **key
    )


# Defaults resolved at call time (not import time) so tests can redirect them.
_DEFAULT: Any = object()


def build_graph(
    llm: BaseChatModel | None = None,
    checkpointer: Any = None,
    leads_dir: Path | None = _DEFAULT,  # None = validate the lead but don't write it
    matcher: Callable[..., list[dict[str, Any]]] = match_providers,
):
    llm = llm or get_llm()
    leads_dir = LEADS_DIR if leads_dir is _DEFAULT else leads_dir
    g = StateGraph(LeadState)

    g.add_node("safety_check", nodes.safety_check)
    g.add_node("emergency_response", nodes.emergency_response)
    g.add_node("extract", partial(nodes.extract, llm=llm))
    g.add_node("route", nodes.route)
    g.add_node("ask_next", partial(nodes.ask_next, llm=llm))
    g.add_node("clarify", partial(nodes.clarify, llm=llm))
    g.add_node("out_of_scope", nodes.out_of_scope)
    g.add_node("wildlife_pivot", nodes.wildlife_pivot)
    g.add_node("no_contact", partial(nodes.no_contact, matcher=matcher))
    g.add_node("match_providers", partial(nodes.match_providers_node, matcher=matcher))
    g.add_node("summarize", partial(nodes.summarize, llm=llm))
    g.add_node("confirm", nodes.confirm)
    g.add_node("consent_reply", partial(nodes.consent_reply, llm=llm))
    g.add_node("create_lead", partial(nodes.create_lead, leads_dir=leads_dir))
    g.add_node("closed", nodes.closed)

    g.add_conditional_edges(START, nodes.entry_route, ["safety_check", "closed"])
    g.add_conditional_edges("safety_check", nodes.after_safety, ["emergency_response", "consent_reply", "extract"])
    g.add_conditional_edges("extract", nodes.after_llm_node("route"), {"route": "route", "end": END})
    g.add_conditional_edges(
        "route",
        nodes.after_route,
        ["ask_next", "clarify", "out_of_scope", "wildlife_pivot", "emergency_response", "no_contact", "match_providers"],
    )
    for terminal in ["ask_next", "clarify", "out_of_scope", "wildlife_pivot", "emergency_response", "no_contact"]:
        g.add_edge(terminal, END)
    g.add_conditional_edges("match_providers", nodes.after_match, {"summarize": "summarize", "end": END})
    g.add_conditional_edges("summarize", nodes.after_llm_node("confirm"), {"confirm": "confirm", "end": END})
    g.add_edge("confirm", END)
    g.add_conditional_edges(
        "consent_reply", nodes.after_consent, {"create_lead": "create_lead", "extract": "extract", "end": END}
    )
    g.add_edge("create_lead", END)
    g.add_edge("closed", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def new_thread_id() -> str:
    return uuid.uuid4().hex[:12]


def get_state(graph, thread_id: str) -> LeadState:
    values = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    return LeadState(**values) if values else LeadState()


def run_turn(
    graph, thread_id: str, text: str, transcripts_dir: Path | None = _DEFAULT, source: str | None = None
) -> LeadState:
    """Send one user message, return the new state, and save the transcript
    (tagged with `source`, e.g. "app" or "test")."""
    transcripts_dir = TRANSCRIPTS_DIR if transcripts_dir is _DEFAULT else transcripts_dir
    graph.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread_id}})
    state = get_state(graph, thread_id)
    if transcripts_dir is not None:
        save_transcript(state, thread_id, transcripts_dir, source)
    return state


# Nodes whose LLM output is the user-facing reply (everything else is either
# internal, like the extractor's JSON, or a fixed template shown after the turn).
STREAMED_NODES = {"ask_next", "clarify"}


def stream_turn(
    graph, thread_id: str, text: str, transcripts_dir: Path | None = _DEFAULT, source: str | None = None
) -> Iterator[str]:
    """Same as run_turn, but yields the reply text as it's generated (for the UI).

    Yields a pending safety notice first (as soon as route sets it), then the
    question's tokens. Fixed replies (confirm, redirects, safety messages) aren't
    streamed; the caller re-renders from state when the turn ends. The graph runs
    exactly as with run_turn, so behavior is identical.
    """
    transcripts_dir = TRANSCRIPTS_DIR if transcripts_dir is _DEFAULT else transcripts_dir
    config = {"configurable": {"thread_id": thread_id}}
    for mode, payload in graph.stream(
        {"messages": [HumanMessage(text)]}, config, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            route_update = payload.get("route") or {}
            if route_update.get("pending_notice"):
                yield route_update["pending_notice"] + "\n\n"
        else:
            chunk, meta = payload
            if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") in STREAMED_NODES:
                if token := nodes.content_text(chunk.content):  # text only; never thinking blocks
                    yield token
    if transcripts_dir is not None:
        save_transcript(get_state(graph, thread_id), thread_id, transcripts_dir, source)


def save_transcript(
    state: LeadState, thread_id: str, transcripts_dir: Path = TRANSCRIPTS_DIR, source: str | None = None
) -> Path:
    """transcripts/{thread_id}.json: messages + final state. Saved every turn, so
    abandoned conversations are kept too. `source` ("app", "test", ...) lets the
    app's "Past conversations" list show only real app conversations."""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    path = transcripts_dir / f"{thread_id}.json"
    record = {
        "thread_id": thread_id,
        "source": source,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "messages": [
            {"role": "user" if m.type == "human" else "assistant", "content": nodes._text(m)}
            for m in state.messages
        ],
        "state": state.model_dump(mode="json", exclude={"messages"}),
    }
    path.write_text(json.dumps(record, indent=2))
    return path
