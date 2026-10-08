"""Streamlit chat page: the main UI.

Run:  streamlit run app.py
"""

import hmac
import os
import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from langsmith import tracing_context

from agent import analytics, config
from agent.categories import CATEGORIES
from agent.graph import build_graph, get_state, new_thread_id, stream_turn
from agent.history import list_conversations, load_conversation
from agent.dispatch import provider_view
from agent.nodes import CLOSED_STATUSES, PROVIDERS_HEADING, REPAIR_WHO, _text, build_lead
from agent.state import LeadState
from providers.match import match_providers

SAMPLE_OPENERS = {
    "Basement flooding": "Water started coming into my basement last night after the storm. I don't know who to call.",
    "Sparking outlet": "One of the outlets in my kitchen sparked when I plugged in the toaster.",
    "Something's wrong": "Something's wrong with my house and I'm not sure who to call.",
}

ASSETS = Path(__file__).parent / "assets"  # simple SVGs drawn for this project (no licensing)
HOW_IT_WORKS = [
    ("step_describe.svg", "1. Describe it", "Tell me what's going on, in your own words."),
    ("step_questions.svg", "2. A few questions", "I'll ask only what a pro needs to help."),
    ("step_match.svg", "3. Get matched", "You'll see local pros near you, with ratings and phone numbers."),
]
ASSISTANT_AVATAR = "🏠"
MORE_PROS_LIMIT = 8  # ranked matches to fetch for "See more pros" (top 3 still get the request)
PAST_CONVERSATIONS_SHOWN = 10
MAPS_URL = "https://www.google.com/maps/place/?q=place_id:{place_id}"
DEMO_NOTICE = (
    "Demo: requests are not actually sent to these businesses. Please use fake contact details. "
    "We record anonymous usage stats (no messages or contact info)."
)
# Quick replies: tappable answers under the latest question, only where the
# answers are fixed. Clicking sends the label as the user's message.
URGENCY_REPLIES = ["Right away", "In a day or two", "This week", "Flexible"]
CONSENT_REPLIES = ["Yes, share my info", "No thanks"]
DETAIL_REPLIES = ["Not sure"]
CATEGORY_ICONS = {
    "water_damage": "💧", "plumbing": "🚰", "electrical": "⚡", "hvac": "🌡️",
    "roofing": "🏠", "pest_control": "🐜", "appliance_repair": "🧺", "handyman": "🔧",
}

st.set_page_config(page_title="Home Services Assistant", page_icon="🏠", layout="wide")

# Read on every run (not cached) so they always match the current secrets.
# DEPLOYED: public demo mode. No past conversations, nothing written to disk
# (visitors' names and phone numbers stay in this server's memory only).
DEPLOYED = config.get_flag("DEPLOYED")
APP_PASSWORD = config.get_secret("APP_PASSWORD")
if DEPLOYED:  # LangSmith tracing is for local debugging only: never send visitors' chats
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"] = "false"


def password_gate() -> None:
    """If APP_PASSWORD is set, show a password screen until it's entered."""
    if not APP_PASSWORD or st.session_state.get("authenticated"):
        return
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.image(str(ASSETS / "logo.svg"), width=180)
        with st.form("login"):
            entered = st.text_input("Password", type="password")
            if st.form_submit_button("Enter", type="primary"):
                if hmac.compare_digest(entered, APP_PASSWORD):
                    st.session_state.authenticated = True
                    st.rerun()
                st.error("Wrong password.")
    st.stop()


password_gate()


@st.cache_resource
def shared_graph():
    """Local: one graph (and one MemorySaver) for the whole server process."""
    return build_graph()


def get_graph():
    """Deployed: each browser session gets its own graph and MemorySaver, kept in
    st.session_state, so a visitor's conversation and lead live only in their
    session and are gone when it ends. Nothing is written to disk (leads_dir=None,
    no transcripts), and the final message says nothing was sent (demo=True)."""
    if not DEPLOYED:
        return shared_graph()
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(leads_dir=None, demo=True)
    return st.session_state.graph


graph = get_graph()
TRANSCRIPTS_DIR = None if DEPLOYED else config.TRANSCRIPTS_DIR

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()


def start_new_conversation(opener: str | None = None, carry: dict | None = None) -> None:
    """New thread. `carry`: zip/contact the user chose to reuse, set with their
    first message so the agent doesn't ask for them again."""
    st.session_state.thread_id = new_thread_id()
    st.session_state.pending = opener
    st.session_state.viewing = None
    st.session_state.carry = carry


def send_quick_reply(text: str) -> None:
    st.session_state.pending = text  # sent on this rerun, same path as typed text


def view_past(thread_id: str | None) -> None:
    """Open a saved conversation read-only (None = back to the current one)."""
    st.session_state.viewing = thread_id


viewing = st.session_state.get("viewing")


# --- Sidebar --------------------------------------------------------------------

st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "logo_icon.svg"))

with st.sidebar:
    st.caption("Davis · Woodland · West Sacramento · Sacramento")
    st.button("New conversation", on_click=start_new_conversation, width="stretch")
    st.subheader("Try an opener")
    for label, text in SAMPLE_OPENERS.items():
        st.button(label, on_click=start_new_conversation, args=(text,), width="stretch")
    if TRANSCRIPTS_DIR is not None:  # hidden when deployed: transcripts are other visitors'
        st.divider()
        st.subheader("Past conversations")
        past = [
            c for c in list_conversations(TRANSCRIPTS_DIR, source="app")
            if c["thread_id"] != st.session_state.thread_id
        ][:PAST_CONVERSATIONS_SHOWN]
        if not past:
            st.caption("None yet.")
        for c in past:
            label = f"{c['saved_at'].astimezone():%b %-d, %-I:%M %p} · {c['category']} · {c['status']}  \n“{c['opener']}”"
            st.button(label, key=f"past-{c['thread_id']}", on_click=view_past, args=(c["thread_id"],), width="stretch",
                      type="primary" if c["thread_id"] == viewing else "secondary")
    st.divider()
    show_brain = st.toggle("Show agent brain", value=False)
    st.caption(f"thread `{viewing or st.session_state.thread_id}`")


# --- Chat ---------------------------------------------------------------------------

# Viewing a past conversation is read-only: input disabled, nothing is sent.
user_input = st.chat_input(
    "Viewing a past conversation (read-only)" if viewing else "Describe what's going on at your home",
    disabled=bool(viewing),
)
pending = st.session_state.pop("pending", None)
message = None if viewing else (user_input or pending)

past_record = load_conversation(TRANSCRIPTS_DIR, viewing) if viewing and TRANSCRIPTS_DIR else None
if past_record:
    state = LeadState(**past_record["state"])
    history = [(m["role"], m["content"]) for m in past_record["messages"]]
else:
    state = get_state(graph, st.session_state.thread_id)
    history = [("user" if m.type == "human" else "assistant", _text(m)) for m in state.messages]


def render_header() -> None:
    text_col, art_col = st.columns([3, 2], vertical_alignment="center")
    with text_col:
        st.markdown("### Tell me what's going on at your home, and I'll connect you with a local pro.")
        st.caption("Serving Davis, Woodland, West Sacramento and Sacramento.")
    art_col.image(str(ASSETS / "hero.svg"), width="stretch")
    for col, (icon, title, text) in zip(st.columns(3), HOW_IT_WORKS):
        with col:
            st.image(str(ASSETS / icon), width=44)
            st.markdown(f"**{title}**  \n{text}")


def provider_links(p: dict) -> list[tuple[str, str]]:
    """(label, url) for a provider's Call / Website / Google Maps buttons."""
    links = []
    if p.get("phone"):
        links.append(("Call", "tel:+1" + re.sub(r"\D", "", p["phone"])[-10:]))
    if p.get("website"):
        links.append(("Website", p["website"]))
    if p.get("place_id"):
        links.append(("Google Maps", MAPS_URL.format(place_id=p["place_id"])))
    return links


def render_provider_cards(providers: list[dict], compact: bool = False) -> None:
    """Bordered cards. Link buttons open in a new tab (tel: opens the dialer)."""
    for p in providers:
        with st.container(border=True):
            rating = f"{p['rating']:.1f}★ ({p['review_count']} reviews)" if p.get("rating") else "No rating yet"
            details = f"**{p['name']}**  \n{p['city']} · {rating} · {p['distance_miles']:.1f} mi"
            if compact:
                st.markdown(details)
            else:
                icon, info = st.columns([0.6, 6.4], vertical_alignment="center")
                icon.markdown(f"## {CATEGORY_ICONS.get(p.get('category'), '🛠️')}")
                info.markdown(details)
            links = provider_links(p)
            for col, (label, url) in zip(st.columns(3), links):
                col.link_button(label, url, width="stretch", type="primary" if label == "Call" else "secondary")


def more_providers(s: LeadState) -> list[dict]:
    """Display-only extra matches (same filters and scoring) beyond the ones that
    get the request. Not stored in state or the lead."""
    if not (s.category and s.zip and s.matched_providers):
        return []
    shown = {p.get("place_id") or p["name"] for p in s.matched_providers}
    ranked = match_providers(s.category, s.zip, n=MORE_PROS_LIMIT, urgency=s.urgency)
    return [p for p in ranked if (p.get("place_id") or p["name"]) not in shown]


def render_assistant_message(text: str, providers: list[dict] | None) -> None:
    """Plain markdown, except the latest confirm message: its provider list is
    shown as cards (the lead summary text above it stays as-is)."""
    if not providers or PROVIDERS_HEADING not in text:
        st.markdown(text)
        return
    summary, rest = text.split(PROVIDERS_HEADING, 1)
    question = rest.strip().rsplit("\n\n", 1)[-1]  # the consent question after the list
    st.markdown(summary)
    st.markdown(f"**{PROVIDERS_HEADING}**")
    render_provider_cards(providers)
    extras = more_providers(state)
    if extras:
        with st.expander(f"See more pros near you ({len(extras)})"):
            st.caption("These pros won't receive your request, but you can contact them directly.")
            render_provider_cards(extras, compact=True)
    st.markdown(question)


def quick_replies(s: LeadState) -> list[str]:
    """Answers to offer for the latest assistant message. Clarify questions are
    free-form (the LLM words a question from the category rule), so they get none."""
    if s.status == "awaiting_confirm":
        return CONSENT_REPLIES
    if s.status != "in_progress":
        return []
    route = s.last_route or ""
    if route == "ask_next:urgency":
        return URGENCY_REPLIES
    if route.startswith("ask_next:detail:"):
        return DETAIL_REPLIES
    return []


def render_quick_replies(s: LeadState) -> None:
    """A wrapping row of buttons (stacks onto new lines on narrow phone screens).
    Keys include the turn so a stale row never fires twice."""
    replies = quick_replies(s)
    if not replies:
        return
    with st.container(horizontal=True, gap="small"):
        for i, label in enumerate(replies):
            st.button(label, key=f"qr-{st.session_state.thread_id}-{s.turn_count}-{i}",
                      on_click=send_quick_reply, args=(label,), width="content")


def render_provider_view(s: LeadState) -> None:
    """The lead as a provider would receive it, built from the Lead object only
    (the same text a real dispatch step would send: agent.dispatch)."""
    v = provider_view(build_lead(s, s.lead_id))
    with st.expander("What the provider receives"):
        with st.container(border=True):
            st.markdown(f"**{v.header}**")
            st.markdown(v.description)
            if v.details:
                st.markdown("  \n".join(f"{label}: {value}" for label, value in v.details))
            st.markdown("  \n".join([f"**Customer:** {v.customer}", v.contact, *v.extras]))
            st.caption(v.footer)
            with st.container(horizontal=True, gap="small"):
                st.button("Accept", key=f"pv-accept-{s.lead_id}", disabled=True, width="content")
                st.button("Decline", key=f"pv-decline-{s.lead_id}", disabled=True, width="content")
            st.caption("Preview only: these buttons don't do anything.")


CARRY_FIELDS = ["zip", "name", "contact_phone", "contact_email"]


def carry_fields(s: LeadState) -> dict:
    """Zip and contact from an ended conversation (an out-of-area zip isn't reused)."""
    carry = {f: getattr(s, f) for f in CARRY_FIELDS if getattr(s, f)}
    if s.out_of_scope_reason == "out_of_area":
        carry.pop("zip", None)
    return carry


def repair_opener(s: LeadState) -> str:
    """First message of the follow-up repair request (wildlife lead with damage)."""
    animal = str(s.category_details.get("animal") or "").strip()
    who = f"the {animal}" if animal and animal.lower() != "unknown" else "the animal"
    pro = REPAIR_WHO.get(s.repair_category, "someone")
    return f"{who.capitalize()} is being handled, but I need {pro} to fix the {s.repair_offer} it caused."


def render_next_steps(s: LeadState) -> None:
    """After any ending: start a new request, reusing zip/contact only if the user
    ticks the box. A wildlife lead with damage also gets a repair button. The
    ended conversation (and its lead) is never touched."""
    carry = carry_fields(s)
    reuse = bool(carry) and st.checkbox("Reuse my zip code and contact info", key=f"reuse-{st.session_state.thread_id}")
    chosen = carry if reuse else None
    with st.container(horizontal=True, gap="small"):
        if s.status == "converted" and s.repair_offer:
            st.button("Yes, find a repair pro", key=f"repair-{st.session_state.thread_id}", type="primary",
                      on_click=start_new_conversation, args=(repair_opener(s), chosen), width="content")
        st.button("Start a new request", key=f"new-{st.session_state.thread_id}",
                  on_click=start_new_conversation, args=(None, chosen), width="content")


def record_analytics(turn_seconds: float) -> None:
    """Anonymous outcome row for this conversation (agent/analytics.py). Each
    conversation gets its own random analytics id, unrelated to the thread id.
    Fire-and-forget: never blocks or breaks the chat."""
    try:
        tracked = st.session_state.setdefault("analytics", {})
        conv = tracked.setdefault(
            st.session_state.thread_id,
            {"id": analytics.new_conversation_id(), "started_at": datetime.now(timezone.utc), "turn_seconds": []},
        )
        conv["turn_seconds"].append(turn_seconds)
        state = get_state(graph, st.session_state.thread_id)
        analytics.record_turn(conv["id"], state, conv["started_at"], conv["turn_seconds"],
                              "deployed" if DEPLOYED else "local")
    except Exception:  # analytics must never break the chat
        pass


def stream_reply(text: str) -> None:
    """Show the user's message and stream the reply.

    The streamed text is only a live preview. The conversation history is
    always rebuilt from the graph's stored state on rerun, so what stays on
    screen is exactly what the agent saved. Replies built by code (confirm,
    safety, out-of-scope) don't stream: if nothing streamed, the final message
    is read from state and shown directly.
    """
    started = time.perf_counter()
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        thinking = st.empty()
        thinking.markdown("_Thinking…_")
        streamed: list[str] = []

        initial = st.session_state.pop("carry", None)  # only with a new request's first message

        def tokens():
            for token in stream_turn(
                graph, st.session_state.thread_id, text, transcripts_dir=TRANSCRIPTS_DIR, source="app",
                initial=initial,
            ):
                if not streamed:
                    thinking.empty()
                streamed.append(token)
                yield token

        # Deployed: tracing off even if LangSmith env vars were set by mistake.
        with tracing_context(enabled=False) if DEPLOYED else nullcontext():
            st.write_stream(tokens())
        if not "".join(streamed).strip():
            final = get_state(graph, st.session_state.thread_id).messages[-1]
            with thinking.container():
                st.markdown(_text(final))
    record_analytics(time.perf_counter() - started)
    st.rerun()


chat_col, brain_col = st.columns([3, 2]) if show_brain else (st.container(), None)

with chat_col:
    if DEPLOYED:
        st.info(DEMO_NOTICE, icon="🧪")
    if viewing:
        banner, back = st.columns([4, 1.4], vertical_alignment="center")
        banner.info("Viewing a past conversation (read-only)." if past_record else "That conversation couldn't be loaded.")
        back.button("Back to current conversation", on_click=view_past, args=(None,), width="stretch")
    if not history and not message and not viewing:
        render_header()
    # Provider cards go with the most recent confirm message only (older ones
    # may have had different matches before the user changed something).
    last_confirm = max(
        (i for i, (role, text) in enumerate(history) if role == "assistant" and PROVIDERS_HEADING in text), default=None
    )
    for i, (role, text) in enumerate(history):
        if role == "user":
            with st.chat_message("user"):
                st.markdown(text)
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                render_assistant_message(text, state.matched_providers if i == last_confirm else None)
    # Only under the latest message, and gone as soon as something is sent.
    if history and history[-1][0] == "assistant" and not viewing and not message:
        render_quick_replies(state)
    if state.status == "converted" and state.lead_id:
        if DEPLOYED:
            st.success(f"Lead `{state.lead_id}` created (demo: not saved or sent anywhere).")
        else:
            st.success(f"Lead created: `leads/{state.lead_id}.json`")
    if state.status == "converted" and state.lead_id:
        render_provider_view(state)
    if state.status in CLOSED_STATUSES and history and not viewing and not message:
        render_next_steps(state)
    if message:
        stream_reply(message)


def render_brain() -> None:
    st.subheader("Agent brain")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", state.status)
    c2.metric("Turns", state.turn_count)
    label = CATEGORIES[state.category].label if state.category else "unknown"
    c3.metric("Category", label, f"confidence {state.category_confidence:.2f}" if state.category else None, delta_color="off")

    st.markdown(f"**Last route decision:** `{state.last_route or '-'}`")
    if state.category_locked:
        st.caption(f"Category accepted after one clarify question: {state.category_locked}")
    if state.declined:
        st.markdown(f"**Declined to share:** {', '.join(state.declined)}")
    if state.repair_offer:
        st.caption(f"Repair request offered after the lead: {state.repair_offer} ({state.repair_category})")
    if state.cause_unknown:
        st.caption("User doesn't know the cause: cause questions skipped")
    st.markdown(f"**Missing required:** {', '.join(state.missing_fields) or 'none'}")

    st.markdown("**Lead fields**")
    fields = {
        "urgency": state.urgency, "zip": state.zip, "name": state.name,
        "phone": state.contact_phone, "email": state.contact_email, "address": state.address,
        "property_type": state.property_type, "owner_or_renter": state.owner_or_renter,
        "availability": state.availability, "consent_to_share": state.consent_to_share,
        "problem_description": state.problem_description,
    }
    st.table({"field": list(fields), "value": ["-" if v is None else str(v) for v in fields.values()]})

    st.markdown("**Facts**")
    st.markdown("\n".join(f"- {f}" for f in state.facts) or "_none yet_")
    if state.category_details:
        st.markdown("**Category details**")
        st.json(state.category_details)
    if state.safety_flags:
        st.markdown(f"**Safety flags:** {', '.join(state.safety_flags)} (warned: {', '.join(state.warned) or 'none'})")

    st.markdown("**Matched providers**")
    if state.matched_providers:
        st.table(
            {
                "name": [p["name"] for p in state.matched_providers],
                "city": [p["city"] for p in state.matched_providers],
                "rating": [f"{p['rating']} ({p['review_count']})" for p in state.matched_providers],
                "miles": [f"{p['distance_miles']:.1f}" for p in state.matched_providers],
            }
        )
    else:
        st.markdown("_not matched yet_")
    if state.extraction_errors:
        with st.expander(f"Extraction errors ({len(state.extraction_errors)})"):
            for e in state.extraction_errors:
                st.code(e)


if brain_col is not None:
    with brain_col:
        render_brain()
