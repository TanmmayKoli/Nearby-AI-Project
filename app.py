"""Streamlit chat page: the main UI.

Run:  streamlit run app.py
"""

import hmac
import re
from pathlib import Path

import streamlit as st

from agent import config
from agent.categories import CATEGORIES
from agent.graph import build_graph, get_state, new_thread_id, stream_turn
from agent.history import list_conversations, load_conversation
from agent.nodes import PROVIDERS_HEADING, _text
from agent.state import LeadState
from providers.match import match_providers

SAMPLE_OPENERS = {
    "Basement flooding": "Water started coming into my basement last night after the storm. I don't know who to call.",
    "Sparking outlet": "One of the outlets in my kitchen sparked when I plugged in the toaster.",
    "Raccoon in attic": "I think there's a raccoon living in my attic. I hear it every night.",
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
def get_graph(deployed: bool):
    """One graph (and one MemorySaver) for the whole server process.
    Deployed: leads are validated and shown, but not written to disk."""
    return build_graph(leads_dir=None) if deployed else build_graph()


graph = get_graph(DEPLOYED)
TRANSCRIPTS_DIR = None if DEPLOYED else config.TRANSCRIPTS_DIR

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()


def start_new_conversation(opener: str | None = None) -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.pending = opener
    st.session_state.viewing = None


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


def stream_reply(text: str) -> None:
    """Show the user's message and stream the reply.

    The streamed text is only a live preview. The conversation history is
    always rebuilt from the graph's stored state on rerun, so what stays on
    screen is exactly what the agent saved. Replies built by code (confirm,
    safety, out-of-scope) don't stream: if nothing streamed, the final message
    is read from state and shown directly.
    """
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        thinking = st.empty()
        thinking.markdown("_Thinking…_")
        streamed: list[str] = []

        def tokens():
            for token in stream_turn(
                graph, st.session_state.thread_id, text, transcripts_dir=TRANSCRIPTS_DIR, source="app"
            ):
                if not streamed:
                    thinking.empty()
                streamed.append(token)
                yield token

        st.write_stream(tokens())
        if not "".join(streamed).strip():
            final = get_state(graph, st.session_state.thread_id).messages[-1]
            with thinking.container():
                st.markdown(_text(final))
    st.rerun()


chat_col, brain_col = st.columns([3, 2]) if show_brain else (st.container(), None)

with chat_col:
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
    if state.status == "converted" and state.lead_id:
        if DEPLOYED:
            st.success(f"Lead `{state.lead_id}` created (demo: not saved or sent anywhere).")
        else:
            st.success(f"Lead created: `leads/{state.lead_id}.json`")
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
    if state.pivot_offered:
        st.caption(f"Wildlife pivot offered (out-of-scope reason: {state.out_of_scope_reason})")
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
