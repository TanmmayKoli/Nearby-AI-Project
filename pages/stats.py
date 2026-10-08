"""Admin stats page (anonymous outcome data only). Gated by ADMIN_PASSWORD.

Not linked from the app: Streamlit's page navigation is hidden in
.streamlit/config.toml (client.showSidebarNavigation = false). Open it at /stats.
"""

import hmac
from datetime import datetime, timezone

import streamlit as st

from agent import analytics, config

st.set_page_config(page_title="Stats", page_icon="📊", layout="wide")

PRIVACY_NOTE = "Anonymous outcome data only. No messages or contact details are stored."


def admin_gate() -> None:
    password = config.get_secret("ADMIN_PASSWORD")
    if not password:
        st.error("The stats page is off: set ADMIN_PASSWORD to use it.")
        st.stop()
    if st.session_state.get("admin"):
        return
    with st.form("admin_login"):
        entered = st.text_input("Admin password", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if hmac.compare_digest(entered, password):
                st.session_state.admin = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


@st.cache_data(ttl=60, show_spinner="Loading stats…")
def load_rows() -> list[dict]:
    return analytics.fetch_rows()


def pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def num(value: float | None, unit: str = "") -> str:
    return "–" if value is None else f"{value:.1f}{unit}"


admin_gate()
st.title("Usage stats")
st.caption(PRIVACY_NOTE)

try:
    rows = load_rows()
except analytics.AnalyticsNotConfigured:
    st.info("Analytics isn't configured: set SUPABASE_URL and SUPABASE_SERVICE_KEY.")
    st.stop()
except Exception as e:
    st.error(f"Couldn't load stats from Supabase ({type(e).__name__}).")
    st.stop()

source = st.radio("Source", ["all", "deployed", "local"], horizontal=True)
s = analytics.summarize(rows, datetime.now(timezone.utc), source)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversations", s["total"])
c2.metric("Conversion rate", pct(s["conversion_rate"]), help="converted ÷ eligible. Eligible leaves out out-of-scope, "
          "no provider match, gas emergencies, and conversations still in progress.")
c3.metric("Raw conversion rate", pct(s["raw_conversion_rate"]), help="converted ÷ all conversations")
c4.metric("Avg turns to conversion", num(s["avg_turns_to_conversion"]))
c5, c6, c7, _ = st.columns(4)
c5.metric("Converted / eligible", f"{s['converted']} / {s['eligible']}")
c6.metric("Avg turn latency", num(s["avg_turn_seconds"], " s"), help="turn-weighted average")
c7.metric("p90 turn latency", num(s["p90_turn_seconds"], " s"), help="90th percentile of per-conversation averages")
st.caption("Conversations still open with no update for 30+ minutes count as abandoned.")

left, right = st.columns(2)
with left:
    st.subheader("Outcomes")
    st.table({"outcome": list(s["outcomes"]), "conversations": list(s["outcomes"].values())})
with right:
    st.subheader("By category")
    st.table({"category": list(s["by_category"]), "conversations": list(s["by_category"].values())})

st.subheader("Conversations per day (UTC)")
if s["per_day"]:
    st.bar_chart({"conversations": s["per_day"]})
else:
    st.caption("No conversations yet.")
