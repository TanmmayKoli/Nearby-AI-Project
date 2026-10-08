"""Settings: Streamlit secrets first (deployed), then environment / .env (local).

Empty values fall back to the default: a line like `AGENT_MODEL=` in .env sets
an empty string, which should still mean "use the default".
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_secret(name: str, default: str | None = None) -> str | None:
    """st.secrets[name] if present (Streamlit Cloud, .streamlit/secrets.toml),
    else the environment variable (.env locally), else `default`."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]) or default
    except Exception:  # no secrets.toml, or not running under Streamlit
        pass
    return os.getenv(name) or default


def get_flag(name: str) -> bool:
    return (get_secret(name) or "").strip().lower() in {"1", "true", "yes", "on"}


AGENT_MODEL = get_secret("AGENT_MODEL", "claude-sonnet-5-5")  # everything the user reads
EXTRACT_MODEL = get_secret("EXTRACT_MODEL", "claude-haiku-4-5-20251001")  # extract node only (speed)
SIMULATOR_MODEL = get_secret("SIMULATOR_MODEL", "claude-haiku-5-5")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
GOOGLE_PLACES_API_KEY = get_secret("GOOGLE_PLACES_API_KEY")
# APP_PASSWORD and DEPLOYED are read by app.py on every run, not here at import
# time, so they always reflect the current secrets.

# Per-request timeout and SDK retries. Worst case per LLM call is about
# (1 + LLM_MAX_RETRIES) * LLM_TIMEOUT_SECONDS before the turn gives up gracefully.
LLM_TIMEOUT_SECONDS = 45
LLM_MAX_RETRIES = 2

# Provider ranking: score = bayesian_rating - penalty_per_mile * miles.
# Urgent jobs weight distance much more heavily (a nearby crew matters more than
# a slightly better rating). E.g. at 0.15, 10 extra miles costs 1.5 stars.
DISTANCE_PENALTY_PER_MILE = {
    "emergency": 0.15,
    "within_48h": 0.10,
    "within_week": 0.03,
    "flexible": 0.02,
}
DEFAULT_DISTANCE_PENALTY_PER_MILE = 0.03  # urgency unknown

LEADS_DIR = PROJECT_ROOT / "leads"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
