"""Test-session setup. Also: analytics is always off under pytest.

LangSmith tracing is for real runs only. The offline suite uses fake models, so
tracing it would just upload junk runs (or hang where the network is blocked).
Set here, before agent.config loads .env (load_dotenv never overrides a variable
that's already set). Live runs (RUN_LIVE=1) keep whatever .env says.
"""

import os

# Analytics never writes rows from a test run (offline or live).
os.environ["ANALYTICS_DISABLED"] = "1"

if not os.getenv("RUN_LIVE"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
