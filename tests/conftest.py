"""Test-session setup.

LangSmith tracing is for real runs only. The offline suite uses fake models, so
tracing it would just upload junk runs (or hang where the network is blocked).
Set here, before agent.config loads .env (load_dotenv never overrides a variable
that's already set). Live runs (RUN_LIVE=1) keep whatever .env says.
"""

import os

if not os.getenv("RUN_LIVE"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
