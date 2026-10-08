"""Scripted scenario suite against the real Anthropic API. Skipped unless RUN_LIVE=1.

    RUN_LIVE=1 pytest tests/test_live.py -s                     # all scenarios
    RUN_LIVE=1 pytest tests/test_live.py -s -k gas_leak         # one
    RUN_LIVE=1 SAVE_SAMPLES=1 pytest tests/test_live.py -s      # also save transcripts/leads to samples/
    RUN_LIVE=1 LIVE_REPEAT=3 pytest tests/test_live.py -s -k vague_opener   # stability: run 3x

No simulated LLM user and no judge: each scenario is a fixed script. The user
answers whatever the agent asks, keyed on route's decision ("urgency",
"detail:extent", "contact", ...); unscripted questions get
"I'm not sure." Optional `first_replies` are sent in order before the keyed
answers (e.g. changing the problem mid-conversation).

Each run prints every turn as it happens (with per-node latency), then a
summary table and a per-node latency table (also written to
samples/scenario_results.md). All user details are fake (555-01XX numbers).
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import pytest

from agent.config import AGENT_MODEL, ANTHROPIC_API_KEY, EXTRACT_MODEL, LEADS_DIR, PROJECT_ROOT
from agent.faithfulness import unsupported_claims
from agent.graph import build_graph, new_thread_id, run_turn
from agent.nodes import _text
from agent.timing import NodeTimer

pytestmark = pytest.mark.skipif(
    not (os.getenv("RUN_LIVE") and ANTHROPIC_API_KEY), reason="set RUN_LIVE=1 and ANTHROPIC_API_KEY"
)

MAX_TURNS = 12
DEFAULT_ANSWER = "I'm not sure."
CONSENT = "Yes, go ahead."
RESULTS_PATH = PROJECT_ROOT / "samples" / "scenario_results.md"
REPEAT = max(1, int(os.getenv("LIVE_REPEAT", "1")))  # run each selected scenario N times


@dataclass
class Scenario:
    name: str
    opener: str
    answers: dict[str, str] = field(default_factory=dict)
    expected_status: str = "converted"
    expected_category: str | None = None
    expected_urgency: str | None = None
    first_replies: list[str] = field(default_factory=list)  # sent in order before keyed answers
    never_asked: list[str] = field(default_factory=list)  # question types that must not be asked
    expect_repair_offer: bool | None = None  # wildlife: offer a separate repair request after the lead?


SCENARIOS = [
    Scenario(
        "basement_flooding",
        "Water started coming into my basement last night after the storm. I don't know who to call.",
        {
            "urgency": "As soon as possible, it's still coming in.",
            "detail:still_active": "Yes, it's still seeping in slowly.",
            "detail:extent": "About an inch of water across most of the basement floor.",
            "zip": "95616",
            "contact": "Jane Doe, 530-555-0123",
        },
        expected_category="water_damage",
        expected_urgency="emergency",
    ),
    Scenario(
        "sparking_outlet_single",
        "One of the outlets in my kitchen sparked when I plugged in the toaster.",
        {
            "urgency": "In the next day or two.",
            "detail:scope": "Just that one outlet, the rest of the kitchen works.",
            "detail:hazard_signs": "Just the one spark. The outlet isn't hot.",
            "detail:breaker": "The breaker didn't trip.",
            "zip": "95618",
            "contact": "Sam Lee, 916-555-0142",
        },
        expected_category="electrical",
        expected_urgency="within_48h",
    ),
    Scenario(
        "sparking_ongoing",
        "The outlet in my bedroom keeps sparking every time something is plugged in.",
        {
            "detail:scope": "Just that one outlet.",
            "detail:hazard_signs": "It keeps sparking, and the cover plate feels warm.",
            "detail:breaker": "The breaker hasn't tripped.",
            "zip": "95616",
            "contact": "Maria Lopez, 530-555-0156",
        },
        expected_category="electrical",
        expected_urgency="emergency",
    ),
    Scenario(
        "gas_leak",
        "I smell gas in my kitchen near the stove.",
        expected_status="emergency",
    ),
    Scenario(
        "raccoon_roof_damage",
        "I think there's a raccoon living in my attic. I hear it every night.",
        {
            "urgency": "This week would be good.",
            "detail:animal": "A raccoon, I've seen it on the roof at night.",
            "detail:location": "In the attic.",
            "detail:damage": "Yeah, it tore up a vent on the roof.",
            "zip": "95616",
            "contact": "Alex Kim, 530-555-0188",
        },
        expected_category="wildlife_removal",
        expect_repair_offer=True,
    ),
    Scenario(
        "raccoon_no_damage",
        "I think there's a raccoon living in my attic. I hear it every night.",
        {
            "urgency": "This week would be good.",
            "detail:animal": "A raccoon.",
            "detail:location": "In the attic.",
            "detail:damage": "No, I haven't noticed any damage.",
            "zip": "95618",
            "contact": "Dana Cruz, 530-555-0172",
        },
        expected_category="wildlife_removal",
        expect_repair_offer=False,
    ),
    Scenario(
        "power_outage_street",
        "The power is out on my whole street since this morning.",
        expected_status="out_of_scope",
    ),
    Scenario(
        "vague_opener",
        "Something's wrong with my house and I'm not sure who to call.",
        {
            "facts": "There's a brown water stain spreading on my ceiling, right under the upstairs bathroom.",
            "category": "There's a brown water stain spreading on my ceiling, right under the upstairs bathroom.",
            "clarify": "There's no water on the floor, just the stain, so probably something leaking from the bathroom.",
            "urgency": "Within the next few days would be good.",
            "zip": "95691",
            "contact": "Priya Shah, 916-555-0177",
        },
        expected_category="plumbing",
        expected_urgency="within_week",
    ),
    Scenario(
        "refuses_phone",
        "My kitchen sink drain is clogged and the water won't go down.",
        {
            "urgency": "Sometime this week.",
            "detail:fixture": "The kitchen sink.",
            "detail:symptom": "It's clogged, the water just sits there.",
            "zip": "95616",
            "contact": "I'm Chris Park. I'd rather not give my phone number, but you can email me at chris.park@example.com.",
            "contact_email": "chris.park@example.com",
        },
        expected_category="plumbing",
    ),
    Scenario(
        "refuses_all_contact",
        "My garbage disposal stopped working.",
        {
            "urgency": "No rush.",
            "detail:appliance": "It's an InSinkErator garbage disposal.",
            "detail:symptom": "It just hums and doesn't spin.",
            "zip": "95616",
            "contact": "I'm Dana. I'd rather not give my phone number.",
            "contact_email": "No, I don't want to share my email either.",
            "contact_required": "No thanks, I'll pass on that.",
        },
        expected_status="declined",
    ),
    Scenario(
        "changes_problem",
        "Water is dripping from my kitchen ceiling. I think a pipe is leaking.",
        {
            "urgency": "Sometime this week is fine.",
            "detail:active_leak": "Only when it rains.",
            "detail:roof_type": "Asphalt shingles.",
            "zip": "95616",
            "contact": "Jordan Lee, 530-555-0164",
        },
        first_replies=["Actually, I just checked and it's the roof. Rain is coming in through the roof above the kitchen."],
        expected_category="roofing",
    ),
    Scenario(
        "zip_first",
        "95618, my toilet won't stop running",
        {
            "urgency": "Sometime this week.",
            "detail:symptom": "It just keeps running after I flush.",
            "contact": "Taylor Nguyen, 530-555-0193",
        },
        expected_category="plumbing",
        expected_urgency="within_week",
        never_asked=["zip"],
    ),
    Scenario(
        "out_of_region",
        "My water heater is leaking. I'm in 90012.",
        expected_status="out_of_scope",
    ),
]

RESULTS: list[dict] = []
NODE_TIMES: dict[str, list[float]] = {}  # node -> every call's seconds, across all scenarios in this run
LLM_NODES = ["extract", "ask_next", "clarify", "summarize", "consent_reply"]


def timed_turn(graph, tid: str, n: int, text: str, node_times: dict[str, list[float]]):
    """Run one turn, print it immediately, and add this turn's node latencies to node_times."""
    print(f"\n--- turn {n} ---\n[user] {text}", flush=True)
    timer = NodeTimer()
    start = time.perf_counter()
    state = run_turn(graph, tid, text, source="test", callbacks=[timer])  # source: app history ignores it
    elapsed = time.perf_counter() - start
    for node, secs in timer.totals().items():
        node_times.setdefault(node, []).extend(secs)
    per_node = "  ".join(f"{node} {sum(secs):.1f}s" for node, secs in timer.totals().items() if sum(secs) >= 0.05)
    print(f"[route] {state.last_route}   ({elapsed:.1f}s: {per_node or 'no LLM'})", flush=True)
    print(f"[agent] {_text(state.messages[-1])}", flush=True)
    return state, elapsed


def save_sample(name: str, tid: str, state) -> None:
    samples = PROJECT_ROOT / "samples"
    samples.mkdir(exist_ok=True)
    (samples / f"{name}.transcript.json").write_text((PROJECT_ROOT / "transcripts" / f"{tid}.json").read_text())
    if state.lead_id:
        (samples / f"{name}.lead.json").write_text((LEADS_DIR / f"{state.lead_id}.json").read_text())


def _mark(expected, actual) -> str:
    if expected is None:
        return "—"
    return "✓" if expected == actual else f"✗ ({actual})"


def results_table(results: list[dict]) -> str:
    header = ("| scenario | expected status | actual status | turns | category | urgency | faithful | seconds "
              "| extract s | notes |")
    rows = [header, "|" + "---|" * 10]
    for r in results:
        rows.append(
            f"| {r['name']} | {r['expected_status']} | {r['status']} {'✓' if r['status_ok'] else '✗'} | "
            f"{r['turns']} | {r['category']} | {r['urgency']} | {r['faithful']} | {r['seconds']:.1f} | "
            f"{r['extract_seconds']:.1f} | {r['notes']} |"
        )
    passed = sum(r["passed"] for r in results)
    rows.append(f"\n**{passed}/{len(results)} scenarios passed** · replies `{AGENT_MODEL}` · extract "
                f"`{EXTRACT_MODEL}` · total {sum(r['seconds'] for r in results):.0f}s · {datetime.now():%Y-%m-%d %H:%M}")
    return "\n".join(rows)


def latency_table(node_times: dict[str, list[float]]) -> str:
    """Per-node latency across every turn of every scenario in this run."""
    rows = ["| node | model | calls | avg s | max s | total s |", "|---|---|---|---|---|---|"]
    for node in LLM_NODES + sorted(set(node_times) - set(LLM_NODES)):
        secs = node_times.get(node)
        if not secs:
            continue
        model = EXTRACT_MODEL if node == "extract" else AGENT_MODEL if node in LLM_NODES else "(code)"
        rows.append(f"| {node} | {model} | {len(secs)} | {sum(secs) / len(secs):.2f} | {max(secs):.2f} | "
                    f"{sum(secs):.1f} |")
    return "\n".join(rows)


@pytest.fixture(scope="module", autouse=True)
def summary():
    """After all scenarios in this run: print the table and write samples/scenario_results.md."""
    yield
    if not RESULTS:
        return
    table = results_table(RESULTS) + "\n\n## Latency per node\n\n" + latency_table(NODE_TIMES)
    print("\n\n========== SCENARIO SUMMARY ==========\n" + table)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text("# Scenario results\n\nScripted runs of `tests/test_live.py`.\n\n" + table + "\n")


RUNS = [(sc, i) for sc in SCENARIOS for i in range(1, REPEAT + 1)]


@pytest.mark.parametrize(
    "sc, run", RUNS, ids=[sc.name if REPEAT == 1 else f"{sc.name}-run{i}" for sc, i in RUNS]
)
def test_scenario(sc: Scenario, run: int):
    label = sc.name if REPEAT == 1 else f"{sc.name} (run {run}/{REPEAT})"
    print(f"\n========== {label} (replies={AGENT_MODEL}, extract={EXTRACT_MODEL}) ==========", flush=True)
    graph = build_graph()
    tid = new_thread_id()
    timings = []
    node_times: dict[str, list[float]] = {}
    scripted = list(sc.first_replies)

    state, t = timed_turn(graph, tid, 1, sc.opener, node_times)
    timings.append(t)
    while state.status in ("in_progress", "awaiting_confirm") and len(timings) < MAX_TURNS:
        if (state.last_route or "").startswith("llm_error"):
            break
        if state.status == "awaiting_confirm":
            reply = CONSENT
        elif scripted:
            reply = scripted.pop(0)
        else:
            reply = sc.answers.get(state.last_route.removeprefix("ask_next:"), DEFAULT_ANSWER)
        state, t = timed_turn(graph, tid, len(timings) + 1, reply, node_times)
        timings.append(t)

    faithful_problems = []
    if state.lead_id:
        lead = json.loads((LEADS_DIR / f"{state.lead_id}.json").read_text())
        faithful_problems = unsupported_claims(
            lead["problem_description"], lead["facts"], lead["category_details"], lead["owner_or_renter"]
        )
    asked_types = [a.split(":")[0] if a.startswith("detail") else a for a in state.asked]
    wrongly_asked = [q for q in sc.never_asked if q in state.asked or q in asked_types]

    notes = []
    if state.llm_errors:
        notes.append(f"llm_error: {state.llm_errors[-1][:60]}")
    if wrongly_asked:
        notes.append(f"asked {wrongly_asked}")
    if faithful_problems:
        notes.append("faithfulness flagged")
    if state.status not in ("converted",) and state.last_route:
        notes.append(f"last_route={state.last_route}")
    if state.out_of_scope_reason:
        notes.append(f"reason={state.out_of_scope_reason}")

    repair_ok = sc.expect_repair_offer is None or bool(state.repair_offer) == sc.expect_repair_offer
    if not repair_ok:
        notes.append(f"repair_offer={state.repair_offer!r}")
    status_ok = state.status == sc.expected_status
    category_ok = sc.expected_category is None or state.category == sc.expected_category
    urgency_ok = sc.expected_urgency is None or state.urgency == sc.expected_urgency
    result = {
        "name": label,
        "expected_status": sc.expected_status,
        "status": state.status,
        "status_ok": status_ok,
        "turns": len(timings),
        "category": _mark(sc.expected_category, state.category),
        "urgency": _mark(sc.expected_urgency, state.urgency),
        "faithful": ("✓" if not faithful_problems else "✗") if state.lead_id else "—",
        "seconds": sum(timings),
        "extract_seconds": sum(node_times.get("extract", [])),
        "notes": "; ".join(notes),
        "passed": status_ok and category_ok and urgency_ok and repair_ok and not wrongly_asked and not faithful_problems,
    }
    RESULTS.append(result)
    for node, secs in node_times.items():
        NODE_TIMES.setdefault(node, []).extend(secs)

    print(f"\n--- RESULT --- thread={tid} status={state.status} turns={len(timings)} total={sum(timings):.1f}s")
    print(f"category={state.category} urgency={state.urgency} asked={state.asked}")
    print(f"facts={state.facts}\ndetails={state.category_details}")
    if state.problem_description:
        print(f"description={state.problem_description!r}  faithfulness={faithful_problems or 'ok'}")
    if state.lead_id:
        print("\n--- LEAD ---\n" + (LEADS_DIR / f"{state.lead_id}.json").read_text())
    if os.getenv("SAVE_SAMPLES") and run == 1:
        save_sample(sc.name, tid, state)

    assert result["passed"], f"{sc.name}: {result}"
