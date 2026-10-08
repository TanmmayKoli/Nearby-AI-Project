# Home Services Lead Agent — Project Context

## The assignment
Take-home for a job interview (~10 hours, 3-day window). Build an agent that helps a user find the right home service provider. The user describes a home problem, has a conversation with the agent, and the conversation ends with a **dispatchable lead** that could be sent to a **real service provider in the Davis–Sacramento, CA region**.

Example opening message: "Water started coming into my basement last night after the storm. I don't know who to call."

The company may spot-check providers to confirm they exist, so **never invent providers**. They may test this with real users in production.

**The two metrics they care about:**
- **Conversion rate**: what share of conversations end with a dispatchable lead.
- **Lead quality**: would a real provider accept and act on the lead?

These trade off. Every extra question improves quality but risks drop-off. Design decisions should manage that tradeoff explicitly.

Visual polish is NOT evaluated. A working agent, real data, and evidence that it works are what matter.

## Deliverables
- The working agent (code + README) with real provider data.
- **No demo video.** The demo is a live walkthrough on the interview call, using the Streamlit chat page with the "Show agent brain" panel to explain decisions as they happen.
- Saved sample transcripts and leads, plus eval results.

## Working style
- Tanmmay is the decision-maker; architecture decisions are made in a separate planning chat. Follow this doc. If something here seems wrong or underspecified, **ask before deviating**.
- Build in the phases below. **Stop after each phase**: summarize what you built, explain key decisions, and say how to run or test it. Tanmmay needs to understand and present every part of this.
- Prefer simple, readable code over clever code. Small functions, type hints, short docstrings.

## Definitions

**Lead**: a structured handoff record a provider could act on cold, without reading the chat. The conversation is how the lead gets collected, not the lead itself.

**Dispatchable**: all required core fields are present and validated, the user has consented to share their contact info, and at least one real matched provider serves that category and area.

## Lead schema — universal core fields

| Field | Required? | Usually inferred or asked? | Example |
|---|---|---|---|
| category | ✅ | Inferred | water_damage |
| problem_description | ✅ | Inferred (summary of the user's words) | "Water entering basement since storm, ~1 inch" |
| urgency | ✅ | Inferred, ask if unclear | emergency / within_48h / within_week / flexible |
| zip | ✅ | Asked | 95616 |
| name | ✅ | Asked, with contact | Jane D. |
| contact_phone | ✅ (or email) | Asked, **last** | (530) 555-0123 |
| contact_email | Optional | Asked if no phone | — |
| address | Optional | Asked after zip if willing | 123 Example St |
| property_type | Optional | Inferred | home / business |
| owner_or_renter | Optional | Asked if relevant | owner |
| availability | Optional | Asked | "home all day" |
| safety_flags | System-set | Inferred | water_near_electrical |
| category_details | Varies | Mostly inferred | per category table below |
| matched_providers | System-set | From provider data | [Provider A, B] |
| consent_to_share | ✅ System-set | Asked at confirm | true |

Principles:
- **Infer before asking.** Extract everything possible from what the user already said. Only ask about gaps.
- **Contact info comes last**, after the user is invested (same as Thumbtack/Angi).
- **Never fabricate field values.** If the user didn't say it, it stays empty.

## Categories

| Category | Detail questions (2–3) | Emergency triggers | Common confusion |
|---|---|---|---|
| water_damage | Still actively coming in? · How much water / area? · Source (storm, pipe, appliance, unknown)? | Active flooding, water near electrical, sewage | plumbing (source vs. cleanup) |
| plumbing | What's affected (toilet, sink, water heater, pipe, drain)? · Leaking, clogged, or no water? · Can you shut off the water? | Burst pipe, can't shut off water, sewage backup, no water | water_damage |
| electrical | One room/outlet or whole house? · Burning smell, sparks, hot outlets? · Breaker tripping? | Burning smell, sparks, smoke, hot outlets, water near electrical | Utility outage (PG&E/SMUD, not an electrician) |
| hvac | Heating or cooling? · Dead, or weak/noisy/leaking? · System type/age if known? | No heat in freezing weather, no AC in extreme heat with vulnerable people, gas smell | electrical (thermostat/breaker) |
| roofing | Active leak inside? · Cause (storm, age, tree)? · Roof type if known? | Active leak during rain, large hole, tree on roof | water_damage (interior cleanup) |
| pest_control | Which pest? · Where/how widespread? · How long? | Wasp/bee nest near entry, rodents in living areas with kids | Wildlife removal (raccoons, possums) |
| appliance_repair | Which appliance/brand? · What's it doing? · Gas or electric? | Gas smell, burning smell, heavy leaking | plumbing (washer/dishwasher leaks), hvac (water heaters) |
| handyman | What task (repair/install/assemble/paint)? · Which area? · One task or several? | Rarely urgent; broken lock/door that won't close | Licensed trades; electrical/plumbing beyond minor fixes needs a licensed pro |

Store this table as data in `agent/categories.py`, not hardcoded into prompts, so it's easy to extend.

**Hard safety rule:** a gas smell, or sparks/smoke/burning, skips lead collection. Give immediate safety instructions (leave the house, call 911 / PG&E at 1-800-743-5000) before anything else. Water near electrical: tell the user to stay out of the water and not touch outlets/panels.

## Tech stack
- Python 3.11+
- **LangGraph** for orchestration (a deliberate choice to showcase in the interview)
- **Anthropic** via `langchain-anthropic`. Model is configurable via env var; default to a Sonnet model for the agent, and a cheaper model (Haiku) is fine for eval simulation.
- **Pydantic** for state and structured extraction output
- LangGraph checkpointer (`MemorySaver` to start; `SqliteSaver` optional) so each conversation is a resumable thread
- **Google Places API (New) Text Search** for provider data, pulled once and cached to `providers/providers.json`. No live API calls during conversations or demos.
- Leads written to `leads/` as JSON (or SQLite)
- **Streamlit web chat page (`app.py`) is the main interface** (replaces the CLI). Sidebar: new conversation, sample openers, and a "Show agent brain" toggle for the live demo.
- LangGraph Studio and LangSmith tracing for demo/debugging (optional, nice-to-have)
- Secrets in `.env` (`ANTHROPIC_API_KEY`, `GOOGLE_PLACES_API_KEY`). Add `.env` to `.gitignore`; never commit keys.

## Graph design

```
user message
     │
 safety_check ──(gas / sparks / smoke)──► emergency_response ──► END
     │
  extract          (LLM → structured output, merges into state)
     │
  route            (plain Python, NOT an LLM call)
     ├─ category ambiguous ─► clarify ─────────┐
     ├─ required field missing ─► ask_next ────┤──► interrupt (wait for user) ──► back to safety_check
     └─ all required present ─► match_providers
                                     │
                                 confirm         (summary + providers + consent to share contact)
                                     │           ──► interrupt
                                 create_lead ──► END
```

Key principles:
- **The LLM extracts; code decides what's next.** `route` is a deterministic check of state (missing fields, category confidence). This keeps flow testable and explainable.
- `extract` **merges** into existing state. It never overwrites a known value with null.
- `ask_next` asks **one** question at a time, prioritized: clarify category → urgency (if unclear) → 1–2 category details → zip → name + contact. It should feel conversational, not like a form.
- `confirm` shows a short lead summary plus the matched providers, and asks for consent to share contact info. No lead without consent.
- Users can drop off at any point; state should record `status` so evals can measure it.

## State (roughly)
- `messages`
- All lead fields above, plus `category_details: dict`
- `facts` (append-only list of what the user said about the problem); `problem_description` is generated once from facts before confirm
- Routing metadata: `category_confidence`, `missing_fields`, `safety_flags`, `turn_count`, `out_of_scope_reason`, `status` (in_progress / emergency / awaiting_confirm / converted / abandoned / out_of_scope)
- `matched_providers`

## Provider data
- Region: Davis, Woodland, West Sacramento, Sacramento.
- For each category, run Places Text Search queries (e.g., "water damage restoration Davis CA"). Keep about 5–10 providers per category.
- Places field mask: `id`, `displayName`, `formattedAddress`, `businessStatus`, `location`, `primaryType`, `nationalPhoneNumber`, `rating`, `userRatingCount`, `websiteUri`. Billed at the Text Search Enterprise tier (phone/rating/website); the other fields are Pro or lower and add no cost.
- Store: name, phone, address, city, zip, lat/lng + `location_source` (exact / zip_centroid / city_centroid), business_status, primary_type, rating, review count, website, place_id, category, source query, and fetched_at timestamp.
- `providers_raw.json` keeps every fetched result; `providers.json` is the selected set (rebuild with `--reselect`, no API calls). Selection, in order: not `OPERATIONAL`; retail `primaryType`; `CATEGORY_TYPE_ALLOWLIST` (per-category allowed `primaryType`, generic types allowed explicitly where Places has no specific type); manual exclusions in `providers/overrides.json`; no phone; fewer than `MIN_REVIEW_COUNT` (10) reviews; no location; >15 mi from all four cities. Then dedupe by phone within a category (keep most reviews) and keep up to 3 per category per nearest city, never backfilling with providers that failed a filter. Every exclusion is logged with a reason; `--review` also prints a category × city coverage grid.
- Distance uses the exact Places location when present, falling back to the zip/city centroid.
- `python -m providers.fetch --review` prints the selected set for manual review (also works with `--reselect`).
- `match.py`: filter by category, then rank by proximity to the user's zip (simple zip→lat/lng lookup is fine), rating, and review count. Return the top 2–3.
- Stretch goal: manually verify a handful of providers against the CA CSLB license lookup and record `license_verified`.

## Project structure
```
home-services-agent/
├── agent/
│   ├── state.py          # Pydantic models: LeadState, ExtractedFields
│   ├── nodes.py          # one function per node
│   ├── graph.py          # wires nodes + edges + checkpointer
│   ├── prompts.py        # extraction + question-asking prompts
│   └── categories.py     # category table as data
├── providers/
│   ├── fetch.py          # one-time Google Places pull → providers_raw.json → providers.json
│   ├── overrides.json    # manual exclusions (place_id, categories, reason)
│   ├── providers_raw.json
│   ├── providers.json
│   ├── models.py, geo.py
│   └── match.py          # filter by category + zip, rank
├── evals/                # (unused: evals are the scripted suite in tests/test_live.py)
├── leads/                # output: {lead_id}.json
├── transcripts/          # every conversation: {thread_id}.json (messages + final state)
├── tests/
├── app.py                # Streamlit chat page (main UI)
├── .env.example
└── README.md
```

## Build phases (stop after each for review)
1. **Scaffold**: repo structure, deps (`requirements.txt` or `pyproject.toml`), `.env.example`, `state.py`, `categories.py`.
2. **Provider data**: `fetch.py`, a cached `providers.json`, and `match.py` with a small test.
3. **Happy path**: `prompts.py`; nodes `extract`, `route`, `ask_next`, `match_providers`, `summarize`, minimal `confirm`, `create_lead`; `graph.py` (one invocation per user turn, `MemorySaver` keyed by thread_id); Streamlit `app.py`. Run the basement example end to end.
4. **Robustness**: `safety_check` / `emergency_response`, `clarify` for ambiguous categories, and `confirm` with consent and interrupts.
5. **Evals (scripted scenario suite; no simulated LLM users, no judge)**: `tests/test_live.py` runs ~13 fixed scripts against the real API (basement flooding, single vs. ongoing sparks, gas, raccoon with/without damage, street power outage, vague ceiling stain, refuses phone, refuses all contact, changes problem, zip first, out-of-region zip). Each scenario has an opener, answers keyed by question type, and expected status/category/urgency. A run prints a summary table (expected vs. actual status, turns, category ✓, urgency ✓, faithfulness ✓, seconds) and writes it to `samples/scenario_results.md`.
6. **Polish for submission**:
   - README covering the lead definition, design decisions, the conversion-vs-quality tradeoff, eval results, known gaps, and next steps.
   - Saved sample transcripts and leads.

## Non-goals
Visual polish, booking/payments, actually sending leads to businesses, and provider availability/pricing.

## Decisions log
- **Service area**: zips around 95616 (Davis, Woodland, West Sacramento, Sacramento). Out-of-area handling is not a focus.
- **Scope**: only the 8 categories above for now. Out-of-scope redirects (utility outage, wildlife, auto) deferred.
- **Confirm edits**: if the user corrects something at confirm, route back to `extract` instead of creating the lead.
- **Provider data**: review fetched providers and keep the ones most likely to respond (operational, has phone, reasonable reviews).
- **Workflow**: Claude builds the code; Tanmmay reviews the workflow and suggests changes.
- Added `agent/config.py` (env settings) and `tests/` beyond the original structure.
- **Out-of-scope reason** is stored on `LeadState` (carried over by `merge_extracted`) so evals can break down out-of-scope endings.
- **Extraction failures** (Phase 3 `extract` node): retry once, passing the validation error back to the model. If it fails again, drop the invalid field(s) (e.g. treat category as unknown → route goes to clarify) and continue. An extraction error never ends the conversation. Log the failure so it shows up in eval runs. Document this in the node's docstring.
- **Facts, not a running summary**: the extractor returns `new_facts` each turn; `merge_extracted` appends (skipping exact duplicates). `problem_description` is generated once from facts by a `summarize` node right before confirm. `Lead` stores both.
- **Validators**: `Lead` requires a phone or email, consent, and ≥1 matched provider; `ExtractedFields` requires `category_confidence` whenever `category` is set; `LeadState.safety_flags` is typed `list[SafetyFlag]`.
- **Safety tiers**: only `gas_smell` is a hard stop (instructions, then END). `sparks_or_smoke` and `burning_smell` are urgent: safety instructions first, urgency forced to `emergency`, then lead collection continues (Phase 4).
- **Out of scope**: `out_of_scope` status + `out_of_scope_reason` (utility_outage / wildlife_removal / other) with redirect messages in `categories.OUT_OF_SCOPE`; no unverified phone numbers. These conversations are excluded from the conversion-rate denominator in evals.
- **Places field mask** adds `businessStatus`, `location`, `primaryType` (Pro tier; no price change since phone/rating/website already bill at Enterprise).
- **HVAC allowlist excludes `plumber`-typed listings** (mostly plain plumbers), accepting the loss of a few combo shops (Lee's, Ace) that remain under plumbing.
- **Overrides** so far: Mosquito & Vector Control District, Cardinal Professional Products, Lawn Master Landscaping, King KitchenAid (all); MAK Design + Build (roofing); Billy & Sons (water_damage); AMERICO Handyman (electrical, plumbing); Skippy Cleaners and Hawley Construction (water_damage, not restoration companies).
- **Thin coverage is OK**: some category × city cells have <2 providers, but the 25 mi match radius still returns 3 providers per category for Davis, Woodland, West Sacramento and Sacramento zips.
- **UI**: Streamlit chat page replaces the CLI; every conversation is saved to `transcripts/` whether or not it converts. No demo video; live walkthrough on the interview call.
- **Safety gate (Phase 4)**: `safety_check` runs on every user message before any LLM call, using keyword patterns in `agent/safety.py` (with simple negation handling), so a gas smell is handled even if the API is down. The extractor's `safety_flags` are a second net, checked in `route`. Gas -> `emergency_response` (instructions, status `emergency`, END). Sparks/burning -> notice prepended to the reply, urgency forced to `emergency`, collection continues. Water near electrical -> one-line tip. Each notice is shown once (`warned`).
- **Clarify**: asked at most once, offering the category and its `confused_with` partner using `confusion_note`. If still ambiguous after that, the extractor's best category is accepted (`category_locked`).
- **Out of scope**: route sends any `out_of_scope_reason` to a redirect message, status `out_of_scope`, END.
- **Contact fallbacks**: phone declined -> offer email once -> if both declined (or ignored), explain once -> end with status `declined`, still showing the matched pros' phone numbers. New status `declined` also covers "no" at consent.
- **Never repeat a detail question**: asked-but-unanswered details are recorded as "unknown".
- **Problem change / confirm edits**: extractor returns `retracted_facts` and `remove_fields`; a category change drops the old category's details. "Use my email instead" at confirm updates the field and re-shows the summary.
- **Urgency calibration**: emergency/within_48h only when stated or clearly implied by active damage or a safety risk; routine issues (running toilet) are within_week/flexible.
- **Samples**: `leads/` and `transcripts/` are gitignored; `samples/` holds curated runs with fake user data (`SAVE_SAMPLES=1` on the live tests).
- **Faithfulness**: details are recorded only when the user directly answers; the summarizer may not add scope claims or inferred negatives. `agent/faithfulness.py` is a deterministic tripwire (guard words like "only"/"limited to"/"single" absent from the source, or sentences mostly unsupported by facts + details); live tests assert it on every converted lead.
- **Urgency is mapped conservatively**: within_48h only for today/tomorrow/ASAP or clear urgency signals; "next few days"/"this week"/"soon" -> within_week. Never round up.
- **Wildlife pivot**: wildlife_removal gets the redirect plus one damage question (`pivot_offered`). If the user describes damage, the conversation continues as a normal roofing/electrical/handyman lead and a conversion counts as a normal conversion in evals; otherwise it ends `out_of_scope`. utility_outage/other still redirect and end.
- **Cause questions** (`DetailQuestion.about_cause`: water_damage.source, plumbing.fixture, roofing.cause) are skipped once the user says they don't know the cause (`cause_unknown`); scope/symptom questions are still asked.
- **Faithfulness check source** = facts + category_details keys and values. Ownership words (homeowner/owner/renter/tenant) are flagged unless `owner_or_renter` is set; summaries say "the customer". Summaries don't restate details the facts already cover, and "unknown" details aren't passed to the summarizer.
- **Detail questions**: the extractor sees `unanswered_detail_keys` every turn and fills any that earlier messages already answer, so they're never asked. After the first "not sure" on any detail, no more details for that lead. Max 2 asked.
- **Ranking guarantee**: the nearest provider that passed all filters is always in the top 3 (replaces #3 if the score left it out).
- **Electrical urgency**: sparks set a floor of within_48h; emergency only for ongoing sparking/smoke, heat or scorching (extractor's call) or a burning smell (always emergency). The safety notice is shown either way.
- **Phase 5 replaced**: scripted scenario suite instead of simulated users + LLM judge (deterministic, cheap, every failure is reproducible). The faithfulness tripwire stands in for the judge's faithfulness check.
- **Service area**: route checks the zip as soon as it's known. A real zip >25 mi from all four cities -> out-of-area message, status `out_of_scope`, reason `out_of_area` (set by code, not the extractor; excluded from conversion like other out-of-scope endings). A zip that doesn't exist is dropped and asked again. Previously an out-of-region user was asked for name + phone before being told we couldn't help.
- **Plumbing vs water damage** (prompt-level rule, single source `categories.PLUMBING_VS_WATER_DAMAGE`, used as both categories' confusion_note, in the extraction prompt's "Category rules" with a worked example, and in the clarify question): active leak or stain with no pooled/standing water and no soaked floors -> plumbing; pooled water, soaked flooring/walls, or flooding -> water_damage; both -> water_damage only for emergencies with standing water, otherwise plumbing. The extractor is told a clarify answer usually settles the category (confidence >= 0.7 when the rule decides). Found via vague_opener: the clarify answer "no water on the floor" was left at water_damage 0.5 and then locked in.
- **Deployment (Streamlit Community Cloud)**: settings come from `st.secrets` first, then env / `.env` (`config.get_secret`). `APP_PASSWORD` set -> password screen before anything renders (skipped when unset, e.g. locally). `DEPLOYED=true` -> "Past conversations" hidden and nothing written to disk (no transcripts; leads are validated and shown but not saved), since visitors' names and phones would otherwise land on a shared server. Runtime deps in `requirements.txt`; pytest in `requirements-dev.txt`.
