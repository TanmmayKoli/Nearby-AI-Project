# Home Services Lead Agent

[![tests](https://github.com/TanmmayKoli/Nearby-AI-Project/actions/workflows/tests.yml/badge.svg)](https://github.com/TanmmayKoli/Nearby-AI-Project/actions/workflows/tests.yml)

A conversational agent that turns "something's wrong with my house" into a **dispatchable lead** for a **real, local service provider** in the Davis–Sacramento area.

The user describes their problem in plain language. The agent asks only the questions a provider actually needs, matches the user with real nearby businesses, and, with the user's consent, produces a structured lead a contractor could act on without reading the chat.

> **Live demo:** https://nearby-ai-project-b5pv5yq5is5zt24hdmech4.streamlit.app/ (password shared separately)
> Built with LangGraph + Claude (Anthropic API) + Streamlit. Provider data comes from Google Places.

---

## Example

```
User:   Water started coming into my basement last night after the storm.
Agent:  I'm sorry you're dealing with that. Is water still actively coming in right now?
User:   Yes, it's still seeping in slowly.
Agent:  Understood. What's your zip code so I can find pros near you?
User:   95616
Agent:  What's your name and the best number for a pro to reach you?
        (Address is optional, but it helps the crew head straight there.)
User:   Jane Doe, 530-555-0123
Agent:  Here's the request I'll send: Water damage restoration · Emergency · 95616
        ... 3 nearby restoration companies ...
        Is it OK to share your contact info with these pros?
User:   Yes
Agent:  You're all set. If you'd rather not wait, you can call any of them now: ...
```

That's 5 turns and about 17 seconds, with only 3 questions asked. Category, urgency and source were inferred from the user's own words.

---

## What a "lead" is

A lead is a **structured handoff record** that a provider can act on cold, without reading the conversation. A lead is **dispatchable** only when all of these hold:

| Field | Required | How it's gathered |
|---|---|---|
| category | ✅ | Inferred (9 categories: water damage, plumbing, electrical, HVAC, roofing, pest control, appliance repair, handyman, wildlife removal) |
| problem description | ✅ | Generated once, at confirmation, from a list of facts the user stated |
| urgency | ✅ | Inferred conservatively; asked only if unclear |
| zip | ✅ | Asked and validated |
| name + phone **or** email | ✅ | Asked **last** and validated |
| consent to share | ✅ | Explicit yes at confirmation |
| ≥1 matched real provider | ✅ | From the provider dataset |
| address, owner/renter, availability, category details | Optional | Inferred or asked only when useful |

The schema is enforced in code (`agent/state.py`). A `Lead` object cannot exist without its required fields, a contact method, and `consent_to_share = True`.

**Provider's view.** After consent, the app shows *What the provider receives*: the lead as a pro would see it (urgency · category · zip, the description, key details, the customer's first name and last initial, the contact method, and "shared with N local pros · customer consented"), with placeholder Accept / Decline buttons. It's built from the `Lead` object only, never internal state, by `lead_to_provider_message()` in `agent/dispatch.py`.

---

## Architecture

```
user message
     │
 safety_check ──(gas smell)──► emergency_response ──► END        (keyword check, no LLM)
     │
  extract          LLM → structured output; appends new facts to state
     │
  route            plain Python: what's missing, what's next
     ├─ out of scope ─► redirect (utility outage, outside the service area)
     ├─ category ambiguous ─► clarify (asked at most once)
     ├─ field missing ─► ask_next (one question per turn) ──► wait for user
     └─ complete ─► match_providers ─► summarize ─► confirm (consent) ─► create_lead

conversation already ended ─► after_end: a 1–2 sentence reply (the lead is never changed)
```

**Key principles**

- **The LLM extracts; code decides what happens next.** Routing is a deterministic check of the state, so the flow is predictable and unit-testable. The LLM is used only where language understanding is needed.
- **Safety runs before the LLM.** Keyword detection catches a gas smell on every message, so a gas leak is handled correctly even if the API is slow or down. Sparks or a burning smell trigger safety instructions first, then lead collection continues, because that user needs an electrician.
- **Facts, not an overwritable summary.** Each turn appends the facts the user stated. The problem description is written once, at the end, from those facts. That keeps it from degrading over the conversation and keeps it faithful to what was actually said.
- **Rules that must not flake live in code.** Two behaviors moved out of the LLM after a flaky scenario run: ongoing electrical hazards ("keeps sparking", smoke, a hot outlet, scorch marks) force urgency to emergency via keyword rules in `agent/safety.py`, and the repair offer after a wildlife lead maps damage words to a trade (vent/roof → roofer, wiring → electrician, insulation/drywall → handyman) in `agent/nodes.py`.
- **Fail soft.** LLM calls have timeouts and retries. If extraction fails validation, the agent retries once, then drops the bad field and re-asks. An API error never crashes or hangs a conversation.

---

## In the app

- **Quick replies:** tappable answers for urgency, consent, and "Not sure" on detail questions. Typing still works, and the buttons disappear once used.
- **Provider cards** with Call, Website and Google Maps links, plus **See more pros near you** (extra matches the user can contact directly; they don't receive the request).
- **Provider's view** of the lead after consent (see above).
- **Repair offer after wildlife leads:** if the user mentions damage (a torn vent, chewed wiring), the lead keeps it as a fact, and afterwards a button starts a separate request for the right trade.
- **No-match fallback:** if no provider serves that category near the zip, the agent says so and names the area we cover, instead of creating an undispatchable lead.
- **After the conversation ends,** a new message gets a short reply to what was actually asked (gas endings repeat the safety instruction first), plus a **Start a new request** button that reuses zip and contact only if the user ticks the box.
- **Show agent brain** in the sidebar displays the agent's live state each turn.

---

## Provider data

All providers are **real businesses** pulled from the **Google Places API (New)**, then filtered:

1. **Pull:** 12 search terms × 4 cities (Davis, Woodland, West Sacramento, Sacramento) = 48 requests (40 in the main pull, plus 8 for wildlife removal), giving 383 raw results.
2. **Filter**, with every exclusion logged with a reason:
   - business must be `OPERATIONAL`, with a phone number, within the service area
   - **category–type allowlist** (e.g. a "plumber" listing must actually be a plumber, not a handyman or a hardware store)
   - **10+ reviews**, to avoid thin or lead-gen listings
   - **dedupe by phone number**
   - **manual overrides**, for cases the rules can't catch (a government agency, a supply distributor, a landscaper, a plumber listed under water damage), each with a written reason
   - **wildlife removal uses a manually reviewed include-list.** Google types wildlife companies the same as insect and rodent pest control, so the type allowlist can't tell them apart. Shelters, government offices and general pest control are excluded; only a company whose own website lists raccoon/possum/skunk/bat removal is included.
3. **Result:** 76 providers. Every city gets 3 matches in every category except wildlife removal, because matching draws from neighboring cities within 25 miles. Wildlife has one vetted provider and matches up to 40 miles, so every city in the service area reaches it.

`providers_raw.json` isn't in the repo; `--reselect` requires a fresh fetch with your own Google key.

**Matching** ranks by rating (Bayesian-adjusted, so a 5.0 with 2 reviews doesn't beat a 4.8 with 300) minus a distance penalty that's **stronger for urgent jobs**. The nearest qualifying provider is always included in the top 3. In the app, every provider card links to its Google Maps listing, so it's easy to verify.

Data was pulled on 2026-10-08 (`fetched_at` is recorded per provider). Only the cached `providers.json` is used at runtime, with no live Google calls.

---

## Design decisions and trade-offs

The prompt names two outcomes: **conversion** (does the conversation produce a dispatchable lead?) and **lead quality** (would a provider act on it?). Most decisions trade one against the other.

| Decision | Conversion | Lead quality |
|---|---|---|
| Infer before asking; one question per turn | ✅ fewer turns | |
| Contact info asked last | ✅ the user is invested by then | |
| Skip detail questions the user already answered or can't know | ✅ | |
| Facts list → summary written once | | ✅ no drift, no invented details |
| Conservative urgency ("next few days" → within a week) | | ✅ providers aren't misled |
| Validation drops bad zip/phone and re-asks | | ✅ no unreachable leads |
| **Explicit consent before dispatch** | ⚠️ costs one turn | ✅ opted-in users respond |
| Sparks → emergency lead, not a hard stop | ✅ keeps the most valuable leads | ✅ correct urgency |
| Raccoon → wildlife lead, then offer a separate repair request for any damage | ✅ one conversation can produce two leads | |
| Out-of-scope and gas leaks excluded from the conversion denominator | | ✅ honest metric |
| Strict provider filters | | ✅ survives a spot-check |

Consent is the one place I deliberately gave up a bit of conversion for quality.

---

## Testing

**Scripted scenario suite** (`tests/test_live.py`): 13 end-to-end conversations against the real agent and model, each checking status, category, urgency, and whether the lead description stays faithful to what the user said.

| Scenario | Expected | Result | Turns | Seconds |
|---|---|---|---|---|
| basement_flooding | converted (emergency) | ✓ | 5 | 16.8 |
| sparking_outlet_single | converted (within 48h) | ✓ | 4 | 13.8 |
| sparking_ongoing | converted (emergency) | ✓ | 4 | 12.4 |
| gas_leak | emergency, no lead | ✓ | 1 | 0.0 |
| raccoon_roof_damage | converted as wildlife removal, repair offered | ✓ | 6 | 20.6 |
| raccoon_no_damage | converted as wildlife removal, no repair offer | ✓ | 6 | 20.5 |
| power_outage_street | out of scope (PG&E/SMUD) | ✓ | 1 | 2.5 |
| vague_opener | converted, plumbing | ✓ | 7 | 25.7 |
| refuses_phone | converted via email | ✓ | 5 | 17.3 |
| refuses_all_contact | declined | ✓ | 8 | 32.8 |
| changes_problem | converted, recategorized | ✓ | 8 | 28.3 |
| zip_first | converted, zip not re-asked | ✓ | 3 | 8.2 |
| out_of_region (90012) | politely declined | ✓ | 1 | 2.7 |

**13/13 passed, 202s total** (replies on `claude-sonnet-5-5`, extraction on `claude-haiku-4-5-20251001`). Full table with per-node latency: `samples/scenario_results.md`.

Moving extraction to Haiku cut end-to-end time ~8% (216s → 198s on the same scenarios) with no accuracy loss; per-node profiling shows extraction is still ~58% of latency, so prompt caching is the next step.

One example of iterating on these results: `vague_opener` (a ceiling stain under a bathroom) flipped between *plumbing* and *water damage* across runs. I made the rule explicit (active leak with no standing water → plumbing, so the source gets fixed first), and it then passed 3 of 3 runs, also getting faster (32s → 23s).

There are also **246 offline tests** (no API calls) covering routing, validation, merging, safety detection, matching, provider selection, the provider message, analytics (payload allow-list, failure isolation, stats math), and the Streamlit UI. They include an **app startup smoke test** that loads `app.py` with no API key and the network blocked (normal and deployed-with-password), and imports every module. **GitHub Actions** runs the offline suite on every push and pull request (`.github/workflows/tests.yml`).

### What these tests can and can't tell you

Scripted scenarios show that **specific behaviors work** and that changes don't break them. They **don't measure real conversion or lead quality**. Real users are messier than scripts, and only real providers can say whether they'd accept a lead. See *What's next* for how I'd measure both in production.

---

## Analytics

The scenario suite can't measure real conversion, so the deployed app records **anonymous outcome data** in Supabase: one row per conversation, updated after every turn (`agent/analytics.py`).

- **Logged:** a random id made just for analytics (not the chat's id), start and last-update time, source (`deployed` / `local`), number of turns, outcome (converted, declined, out of scope, no match, emergency, in progress), category, urgency, number of providers matched, and average turn time.
- **Never logged:** message text, names, phone numbers, emails, zips, addresses, the facts the user described, IPs or any user identifier. A unit test checks the payload against this allow-list.
- **Can't break the chat:** writes run in a background thread with a 2-second timeout; any error is logged as a warning and the conversation carries on. It's off when Supabase isn't configured, and always off in tests and CI.

**Analytics is optional and off by default.** The app works the same without it. To enable it locally:

1. Create your own Supabase project.
2. Run `supabase/schema.sql` in its SQL editor (creates the `conversations` table with row-level security on).
3. Set `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` and `ADMIN_PASSWORD` in `.env` (see `.env.example`), then run the app and open `/stats`.

An admin-only **stats page** (`/stats`, behind `ADMIN_PASSWORD`, not linked from the app) shows total conversations, conversion rate (converted ÷ eligible, where eligible leaves out out-of-scope, no-match and gas-emergency conversations, plus ones still in progress), raw conversion rate, outcomes, conversations by category, average turns to conversion, average and p90 turn latency, and conversations per day, filterable by source. Conversations with no update for 30+ minutes count as abandoned.

---

## Running it

Requires Python 3.11 and an **Anthropic API key**. No Google key is needed; provider data is cached.

```bash
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY
streamlit run app.py            # opens http://localhost:8501
```

Settings (`.env` locally, or Streamlit secrets when deployed):

| Variable | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `AGENT_MODEL` | `claude-sonnet-5-5` | every reply the user reads, the summary and the consent check |
| `EXTRACT_MODEL` | `claude-haiku-4-5-20251001` | the extraction step only (faster) |
| `APP_PASSWORD` | unset | if set, a password screen comes first |
| `DEPLOYED` | unset | `true` = public demo: no past conversations, nothing written to disk, nothing sent |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | off | **local only**: traces each turn to LangSmith, tagged with its thread id. Never enabled when `DEPLOYED=true`, and off for the offline tests |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | unset | anonymous analytics (see above); off when unset. The service key is only used server-side |
| `ADMIN_PASSWORD` | unset | password for the `/stats` page (separate from `APP_PASSWORD`); the page is off when unset |

Tests:

```bash
pip install -r requirements-dev.txt        # adds pytest
pytest                                     # offline tests (no API calls)
RUN_LIVE=1 pytest tests/test_live.py -s    # scripted scenarios against the real model
```

Refreshing provider data (needs `GOOGLE_PLACES_API_KEY`):

```bash
python -m providers.fetch --dry-run     # show queries + request count, no calls
python -m providers.fetch               # pull from Google Places (writes providers_raw.json)
python -m providers.fetch --categories wildlife_removal   # pull only some categories and merge them in
python -m providers.fetch --reselect    # re-apply filters to providers_raw.json, no API calls
python -m providers.fetch --review      # print the selected providers for manual review
```

---

## Project structure

```
agent/        state.py (schemas, merge, validation) · categories.py (category data)
              safety.py (keyword safety rules) · prompts.py · nodes.py · graph.py · config.py
              dispatch.py (what a provider receives) · history.py (past conversations)
              timing.py (per-node latency) · faithfulness.py (description check)
              analytics.py (anonymous usage stats + stats calculations)
providers/    fetch.py · match.py · geo.py · models.py
              providers.json · overrides.json (exclusions + wildlife include-list)
tests/        offline tests · test_app_smoke.py (startup) · test_live.py (scenario suite)
samples/      example transcripts and leads (fake contact details only), scenario_results.md
app.py        Streamlit chat UI
pages/        stats.py (admin stats page, password-gated)
.github/workflows/tests.yml   CI: offline tests on every push and PR
```

---

## Known limitations

- **Not measured on real users.** The scenario suite shows behaviors work, not real conversion rates.
- **Leads aren't actually sent.** Dispatch (SMS, email, or a provider API) is out of scope; locally the lead is written as JSON.
- **The deployed demo stores no personal data and sends no leads.** Conversations live only in the browser session; only anonymous outcome stats are recorded (see *Analytics*).
- **Wildlife removal has thin coverage** (one vetted provider, in Roseville; its Yolo County coverage is unconfirmed).
- **Provider availability and pricing are unknown.** We match on category, location and reputation only.
- **Licensing isn't verified.** Providers pass type and review filters, but CSLB license status isn't checked.
- **In-memory conversation state.** Conversations don't survive an app restart; history on the deployed demo is disabled for privacy.
- **English-first.** Other languages aren't explicitly handled.
- **Latency** is about 3.4 seconds per turn on average in the final run, occasionally up to ~9 seconds (mitigated by streaming replies).

---

## What's next

1. **Measure for real.** Track conversion per conversation, and get provider feedback on each lead (accepted, called back, booked). A/B test agent changes against those numbers.
2. **Simulated-user evals before shipping changes.** Run 20+ LLM-played personas (impatient, vague, renter, refuses phone…) with a "contractor judge" scoring each lead. This is a cheap regression signal between versions, though not a substitute for real metrics.
3. **Production architecture.** Put the agent behind an API service (e.g. FastAPI) with a persistent checkpointer (Postgres) and a separate frontend, plus logging for the metrics above.
4. **Provider quality.** Verify CSLB licenses, refresh data periodically, and use provider response rates in ranking.
5. **Lead dispatch.** Send to providers by SMS or email. `lead_to_provider_message()` in `agent/dispatch.py` is the hook a real dispatch step would use.
6. **Prompt caching for extraction.** Extraction is ~58% of latency, and most of its prompt (instructions, category list) is identical every turn.

---

*Built with AI tools, as the assignment allowed: Claude for planning and design review, and Claude Code for implementation. Design decisions, trade-offs and testing were mine.* TODO: edit this line to describe your process accurately.