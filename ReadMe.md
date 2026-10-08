# Home Services Lead Agent

A conversational agent that turns "something's wrong with my house" into a **dispatchable lead** for a **real, local service provider** in the Davis–Sacramento area.

The user describes their problem in plain language. The agent asks only the questions a provider actually needs, matches the user with real nearby businesses, and, with the user's consent, produces a structured lead a contractor could act on without reading the chat.

> **Live demo:** TODO_LINK (password: shared separately)
> Built with LangGraph + Claude (Anthropic API) + Streamlit. Provider data comes from Google Places.

---

## Example

```
User:   Water started coming into my basement last night after the storm.
Agent:  I'm sorry, that's stressful. How soon do you need someone?
User:   As soon as possible, it's still coming in.
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

That's 5 turns and about 20 seconds, with only 3 questions asked. Category, urgency and source were inferred from the user's own words.

---

## What a "lead" is

A lead is a **structured handoff record** that a provider can act on cold, without reading the conversation. A lead is **dispatchable** only when all of these hold:

| Field | Required | How it's gathered |
|---|---|---|
| category | ✅ | Inferred (8 categories: water damage, plumbing, electrical, HVAC, roofing, pest control, appliance repair, handyman) |
| problem description | ✅ | Generated once, at confirmation, from a list of facts the user stated |
| urgency | ✅ | Inferred conservatively; asked only if unclear |
| zip | ✅ | Asked and validated |
| name + phone **or** email | ✅ | Asked **last** and validated |
| consent to share | ✅ | Explicit yes at confirmation |
| ≥1 matched real provider | ✅ | From the provider dataset |
| address, owner/renter, availability, category details | Optional | Inferred or asked only when useful |

The schema is enforced in code (`agent/state.py`). A `Lead` object cannot exist without its required fields, a contact method, and `consent_to_share = True`.

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
     ├─ out of scope ─► redirect (wildlife: offer help with any damage)
     ├─ category ambiguous ─► clarify (asked at most once)
     ├─ field missing ─► ask_next (one question per turn) ──► wait for user
     └─ complete ─► match_providers ─► summarize ─► confirm (consent) ─► create_lead
```

**Key principles**

- **The LLM extracts; code decides what happens next.** Routing is a deterministic check of the state, so the flow is predictable and unit-testable. The LLM is used only where language understanding is needed.
- **Safety runs before the LLM.** Keyword detection catches a gas smell on every message, so a gas leak is handled correctly even if the API is slow or down. Sparks or a burning smell trigger safety instructions first, then lead collection continues, because that user needs an electrician.
- **Facts, not an overwritable summary.** Each turn appends the facts the user stated. The problem description is written once, at the end, from those facts. That keeps it from degrading over the conversation and keeps it faithful to what was actually said.
- **Fail soft.** LLM calls have timeouts and retries. If extraction fails validation, the agent retries once, then drops the bad field and re-asks. An API error never crashes or hangs a conversation.

---

## Provider data

All providers are **real businesses** pulled from the **Google Places API (New)**, then filtered:

1. **Pull:** 10 search terms × 4 cities (Davis, Woodland, West Sacramento, Sacramento) = 40 requests, giving 352 raw results.
2. **Filter**, with every exclusion logged with a reason:
   - business must be `OPERATIONAL`, with a phone number, within the service area
   - **category–type allowlist** (e.g. a "plumber" listing must actually be a plumber, not a handyman or a hardware store)
   - **10+ reviews**, to avoid thin or lead-gen listings
   - **dedupe by phone number**
   - **manual overrides**, for cases the rules can't catch (a government agency, a supply distributor, a landscaper, a plumber listed under water damage), each with a written reason
3. **Result:** 77 providers. Every city still gets 3 matches in every category, because matching draws from neighboring cities within 25 miles.

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
| Raccoon → redirect, then offer help with damage | ✅ recovers otherwise-lost conversations | |
| Out-of-scope and gas leaks excluded from the conversion denominator | | ✅ honest metric |
| Strict provider filters | | ✅ survives a spot-check |

Consent is the one place I deliberately gave up a bit of conversion for quality.

---

## Testing

**Scripted scenario suite** (`tests/test_live.py`): 13 end-to-end conversations against the real agent and model, each checking status, category, urgency, and whether the lead description stays faithful to what the user said.

| Scenario | Expected | Result | Turns | Seconds |
|---|---|---|---|---|
| basement_flooding | converted (emergency) | ✓ | 5 | 19 |
| sparking_outlet_single | converted (within 48h) | ✓ | 6 | 26 |
| sparking_ongoing | converted (emergency) | ✓ | 6 | 25 |
| gas_leak | emergency, no lead | ✓ | 1 | 0.0 |
| raccoon_roof_damage | converted as roofing | ✓ | 7 | 25 |
| raccoon_no_damage | out of scope | ✓ | 2 | 6 |
| power_outage_street | out of scope (PG&E/SMUD) | ✓ | 1 | 3 |
| vague_opener | converted, plumbing | ✓ | 7 | 23 |
| refuses_phone | converted via email | ✓ | 4 | 15 |
| refuses_all_contact | declined | ✓ | 8 | 36 |
| changes_problem | converted, recategorized | ✓ | 6 | 27 |
| zip_first | converted, zip not re-asked | ✓ | 3 | 9 |
| out_of_region (90012) | politely declined | ✓ | 1 | 2 |

TODO: replace with the final full run (`samples/scenario_results.md`).

One example of iterating on these results: `vague_opener` (a ceiling stain under a bathroom) flipped between *plumbing* and *water damage* across runs. I made the rule explicit (active leak with no standing water → plumbing, so the source gets fixed first), and it then passed 3 of 3 runs, also getting faster (32s → 23s).

There are also **80+ offline unit tests** (no API calls) covering routing, validation, merging, safety detection, matching and provider selection.

### What these tests can and can't tell you

Scripted scenarios show that **specific behaviors work** and that changes don't break them. They **don't measure real conversion or lead quality**. Real users are messier than scripts, and only real providers can say whether they'd accept a lead. See *What's next* for how I'd measure both in production.

---

## Running it

Requires Python 3.11 and an **Anthropic API key**. No Google key is needed; provider data is cached.

```bash
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY
streamlit run app.py            # opens http://localhost:8501
```

Tests:

```bash
pytest                                     # offline unit tests (no API calls)
RUN_LIVE=1 pytest tests/test_live.py -s    # scripted scenarios against the real model
```

Refreshing provider data (needs `GOOGLE_PLACES_API_KEY`):

```bash
python -m providers.fetch --dry-run     # show queries + request count, no calls
python -m providers.fetch               # pull from Google Places
python -m providers.fetch --reselect    # re-apply filters to the raw data, no API calls
python -m providers.fetch --review      # print the selected providers for manual review
```

In the app, **Show agent brain** in the sidebar displays the agent's live state each turn: extracted fields, facts, what's still missing, the routing decision, and matched providers.

---

## Project structure

```
agent/        state.py (schemas, merge, validation) · categories.py (category data)
              safety.py · prompts.py · nodes.py · graph.py · config.py
providers/    fetch.py · match.py · geo.py · models.py
              providers.json · overrides.json
tests/        offline unit tests · test_live.py (scenario suite)
samples/      example transcripts and leads (fake contact details only)
app.py        Streamlit chat UI
```

---

## Known limitations

- **Not measured on real users.** The scenario suite shows behaviors work, not real conversion rates.
- **Leads aren't actually sent.** Dispatch (SMS, email, or a provider API) is out of scope; the lead is written as JSON.
- **Provider availability and pricing are unknown.** We match on category, location and reputation only.
- **Licensing isn't verified.** Providers pass type and review filters, but CSLB license status isn't checked.
- **In-memory conversation state.** Conversations don't survive an app restart; history on the deployed demo is disabled for privacy.
- **English-first.** Other languages aren't explicitly handled.
- **Latency** is about 3–8 seconds per turn (mitigated by streaming replies).

---

## What's next

1. **Measure for real.** Track conversion per conversation, and get provider feedback on each lead (accepted, called back, booked). A/B test agent changes against those numbers.
2. **Simulated-user evals before shipping changes.** Run 20+ LLM-played personas (impatient, vague, renter, refuses phone…) with a "contractor judge" scoring each lead. This is a cheap regression signal between versions, though not a substitute for real metrics.
3. **Production architecture.** Put the agent behind an API service (e.g. FastAPI) with a persistent checkpointer (Postgres) and a separate frontend, plus logging for the metrics above.
4. **Provider quality.** Verify CSLB licenses, refresh data periodically, and use provider response rates in ranking.
5. **Lead dispatch.** Send to providers by SMS or email, with a provider-facing view of the lead.

---

*Built with AI tools, as the assignment allowed: Claude for planning and design review, and Claude Code for implementation. Design decisions, trade-offs and testing were mine.* TODO: edit this line to describe your process accurately.