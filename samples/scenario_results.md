# Scenario results

Scripted runs of `tests/test_live.py`.

| scenario | expected status | actual status | turns | category | urgency | faithful | seconds | extract s | notes |
|---|---|---|---|---|---|---|---|---|---|
| basement_flooding | converted | converted ✓ | 5 | ✓ | ✓ | ✓ | 16.8 | 10.0 |  |
| sparking_outlet_single | converted | converted ✓ | 4 | ✓ | ✓ | ✓ | 13.8 | 7.6 |  |
| sparking_ongoing | converted | converted ✓ | 4 | ✓ | ✓ | ✓ | 12.4 | 6.1 |  |
| gas_leak | emergency | emergency ✓ | 1 | — | — | — | 0.0 | 0.0 | last_route=emergency_response |
| raccoon_roof_damage | converted | converted ✓ | 6 | ✓ | — | ✓ | 20.6 | 11.4 |  |
| raccoon_no_damage | converted | converted ✓ | 6 | ✓ | — | ✓ | 20.5 | 12.4 |  |
| power_outage_street | out_of_scope | out_of_scope ✓ | 1 | — | — | — | 2.5 | 2.5 | last_route=out_of_scope; reason=utility_outage |
| vague_opener | converted | converted ✓ | 7 | ✓ | ✓ | ✓ | 25.7 | 13.3 |  |
| refuses_phone | converted | converted ✓ | 5 | ✓ | — | ✓ | 17.3 | 11.1 |  |
| refuses_all_contact | declined | declined ✓ | 8 | — | — | — | 32.8 | 19.4 | last_route=no_contact |
| changes_problem | converted | converted ✓ | 8 | ✓ | — | ✓ | 28.3 | 15.5 |  |
| zip_first | converted | converted ✓ | 3 | ✓ | ✓ | ✓ | 8.2 | 4.6 |  |
| out_of_region | out_of_scope | out_of_scope ✓ | 1 | — | — | — | 2.7 | 2.6 | last_route=out_of_scope; reason=out_of_area |

**13/13 scenarios passed** · replies `claude-sonnet-5-5` · extract `claude-haiku-4-5-20251001` · total 202s · 2026-10-08 14:33

## Latency per node

| node | model | calls | avg s | max s | total s |
|---|---|---|---|---|---|
| extract | claude-haiku-4-5-20251001 | 49 | 2.38 | 3.20 | 116.6 |
| ask_next | claude-sonnet-5-5 | 37 | 1.61 | 3.06 | 59.7 |
| summarize | claude-sonnet-5-5 | 9 | 1.41 | 3.24 | 12.7 |
| consent_reply | claude-sonnet-5-5 | 9 | 1.23 | 1.42 | 11.1 |
| confirm | (code) | 9 | 0.00 | 0.00 | 0.0 |
| create_lead | (code) | 9 | 0.00 | 0.00 | 0.0 |
| emergency_response | (code) | 1 | 0.00 | 0.00 | 0.0 |
| match_providers | (code) | 9 | 0.00 | 0.00 | 0.0 |
| no_contact | (code) | 1 | 0.00 | 0.00 | 0.0 |
| out_of_scope | (code) | 2 | 0.00 | 0.00 | 0.0 |
| route | (code) | 49 | 0.01 | 0.30 | 0.4 |
| safety_check | (code) | 59 | 0.00 | 0.00 | 0.1 |
