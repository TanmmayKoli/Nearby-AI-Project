# Samples

Curated example conversations and the leads they produced, for reviewers.

- **Fake user data only.** Names are made up and phone numbers use the reserved
  555-01XX range. Provider names, phones and ratings are real (from Google Places).
- `transcripts/` and `leads/` at the project root are gitignored because real test
  runs can contain real contact info. Copy a run here only after checking it.

Each sample is `<scenario>.transcript.json` (+ `<scenario>.lead.json` when the
conversation converted). Transcripts include the final agent state, so you can
see facts, route decisions, safety flags and matched providers.
