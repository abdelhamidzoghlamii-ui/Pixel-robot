
## My notes on the review (checked read-only, no code changed)

- **Finding 1 is contradicted by the data.** Choices are stored as meaning-level keys, not letters. For example, `s1o.json` row 0: canonical offered `['search_here', …, 'travel:toilet']` and chose `search_here`; reversed offered `['travel:toilet', …, 'search_here']` and also chose `search_here`. So a flip means the model chose a different action, which is what a flip should mean. s1o's letters are mapped back to option keys by position before scoring. The mock adapter scores the key names, so its choice doesn't depend on option order, and 0 flips is correct for it. The chosen positions are also spread out: s1o's canonical picks fall at positions 0/1/2/3/4 = 20/10/32/3/1, not "always A".
- **Finding 3:** the parity failure is real, and it isn't caused by my check. States are plain strings, not `json.dumps` output, and both tokenizers are the same library. The first difference is `▁Mission` → `▁Mis` + `sion`.
- **Finding 5 is wrong:** `dev-full/*.json` were in the workspace (about 470 KB each). The reviewer didn't read them, so it never recomputed the reversed-order numbers, flips or warm medians.
- **Findings 2 and 4 are fair.** The held-out set is sealed procedurally, not cryptographically: anyone with `cases.py` can regenerate it. The per-family cells are "preferred/acceptable", which reads ambiguously.

I've made no changes, and any change would need a new review. Nothing is committed or pushed, and `docs/` is untouched.

**Decision pending:** re-run the reviewer with the per-candidate summaries and a sample of rows included in the prompt, so it can actually recompute? And fix the report layout (finding 4), which would then need that new review anyway?
