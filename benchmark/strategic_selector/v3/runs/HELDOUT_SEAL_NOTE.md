# Held-out seal decision (v3)

2026-09-25, human decision relayed to the Coder: the v3 held-out split stays **procedurally sealed**, with no code change.

- File: `benchmark/strategic_selector/v3/heldout.jsonl`, 66 cases, seed 7919, SHA-256 `96d41308eb0d2fa0c3bf30c1370eb93d1b4b7afbfd13cd433370cfd5cc3b2601` (recorded in `v3/CASES.json`).
- Threat model: the risk is the team tuning on the held-out cases, not an outsider regenerating them. Anyone can regenerate it from `cases.py` and the seed; that is accepted.
- Control: `run.py` refuses `--split heldout` unless `--unseal-heldout` is passed, and `test_v3.py` checks that. Nobody prints, inspects or scores it until an explicit unseal decision.
- Status at this date: never loaded, printed or scored. Every dev-full run JSON records `cases_sha256` = the dev hash `a4e0f0bf…`.
