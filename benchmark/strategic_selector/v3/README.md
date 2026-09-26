# Strategic-selector harness v3 (2026-09-24 to 2026-09-26)

Status: **partial** — development-set and finalist results are archived; the held-out split is
**sealed until final scoring** and neither it nor its generator is in Git. Results are recorded in
DECISIONS #112; the follow-on fine-tuning track is DECISIONS #113. Research only: offline synthetic
text cases, no motors, no robot integration.

## What v3 measures

66 development cases (11 v2 families × 6, seed 3101) with v2's labels unchanged and seeded numeric
noise, identical bytes for every candidate. Each candidate chooses among the offered options in normal
and reversed order; the harness records preferred/acceptable matches, order flips, load and per-call
timing, and external peak RSS. After DECISIONS #113 the selector is judged only on the four judgment
families (room_finished, hall_hint, repeat_search, heard_from_room: 24 dev cases); the rule families
move to Python.

## Results (provisional timings; agent resident)

- dev-full, 66 cases, `filtered_text` (`runs/dev-full.report.v2.md`): s1o P 46 / A 64 but warm median
  27.2 s; laya_en P 34 / A 54, 19/66 flips, 3.5 s; von11 P 33 / A 49, 55/66 flips, 2.8 s; laya_multi and
  laya_micro lower.
- Finalists on the 24 judgment cases (`runs/fin/table_*.md`): s1o 24/24 acceptable but best warm median
  ~24.9 s → out on speed; laya_en ONNX fp32 24/24 acceptable (canonical order), 7/24 reversed, ~2.1 s,
  1646 MiB, parity 48/48 vs torch; von11 24/24 acceptable in both orders, ~2.75 s; von11
  permutation-averaged (`runs/fin/von11_perm/`) and the Von NLI head (`runs/fin/von10_nli/`, ~8.8 s)
  were measured as order-invariance options.
- Measurement findings: Android can move Termux from cpuset 0–7 to `/foreground` 0–5, silently
  truncating `taskset` pins (`runs/cpuset_check2/`, `runs/fin/_failed_cpuset/`); `measure.py` now
  records allowed CPUs and refuses truncated pins. llama.cpp 1609's prompt cache changed s1o
  probabilities by up to 0.18 (`runs/fin/s1o_cache_diag/`, `runs/fin/s1o_cache_check.txt`).

## Sealed and withheld files (never commit)

| File | SHA-256 | Status |
|---|---|---|
| `cases.py` (case generator) | `892a946dab0ff0973224b1fceb01400661f7a0f40f22892ace306d2dd4678781` | withheld from Git; stays on the phone in this folder |
| `heldout.jsonl` | `96d41308eb0d2fa0c3bf30c1370eb93d1b4b7afbfd13cd433370cfd5cc3b2601` | **66 cases, seed 7919, sealed until final scoring**; never loaded, printed or scored (`runs/HELDOUT_SEAL_NOTE.md`) |
| `review_v3_request.txt` (first v3 review request, 44744 bytes) | `fb2fc92c523fcc599aa67b1e31e850ecdf5e0b2c897ef74423fececd385ab73b` | withheld from Git: it inlines the full `cases.py` source; kept on the phone only at `/termux-home/v3-runs/withheld/` |

The seal is procedural: anyone with `cases.py` and the seed can regenerate the split. Copies of
`cases.py` in `v3-runs/base_v3_pre_finalists/` and in review workspaces were excluded too.

**Generator imports.** `run.py`, `report.py` and `s1o_probs_evidence.py` import `load_v2` from
`cases.py`, and `test_v3.py` imports `encode`, `generate` and `load_v2`. `adapters.py`, `measure.py`,
the jevlike tool (`../manual/jevlike/`) and the ladder runner do **not** import it. Without `cases.py`,
`run.py`, `report.py`, `s1o_probs_evidence.py` and `test_v3.py` do not run from the repository alone.
Proposed split (not done here, because executed sources are archived byte-identical and a change needs
its own review): move `load_v2()` — the hash-checked loader of `../robot_selector_benchmark.py`, which
holds no case logic — into a new committed `v2_loader.py`, and change the three runtime imports to
`from v2_loader import load_v2`. `cases.py` would import it from there and stay uncommitted.
`test_v3.py`'s regeneration checks need the generator and would stay phone-only.

## Contents

| Path | What |
|---|---|
| top level: `adapters.py`, `measure.py`, `report.py`, `run.py`, `s1o_probs_evidence.py`, `test_v3.py`, `CASES.json`, `dev.jsonl` | the current harness (the version jevlike and the ladder use); `dev.jsonl` is the unsealed development split |
| `sources/pre_finalists/` | harness as executed for dev-full (adapters `ae7dd0a0…`, measure `e3fcbe3f…`, run `caeb2a3c…`, report `f10d94b1…` for report v2) |
| `sources/dev_full_report.py` | report.py `1fc7ff87…`, which produced `runs/dev-full.report.md` |
| `sources/finalists_final/` | harness at the end of the finalists (adapters `23e90a3d…`, measure `7040be22…` with the cpuset guard, run `a491c364…`), its diff, tables and diagnostics as given to the finalist Reviewer |
| `runs/` | every v3 run folder from `/termux-home/v3-runs/` (adapter-check, mock-check, dev-full, evidence, fin/…), the partial dev run of 2026-09-24, the cpuset check, the laya_en ONNX export logs, harness hash files, the seal note |
| `review/` | Reviewer outputs and texts for the three reviews, the v3b and finalist requests, the v3b review workspace |
| `reports/` | the Coder's reports verbatim and the report parts sent at the time |

Top-level `adapters.py` (`b452ab09…`) adds the speed-variant environment overrides used by the
ladder s1o speed run (`../ladder/`). [RUN_INDEX.md](RUN_INDEX.md) lists runs; [ARTIFACTS.md](ARTIFACTS.md)
hashes every archived file.

## Review status

AGY `gemini-3.1-pro-high`, `--sandbox`. Review v3 (dev-full): **FAIL / INCOMPLETE** — its central
"inverted flips" finding was contradicted by the result files and it could not find the JSONs; treated
as incomplete. Re-review v3b: **PASS**, confirming that finding was wrong. Finalists review: **PASS
WITH CAVEATS**.

## Evidence gaps

- The finalists' pre-guard `measure.py` (`2c578660…`, in `runs/harness_sha256_finalists_prerun.txt`)
  has no surviving copy; runs before the guard have uncertain core allocation.
- All timings are provisional (agent resident, cpuset changes, thermal state varies).
- The held-out split is unscored; generator-derived data does not decide the winner (DECISIONS #113).
- The ladder cases in `../ladder/` are a development bench, not the final exam.
