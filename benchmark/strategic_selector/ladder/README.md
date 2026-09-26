# Selector ladder benchmark (2026-09-25 to 2026-09-26)

Status: **complete** as a development bench. Research only: offline text cases, no motors.
Relates to DECISIONS #109, #112, #113.

**The ladder cases are a development bench, not the final exam.** `cases/ladder_cases_v1.jsonl`
(SHA-256 `41eeafd2…ebb473`) holds 60 hand-written cases in 5 difficulty levels × 3 phrasings
(direct, indirect, narrative). The model that goes on the robot is decided on the independent test
set kept outside this repository (DECISIONS #113), not on these cases.

## Runner

`ladder.py` (runner, SHA-256 `c1a4f65d…` at archive time) starts each model as `ladder_worker.py`
in its own venv, reusing the v3 adapters (`../v3/adapters.py`) through the jevlike model table
(`../manual/jevlike/jevlike.py`). Per model it runs both option orders at 4 threads plus 2/3/4-thread
blocks, records decision time split into tokenise/forward/post, cold vs cached load, peak RSS, order
flips and thermal readings, and prints hints. `measure.py` here is a byte-identical copy of
`../v3/measure.py` (`7040be22…`): it records the allowed CPUs and refuses a truncated `taskset` pin
(the cpuset guard; see `../v3/README.md`). `test_ladder.py` is the runner's model-free test.

The phone runs were started from native Termux by `oneshot.sh` (thermal logger, screen kept on, 5 min
idle, root cache drop on request) calling `run_s1o_speed.sh` or `run_conversation.sh`. Versions:

| File | SHA-256 | Executed by |
|---|---|---|
| `oneshot_executed_s1o_resume.sh` | `0edcae55…` | the resumed s1o speed run (copy saved at 06:51Z, the resume started 06:51Z) |
| `oneshot_executed_conversation_run.sh` | `db1f3156…` | the conversation speed run (`../../llm_objective_setting/conversation/`) |
| `oneshot.sh` | `f3d531fc…` | **final version** (from `/sdcard/Download/`, 20:11Z): screen-timeout handling hardened; never executed |

## Results (provisional timings)

Real run, 7 models, 60 cases × 2 orders, cores 4–7 (`runs/real_run/report.txt`):

| Model | Correct (both orders) | Acceptable | Order flips | Decision median / P95 ms |
|---|---:|---:|---:|---|
| laya_en (ONNX) | 48/120 | 52/120 | 17/60 | 763 / 2228 |
| von11 | 66/120 | 72/120 | 33/60 | 846 / 1690 |
| s1o | 92/120 | 97/120 | – | 7605 / 14655 |
| laya_multi (dropped) | 57/120 | 63/120 | – | 383 / 638 |
| laya_micro (dropped) | 56/120 | 62/120 | – | 261 / 476 |
| von12 (dropped) | 66/120 | 72/120 | – | 756 / 1595 |
| von10 NLI (dropped) | 78/120 | 86/120 | – | 1602 / 6194 |

s1o letter-scoring speed run, resumed full run (`runs/s1o_speed_20260926T045648Z/report.txt`):
b1609 92/120, 6650 ms median; b2351 93/120, 2052 ms; b2351 + Gemma Q4_0 93/120, 1547 ms; b2351 +
Qwen3.5-2B 81/120, 1268 ms; b2351 + Qwen3.5-0.8B 53/120, 505 ms. The dotprod rebuild of b1609 is in
`../../llama_dotprod_rebuild/`.

## Contents

| Path | What |
|---|---|
| `ladder.py`, `ladder_worker.py`, `test_ladder.py`, `measure.py`, `run_s1o_speed.sh`, `run_conversation.sh`, `oneshot*.sh` | runner, worker, test, cpuset-guarded measure, run scripts |
| `cases/`, `toy_cases.jsonl` | the 60 development cases; 3 toy cases for runner tests |
| `runs/old_runs/`, `runs/toy_run/`, `runs/smoke_run/` | toy and smoke runs during the build (2026-09-25) |
| `runs/real_run/` | the 7-model run |
| `runs/s1o_speed_20260926T044457Z/` | **aborted partial** s1o speed run: stopped by the cpuset guard (cores 0–5 only), empty decisions |
| `runs/s1o_speed_20260926T045648Z/` | the s1o speed run, interrupted and **resumed** (`decisions.jsonl.before_resume_*` keeps the pre-resume rows) |
| `logs/` | stdout/stderr of each phone run, `oneshot` consoles, `thermal.log` (snapshot taken 2026-09-26T22:07:53Z; the logger was still appending) |
| `s1o_speed_prep/` | the Coder's preparation of the speed run: quick runs, diffs, review requests/outputs and workspaces |
| `reviews/ladder-review-*` | the two ladder-build reviews (request, frozen candidate, output) |
| `reports/` | the Coder's reports verbatim and `s1o_speed_prep_report.md` from `/sdcard/Download/` |

[RUN_INDEX.md](RUN_INDEX.md) lists runs; [ARTIFACTS.md](ARTIFACTS.md) hashes every file.

## Review status

Ladder build: round 1 (AGY `gemini-3.1-pro-high`) raised findings including a MAJOR thread-hint
defect; round 2 **PASS**. s1o speed preparation: first request returned an empty response
(incomplete), then **APPROVE WITH NOTES**; a later request failed with a network ERROR (incomplete)
and its re-run was **APPROVE**; final round **APPROVE**. The run results themselves were not reviewed.

## Evidence gaps

- Run folders record the cases hash, not the runner hash. `real_run` most likely used `ladder.py`
  `00fbe2fa…` (the round-2 reviewed candidate, also `s1o_speed_prep/orig/`); this is inferred.
- The `oneshot.sh` used for the first s1o speed start (04:51Z) was edited in place and has no copy.
- `real_run` timings are marked DISTURBED in its own report (repeat checks up to 4.1×); all timings
  are provisional (agent resident, cpuset and thermal state vary).
- s1o speed results before and after the resume come from two sessions.
- The models' weights and venvs are not in Git; von12's venv and weights and laya_micro's graphs were
  deleted from the phone on 2026-09-26, and their jevlike menu entries removed, so those two rows cannot
  be rerun as-is.
