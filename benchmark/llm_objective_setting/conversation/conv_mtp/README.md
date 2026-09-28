# Conversation speed with MTP speculative decoding (2026-09-28)

Status: **complete** (one unattended timed run). Research only: offline, no motors, no `main.py`.

## What

`../conv_speed_mtp.py` (SHA-256 `e8fbbdb6…`, run by `../run_conv_mtp.sh`, `709a132d…`) reuses
`../conv_speed.py` (`9b721d86…`), `../../bench.py` (`0792beee…`) and the ladder's thermal/cpuset/cold-load
helpers (`../../../strategic_selector/ladder/ladder.py`, `c7e6d3c9…`). It runs five configs, one cold
block each, over 13 of the DECISIONS #110 prompts (A5 A6 B1 B2 B6 B7 C1 C6 C7 C8 C9 C10 C11) at
temperature 0 on the robot's dotprod llama.cpp b1609 build (`libggml-cpu.so` `a4ef0d97…`) with
`server_manager.py`'s flags plus `--cache-ram 0`, cores 4–7, 4 threads:

| Config | Model | Speculative decoding |
|---|---|---|
| gemma_e2b_q40 | gemma-4-E2B-it-Q4_0 | off |
| gemma_e2b_q40_mtp | same | `--model-draft mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3` |
| qwen35_4b_q4km | Qwen3.5-4B-Q4_K_M | off |
| qwen35_4b_q40mtp | Qwen3.5-4B-Q4_0-MTP | off |
| qwen35_4b_q40mtp_on | same | `--spec-type draft-mtp --spec-draft-n-max 3` (built-in MTP head) |

Each prompt's bytes are reused across the MTP off/on pair, so `same` counts byte-identical replies.
Exact server commands are in `conv_mtp_20260928T081906Z/run_20260928T081907Z.json`; model hashes in
`ARTIFACTS.md`. Started from native Termux by `oneshot.sh` (the live copy is byte-identical to
`../../../strategic_selector/ladder/oneshot_executed_conversation_run.sh`, `db1f3156…`).

## Results

`conv_mtp_20260928T081906Z/report.txt` (medians; gen tok/s token-weighted):

| Config | load s | TTFT s | prompt tok/s | gen tok/s | peak MiB | accept | same as MTP off |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma_e2b_q40 | 5.8 | 3.08 | 73.0 | 14.03 | 3975 | – | – |
| gemma_e2b_q40_mtp | 8.0 | 3.11 | 73.5 | **20.54** | 4584 | 44.5% | 13/13 |
| qwen35_4b_q4km | 6.8 | 6.68 | 31.7 | 7.18 | 5006 | – | – |
| qwen35_4b_q40mtp | 6.0 | 5.77 | 36.9 | 7.72 | 4965 | – | – |
| qwen35_4b_q40mtp_on | 14.2 | 5.99 | 35.1 | **8.60** | 4953 | 58.1% | 13/13 |

MTP raised Gemma E2B generation by +46% (+609 MiB peak, +2.2 s load) and Qwen3.5-4B Q4_0 by +11%
(load 6.0 → 14.2 s), with every reply byte-identical to its MTP-off twin. TTFT and prompt speed are
unchanged. `c_scores.txt` (`../score_c.py`, report only, 7 C prompts): both Gemma configs 6/7 exact
(C7 answered room `"schlafzimmer"`), all Qwen configs 7/7; no errors, no truncation.

## Contents

| Path | What |
|---|---|
| `conv_mtp_20260928T081906Z/` | `turns.jsonl`, `blocks.jsonl`, `report.txt`, `c_results.json`, `c_scores.txt`, run manifest, server logs |
| `conv_mtp_20260928T081906Z.{stdout,stderr}.txt`, `oneshot_console_20260928T081359Z.log` | run output and `oneshot` console |
| `reviews/review1…5/` | per round: request, frozen candidate diff (and delta), `frozen.sha`, checks, final verdict, stdout, stderr |
| `../conv_speed_mtp.py`, `../run_conv_mtp.sh` | the executed scripts (kept beside `conv_speed.py`; `run_conv_mtp.sh` hardcodes that directory) |

[RUN_INDEX.md](RUN_INDEX.md) lists the run; [ARTIFACTS.md](ARTIFACTS.md) hashes every file.

## Review status

Codex CLI `gpt-6-sol`, read-only sandbox, medium effort, fresh session per round. Rounds 1–4
**REQUEST CHANGES** (prompt pairing, resume after interruption, grader exceptions); round 5 **APPROVE**
(`reviews/review5/final.md`). `reviews/review5/frozen.sha` equals the executed scripts' hashes. The run
results themselves were not reviewed.

## Evidence gaps

- One run, 13 prompts; `same 13/13` shows identical output only on these prompts at temperature 0.
- Preparation smoke logs and toy runs are not archived (see RUN_INDEX).
- Model files are not in Git; their hashes are in `ARTIFACTS.md`.
