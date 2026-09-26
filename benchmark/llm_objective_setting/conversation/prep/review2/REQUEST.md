# Re-review: --cache-ram 0 added to the conversation benchmark's server command (Pixel Robot, research only)
ROLE: Reviewer. PONYTAIL: off. Independent correctness/safety review. Do NOT call any tool or run any command, do not
edit, create, stage, commit or push anything, do not run motors, and do not launch another reviewer. All material is
inlined below; answer from this text only.

## Context
The previous candidate (rubric fixes, blind sheet, conv_speed.py/run_conversation.sh speed script) was reviewed
APPROVE WITH NOTES. Afterwards the full quality run showed a problem: with server_manager.py's flags, Qwen3.5-4B's
llama-server died silently after ~25 turns (twice at the same point; the second time Android also killed Termux).
Per-turn measurement (mem_diag) showed the server's anonymous RSS growing ~104 MiB per request while its file-backed
weight pages were squeezed out; with --cache-ram 0 anonymous RSS stayed flat over 20 turns. The human chose: add
--cache-ram 0 to the benchmark's server flags only (quality rerun for Qwen3.5-4B + speed script), report it as a
deviation, re-review. server_manager.py itself was not changed.

## Changes since the reviewed version (reviewed files reconstructed; their SHA-256 match the earlier frozen hashes)
conv_speed.diff, run_conversation.diff below. conv_quality.py uses conv_speed.server_cmd, so the Qwen3.5-4B quality
rerun used --cache-ram 0; the other four models' quality rows come from the first run, which used server_manager's
flags without it (all 81 turns completed for each). Combination: rows of the first run minus qwen35_4b, plus the
qwen35_4b rerun (81 rows) -> conversation_quality_combined.json. The killed attempt's directory was kept, renamed.

## Please report
Correctness of the change; whether mixing the four first-run models (no --cache-ram 0) with the Qwen3.5-4B rerun
(with it) invalidates any quality comparison (prompts carry a random nonce and cache_prompt is the server default);
whether the speed comparison stays fair; any other issue. End with APPROVE / APPROVE WITH NOTES / CHANGES REQUESTED.
