# ARCHIVE-2 — archive2

E4B Q4_0 OFF vs ON (MTP drafter), runner 8a84040, chat settings (ctx 4096, temperature 0.7, n_predict 512, cores 4–7), owner-run native Termux, 13/13 prompts per config, one block each. OFF: gen 7.31 tok/s, load 9.7 s, prompt 0.81 s — VALID single run. ON: gen 10.57 tok/s (+45 %), draft acceptance 43.0 %, load 10.2 s, prompt 0.77 s — INDICATIVE: WARM START (skin 27.42 °C vs 25.9 °C gate after 486 s). Memory (max PSS) 3,898 vs 4,948 MiB, min MemAvailable 1,923 / 1,920 MiB; J/token 1.29 -> 0.85 INDICATIVE ONLY. Heat: the OFF block raised skin 24.43 -> 36.58 °C in 140 s; Android status 0. Reply quality NOT EVALUATED. thermal_20261017.log z9/z10/z11 are CPU-core readings, NOT TRUSTED as a heat state.

E4B-UP download and smoke evidence (commit 8a84040). Downloads pinned to unsloth revision bfc15c38…; Q4_0 4,836,002,944 B SHA-256 4a403d2e…; drafter 98,653,248 B SHA-256 b6a72311…; local E2B files unchanged before/after. Smokes untimed, agent resident: NOT measurements.

Fixed labels: Local AI 2026-10-17; L2 probability phrase corrected by Local AI, with owner authorization, on 2026-10-17. Source files are unchanged byte copies. Script files here are historical evidence, not entry points.

[Run index](RUN_INDEX.md) · [Artifacts](ARTIFACTS.md).

This metadata covers the new loose console and thermal log beside it and the E4B run/evidence folders. Existing runs/RUN_INDEX.md, runs/ARTIFACTS.md and runs/thermal.log are untouched. Old thermal.log is an exact byte and line prefix: 220 of 424 lines.
