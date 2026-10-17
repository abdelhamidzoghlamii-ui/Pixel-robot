# ARCHIVE-2 run index — archive2

Fixed labels (related complete session where this is a report/partial-attempt folder):

E4B Q4_0 OFF vs ON (MTP drafter), runner 8a84040, chat settings (ctx 4096, temperature 0.7, n_predict 512, cores 4–7), owner-run native Termux, 13/13 prompts per config, one block each. OFF: gen 7.31 tok/s, load 9.7 s, prompt 0.81 s — VALID single run. ON: gen 10.57 tok/s (+45 %), draft acceptance 43.0 %, load 10.2 s, prompt 0.77 s — INDICATIVE: WARM START (skin 27.42 °C vs 25.9 °C gate after 486 s). Memory (max PSS) 3,898 vs 4,948 MiB, min MemAvailable 1,923 / 1,920 MiB; J/token 1.29 -> 0.85 INDICATIVE ONLY. Heat: the OFF block raised skin 24.43 -> 36.58 °C in 140 s; Android status 0. Reply quality NOT EVALUATED. thermal_20261017.log z9/z10/z11 are CPU-core readings, NOT TRUSTED as a heat state.

E4B-UP download and smoke evidence (commit 8a84040). Downloads pinned to unsloth revision bfc15c38…; Q4_0 4,836,002,944 B SHA-256 4a403d2e…; drafter 98,653,248 B SHA-256 b6a72311…; local E2B files unchanged before/after. Smokes untimed, agent resident: NOT measurements.

| Run/file | UTC start / provenance | Label |
|---|---|---|
| `../chat_mtp_e4b_20261017T115533Z/` | 2026-10-17T11:55:40Z (run.json started) | E4B Q4_0 OFF vs ON (MTP drafter), runner 8a84040, chat settings (ctx 4096, temperature 0.7, n_predict 512, cores 4–7), owner-run native Termux, 13/13 prompts per config, one block each. OFF: gen 7.31 tok/s, load 9.7 s, prompt 0.81 s — VALID single run. ON: gen 10.57 tok/s (+45 %), draft acceptance 43.0 %, load 10.2 s, prompt 0.77 s — INDICATIVE: WARM START (skin 27.42 °C vs 25.9 °C gate after 486 s). Memory (max PSS) 3,898 vs 4,948 MiB, min MemAvailable 1,923 / 1,920 MiB; J/token 1.29 -> 0.85 INDICATIVE ONLY. Heat: the OFF block raised skin 24.43 -> 36.58 °C in 140 s; Android status 0. Reply quality NOT EVALUATED. thermal_20261017.log z9/z10/z11 are CPU-core readings, NOT TRUSTED as a heat state. |
| `../e4bup_evidence/` | UNKNOWN (evidence has no canonical UTC start) | E4B-UP download and smoke evidence (commit 8a84040). Downloads pinned to unsloth revision bfc15c38…; Q4_0 4,836,002,944 B SHA-256 4a403d2e…; drafter 98,653,248 B SHA-256 b6a72311…; local E2B files unchanged before/after. Smokes untimed, agent resident: NOT measurements. |
| `../oneshot_console_20261017T115025Z.log` | 2026-10-17T11:50:25Z (filename) | E4B native launcher console; not a separate result. |
| `../thermal_20261017.log` | 2026-10-14T22:06:56Z (first retained line) | Cumulative CPU-zone log; NOT TRUSTED as heat state; old 220 lines are exact prefix of 424 lines. |
