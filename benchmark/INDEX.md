# Benchmark index

One line per archive folder. Each folder has a README (what, results, review status, evidence gaps),
a RUN_INDEX and an ARTIFACTS file with the SHA-256 of every archived file. No weights or venvs are in
Git. Case sets generated from the v2/v3 families and the ladder cases are **development benches, not
the final exam**: the selector is decided on an independent test set kept outside this repository
(DECISIONS #113).

| Folder | What | Date | Key result | Status |
|---|---|---|---|---|
| [strategic_selector/](strategic_selector/README.md) | Laya/Von high-level selector harness v1/v2: offline synthetic cases, four archived runs | 2026-09-23 | Laya `filtered_text` 9/11 acceptable held-out, 4/11 order flips; Von `two_stage_text` 8/11, 9/11 flips | superseded by v3 |
| [strategic_selector/von12_rerun/](strategic_selector/von12_rerun/README.md) | v2 procedure rerun with von-sdk 1.2.0 | 2026-09-24 | order flips 0/11, but held-out acceptable fell (8/11 → 2/11 and 5/11); Von 1.2 dropped | complete |
| [strategic_selector/candidates_phase0/](strategic_selector/candidates_phase0/README.md) | availability and smoke test of new selector candidates | 2026-09-24 | Laya 0.3.20, laya-micro and s1o load and decide offline; Decision-1.0 has no CPU path | complete |
| [strategic_selector/v3/](strategic_selector/v3/README.md) | harness v3: 66 shared dev cases, finalists on 24 judgment cases | 2026-09-24/26 | laya_en ONNX 24/24 acceptable ~2.1 s, von11 24/24 ~2.75 s, s1o out on speed (~24.9 s); cpuset-truncation and prompt-cache findings | partial (held-out sealed, unscored) |
| [strategic_selector/manual/jevlike/](strategic_selector/manual/jevlike/README.md) | `robot-jevlike`, interactive manual selector playground; its 2026-09-25 session logs are in [logs/](strategic_selector/manual/jevlike/logs/README.md) | 2026-09-25 | – (manual runs, no score) | tool; logs complete |
| [strategic_selector/ladder/](strategic_selector/ladder/README.md) | 60-case, 5-level ladder runner; 7-model run; s1o letter-scoring speed across builds and models | 2026-09-25/26 | real run: von11 66/120, laya_en 48/120, s1o 92/120 but 7.6 s; s1o on b2351 + Gemma Q4_0 1.5 s | complete (development bench) |
| [llm_objective_setting/](llm_objective_setting/README.md) | on-device LLM conversation/objective-setting benchmark (#110), plus the 2026-09-26 rubric-fixed rerun, blind conversation grades, conversation speed, server memory and voice-timeout checks | 2026-09-18/26 | E2B ≈ E4B on objective setting at half the cost; E2B Q4_0 blind-graded 41/70 at ~12 gen tok/s; robot-shaped prompts showed no server memory growth with or without `--cache-ram 0`; one unique-question conversation run grew ~1.9 MiB/turn without it (then dropped 82 MiB, cause unrecorded) vs ~0.1 MiB/turn with it, at no measurable cost | complete (the #110 part: see its README for invalid/unrun configs) |
| [llama_dotprod_rebuild/](llama_dotprod_rebuild/README.md) | llama.cpp b1609 rebuilt with `armv8.2-a+dotprod+fp16` for the robot | 2026-09-26 | `sdot` 0 → 1045; s1o 6650 → 2478 ms median; voice parses identical; gen +32% | complete |
| [qwen_vision/](qwen_vision/README.md) | #105 Qwen3.5 VL benchmark harness, its offline test, projector sourcing report, AGY review and the earlier `qwen35_vision_probe.py` | 2026-09-14/16 | – (no result files archived; their location is unknown); harness edited after review, not re-reviewed | archived sources only |
| [failclosed_108/](failclosed_108/README.md) | live verification output cited by #108, from local candidate `e5b4394`; the verified code was removed by #111 | 2026-09-14 | forced exception and `ERROR` response → STOP; KeyboardInterrupt propagated | historical evidence only |

Loose files: `run_benchmark.py` and `live_capture.jpg` predate these archives (April 2026) and are not
part of any folder above.
