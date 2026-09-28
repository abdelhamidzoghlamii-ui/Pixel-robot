# Ladder — run index

Cases `ladder_cases_v1.jsonl` `41eeafd2…` (development bench, not the final exam) unless noted.

| Run | When (UTC) | Output | Status | Result |
|---|---|---|---|---|
| toy runs (3) | 2026-09-25 20:33–20:42 | `runs/old_runs/` | superseded | runner build checks, 3 toy cases, laya_en + von11 |
| toy run (post-review) | 2026-09-25 21:09 | `runs/toy_run/` | complete | runner after round-2 fixes |
| smoke run | 2026-09-25 | `runs/smoke_run/` | complete | the other five models load and decide |
| real run, 7 models | 2026-09-25 ~22:35 – 09-26 00:08 | `runs/real_run/` | complete (timings disturbed) | see README |
| s1o speed, first start | 2026-09-26T04:44Z | `runs/s1o_speed_20260926T044457Z/` | **aborted** | cpuset guard: only cores 0–5 allowed |
| s1o speed, 5 variants | 2026-09-26 04:56–08:39Z | `runs/s1o_speed_20260926T045648Z/` | complete (**resumed** at 06:56Z) | b2351 + Q4_0 fastest Gemma, 1547 ms |
| s1o speed prep quick runs | 2026-09-26 02:50–03:36Z | `s1o_speed_prep/quick_run*/` | superseded | Coder dry runs |
| s1o dotprod b1609 + Gemma Q4_0, main block (oneshot 22:37:53Z) | 2026-09-27 22:43–22:46Z | `runs/s1o_dp_q40_20260927T224301Z/`, `logs/s1o_dp_q40_*`, `logs/oneshot_console_20260927T223753Z.log` | complete | 93/120, 1408 ms median |
