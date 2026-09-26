# Candidates phase 0 — run index

| Run | When (UTC) | Output | Status |
|---|---|---|---|
| Laya 0.3.20 download + smoke | 2026-09-24 ~21:00 | `raw/laya0320/` | complete, loads and decides |
| laya-micro build (seed-top default) + smoke | 2026-09-24 21:16–21:21 | `raw/laya-micro/build{1,2,3}_*`, `smoke.*` | complete |
| laya-micro 8192-vocab rebuild + shape timing | 2026-09-24 22:33–22:36 | `raw/laya-micro/*_8192*`, `shape_timing.*` | complete (used later by v3) |
| system-one-open zero-shot smoke | 2026-09-24 ~21:20 | `raw/s1o/` | complete |
| Decision-1.0 (5 models) | – | `THIRD_PARTY_SHA256.txt` (code only) | not run: no CPU path |
