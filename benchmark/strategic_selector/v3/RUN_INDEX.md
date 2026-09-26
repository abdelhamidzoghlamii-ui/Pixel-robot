# Harness v3 — run index

All runs: rooted Pixel 7, Debian proot, dev split `a4e0f0bf…` (never the held-out). Timings provisional.

| Run | When (UTC) | Output | Status | Result |
|---|---|---|---|---|
| mock-check, adapter-check | 2026-09-24 | `runs/mock-check/`, `runs/adapter-check/` | complete | harness end-to-end without and with each model |
| partial dev run | 2026-09-24T23:10 | `runs/partial_dev_20260924/` | superseded by dev-full | laya_en, laya_micro, laya_multi, von11 JSONs |
| dev-full, 5 candidates × 66 cases × 2 orders | 2026-09-24/25 | `runs/dev-full/`, `runs/dev-full.report*.md` | complete | see README; s1o best accuracy, too slow |
| s1o probability evidence | 2026-09-25 | `runs/evidence/` | complete | letter probabilities pre-sampling |
| finalists: laya_en ONNX sweeps (cores 4–7 / 6–7 × 2–4 threads) | 2026-09-25 | `runs/fin/onnx_*`, `table_laya_en_onnx.md` | complete (one failed attempt kept: missing mask id) | ~2.1 s at 4–7 × 4 |
| finalists: von11 sweeps | 2026-09-25 | `runs/fin/von11_*`, `table_von.md` | partial: 6–7 × 3/4 failed on the cpuset (`_failed_cpuset/`) | ~2.75 s |
| finalists: von11_perm, von10_nli | 2026-09-25 | `runs/fin/von11_perm/`, `runs/fin/von10_nli/`, smokes | complete | order-invariant options; NLI ~8.8 s |
| finalists: s1o_if (instruction first, prefix cache on/off), cache diagnostics, llama-bench | 2026-09-25 | `runs/fin/s1o_*`, `table_s1o.md` | complete | ~24.9 s best warm median; cache changes probabilities |
| cpuset check | 2026-09-25T06:12 | `runs/cpuset_check2/` | complete | guard refuses truncated pins |
| laya_en ONNX export | 2026-09-25T03:33 | `runs/laya_en_onnx_export/` | complete | export logs, meta, sha256 (graph not archived) |
| held-out | – | – | **sealed** | 66 cases, never loaded |
