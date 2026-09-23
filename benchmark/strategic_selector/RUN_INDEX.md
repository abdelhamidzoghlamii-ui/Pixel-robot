# Run and phone evidence index

## Archived selector runs

Run IDs use the phone result file's UTC modification time as a label, **not** a measured invocation start time. `manifest.json` in each directory holds the original phone path and SHA-256, archived SHA-256 and size for every copied raw file, plus source hash, runtime, framing and fixture split. The originals remain in `/termux-home/laya-test/`. The archived files are historical evidence: do not edit a run in place; create a new run ID.

| Run directory under `results/` | Version / status | Raw evidence | Key result or gap |
|---|---|---|---|
| [2026-09-23T145245Z-laya-v1](results/2026-09-23T145245Z-laya-v1/) | v1 original | `results.json`, `stdout.txt`, manifest | Selected filtered_text, held-out 7/11 acceptable under **defective v1 rubric**; historical only. Exact v1 source is [retained](robot_selector_benchmark_v1.py). |
| [2026-09-23T150926Z-von-v2](results/2026-09-23T150926Z-von-v2/) | v2 original | `results.json`, `stdout.txt`, manifest | Development-selected two_stage_text; held-out 8/11 acceptable, 9/11 order flips. |
| [2026-09-23T154330Z-laya-v2](results/2026-09-23T154330Z-laya-v2/) | v2 original | `results.json`, `stdout.txt`, manifest | Development-selected filtered_text; held-out 9/11 acceptable, 4/11 order flips. |
| [2026-09-23T155449Z-von-filtered-followup](results/2026-09-23T155449Z-von-filtered-followup/) | v2 **post hoc** | `stdout.txt`, manifest | Filtered_text after observing held-out result: 8/11 acceptable, 9/11 flips. **No structured JSON or exact ad hoc wrapper source was saved.** |

The v2 original JSON files each have 132 rows: 22 development cases × five framings, 11 selected held-out cases and 11 order-reversed cases. The v1 JSON likewise has 132 rows. The follow-up stdout reports normal and reversed summaries and choices, but lacks per-call probabilities and structured states. It is not an untouched evaluation. There is no saved Von v1 selector run. No missing result was recreated or rerun for this archive.

## Earlier speed-test evidence retained on phone

[PHONE_HISTORY.tsv](PHONE_HISTORY.tsv) records 49 precise phone paths, sizes and SHA-256 values for the small source/log/package files below. It is an **inventory**, not a copy. These experiments ask different latency or runtime questions and use low-level example options, not the strategic fixture set. The downloaded research README contained some pasted summaries; the corresponding raw phone files were inspected where available. No weights, cache, virtualenv, photos, private scans or third-party browser repository were copied.

| Phone directory | Available raw/code | Missing or limited evidence |
|---|---|---|
| `/termux-home/laya-test/` | `laya_smoke.py`; smoke, warm, thread, affinity, input-length and pinned-core stdout logs. | Several ad hoc timing harnesses are not saved as exact standalone files. These logs are raw phone evidence, but do not fully reproduce every timing command. |
| `/termux-home/laya-onnx-test/` | `onnx-fp32-output.txt`, `onnx-sustained-output.txt`, package manifests. Sustained log reports about 929 ms median and 940 ms P95 over 30 calls. | Exact separate ONNX benchmark source is not present in this directory; model cache is excluded. |
| `/termux-home/laya-q8-test/` | Native and Node-WASM `.mjs`/`.cjs` scripts, thread sweeps, raw output logs, package manifests. Native 30-call median about 1112 ms; buffered single-thread Node-WASM median about 5618 ms. | Probability delta 0.009707 is one reference-state parity check, not broad calibration. Failed threaded Node/proot setup says nothing about browser threading. Weights excluded. |
| `/termux-home/laya-web-browser/` | Separate browser repo has `app/src/benchmark.ts` and `app/src/parity.ts`; only their paths/hashes are indexed. | Pixel browser 4-thread median about 2465 ms and P95 about 2581 ms exist as user-pasted summary; **no saved raw browser output was found in this directory**. Do not treat the pasted figures as raw-verified or copy the repository. |
| `/termux-home/von-test/` | `von-first-load-output.txt`, `von-warm-four-thread-output.txt`, `von-safety-matrix-output.txt`; warm log reports about 1082 ms median and 1099 ms P95 for a short example. | Exact standalone speed-test script is not saved here. The strategic Von v2 raw JSON/log are archived separately above. |

A previous Laya FP32 8-thread median about 1965 ms and 4-core median about 1271 ms were in user-pasted research summaries; phone logs cover related runs but the exact command-to-summary mapping is not complete. Do not silently relabel a different log as that run. Browser numbers remain summary-only. All earlier speed figures use different input lengths and runtime setups; none is a fair comparison with v2 strategic decision latency.

## Provenance checks and gaps

- `python archive_run.py verify` compares archived hashes in every manifest. It does not certify model correctness or infer the original command from a log.
- The downloaded `DOC_DIFF_STRATEGIC_SELECTOR.md` and `README.md` matched the supplied hashes, but are proposals, not executed evidence. The ambiguous Downloads `robot_selector_benchmark.py` is v1; `robot_selector_benchmark-1.py` is v2 with two extra comments. The **executed** v2 source here has SHA-256 `3aa9d399d6978710c143a6f1dfe69541b288cea329f27077b2500e322e8c14e2`.
- The phone raw selector JSON/logs contain generated room labels, a first name used in synthetic examples, model choices and local cache paths. Inspection found no photos, scans, credentials, audio or live apartment map. If later runs include sensitive real inputs, keep the immutable original outside Git and commit a redacted derivative plus original hash, size, protected location and redaction method in its manifest.
- The older `benchmark/live_capture.jpg` and `benchmark/run_benchmark.py` are already tracked and were left untouched. `.gitignore` still excludes other new files directly under `benchmark/`; the selector directory is explicitly visible to Git. Check `git status --short --untracked-files=all` before any future commit.
