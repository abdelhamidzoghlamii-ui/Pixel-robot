# Selector candidates, phase 0: availability and smoke test (2026-09-24)

Status: **complete** (a smoke test, not a benchmark). Relates to DECISIONS #109 and #112.

Which new high-level selector candidates exist, install and make one decision offline on the Pixel 7?
Each ran the same toy state (`raw/toy.json`: 4 options) on cores 4–7, 4 threads, with peak RSS from
outside the process (`wait4`).

## Result

| Candidate | Revision | Loads | Load ms | First call ms | Peak RSS |
|---|---|---|---:|---:|---:|
| Laya 0.3.20 (later `laya_en`) | `convaiinnovations/laya@55cf4c4e` | yes | 14451 | 1364 | 2810 MiB |
| laya-micro (int8 ONNX, built on the phone; later `laya_micro`) | code `686fc466`, weights `55cf4c4e` | yes | see report | see report | see report |
| system-one-open zero-shot on the installed Gemma 4 E2B GGUF (later `s1o`) | code `77f1f7cc`, llama.cpp 1609 | yes | 4798 | 8268 | 3506 MiB |
| Decision-1.0 Kai/Lex/Eos/Sol/Nox | `llm-semantic-router` | **not tested**: every runtime requires a ROCm GPU, no CPU path | – | – | – |

The Decision-1.0 weights were never downloaded. Full table, per-candidate choices and the verbatim
warnings are in `reports/CODER_REPORT_candidates_phase0.md`. Later outcomes: laya_micro and
laya_multi were dropped, laya_en and von11 moved to fine-tuning (DECISIONS #112, #113).

## Contents

| Path | What |
|---|---|
| `raw/run_smoke.py`, `raw/toy.json`, `raw/base_pins.txt` | the executed smoke driver, the toy state, base package pins |
| `raw/laya0320/` | Laya 0.3.20: pins, pip/download logs, `sha256.txt` of the downloaded files, `smoke_laya.py` and its outputs |
| `raw/laya-micro/` | laya-micro: build logs (prune, export, quantize; first and 8192-vocab builds), `build*_sha256.txt`, meta files, `smoke_micro.py`, `shape_timing.py` and outputs |
| `raw/s1o/` | system-one-open zero-shot: `smoke_s1o.py`, outputs, llama-server log |
| `THIRD_PARTY_SHA256.txt` | hashes of third-party code that was inspected but not archived (Decision-1.0 repos, system-one-open; laya-micro and s1o clone HEADs) |
| `reports/CODER_REPORT_candidates_phase0.md` | the Coder's reports, verbatim |

No weights, ONNX graphs, venvs or HF caches are archived. [ARTIFACTS.md](ARTIFACTS.md) hashes every file.

## Review status

**Not reviewed** — by instruction, no Reviewer was run for this step (stated in the Coder report).

## Evidence gaps

- A single toy decision per candidate: no accuracy or speed conclusion follows from it.
- laya-micro's pruned tokenizer diverged from stock on the toy text; its int8 graph was slower than fp32 on this phone.
- The laya-micro ONNX builds and pruned checkpoints were deleted from the phone on 2026-09-26
  (human decision), after `raw/` was archived; their hashes stay in `raw/laya-micro/build*_sha256.txt`.
- The ladder cases in `../ladder/` are a development bench, not the final exam.
