# Qwen3.5 VL on-device vision benchmark (#105) — archived sources (2026-09-14, archived 2026-09-27)

Status: **archived, no results**. These files were untracked in the repository root and were moved
here unchanged (same names, same bytes) on 2026-09-27 (human decision). DECISIONS #105 records that
on-device LLM vision is not viable on this Pixel; YOLO remains the perception path.

## Files

| File | What |
|---|---|
| `qwen_vision_bench.py` | the #105 Qwen VL benchmark harness (fresh `llama-server` b2351 per image, `/completion` with `multimodal_data`) |
| `qwen_vision_bench_test.py` | its offline test: a tiny fake HTTP child only, never loads llama or model weights |
| `qwen_vision_bench_report.md` | the sourcing report: pinned VL projector downloads, local paths, bytes and the SHA-256 values that #105 cites ("hashes in qwen_vision_bench_report.md") |
| `qwen_vision_bench_review.md` | the AGY `gemini-3.1-pro-high` review of the frozen harness (verdict PASS) |
| `qwen_vision_bench.sha256` | the SHA-256 values of the reviewed frozen candidate (four files above) |
| `qwen35_vision_probe.py` | the earlier file-only Qwen3.5-2B probe (real robot client, scratch server) |

## Reviewed hashes

`qwen_vision_bench.sha256` records the reviewed hashes. Checked from this folder on 2026-09-27 with
`sha256sum -c qwen_vision_bench.sha256`:

| File | Result |
|---|---|
| `qwen_vision_bench_test.py` | OK |
| `qwen_vision_bench_report.md` | OK |
| `qwen_vision_bench_review.md` | OK |
| `qwen_vision_bench.py` | **FAILED**: reviewed `de8e60ce6b0cc8c4cdfce95ec1abdcce32d511a36d5068b4b3ac5db9d3010b9a`, archived `d9ba7661fc25d8d9a50b83ea84f20c0c0e2817d89439a4ff1689d38742732f28` |

The `qwen_vision_bench.py` difference is expected and consistent with #105's post-review
health-timeout fix ("its health-timeout bug fixed"); its file mtime is 2026-09-16 15:26 UTC, after the
review files (2026-09-14 23:53 UTC). The edited version was **NOT re-reviewed**.
No copy of the reviewed `de8e60ce…` version is archived. `qwen35_vision_probe.py` is not listed in
`qwen_vision_bench.sha256` and has no recorded review.

## Running

The scripts were written for the repository root: they read `bench_photos/` next to themselves, and
`qwen35_vision_probe.py` imports `load_robot_module` from the root `run_cycle_safety_test.py`.
They are archived as evidence, not as runnable tools from this folder. Do not run them against the
robot; they start `llama-server` and load multi-GB model weights.

## Evidence gaps

- No benchmark result files are included. Where the results of the #105 runs (Gemma mmproj tests,
  the Qwen3.5-2B proof cell, the stopped 4B sweep) are stored is **UNKNOWN**.
- The edited `qwen_vision_bench.py` was not re-reviewed.
- The projector and model files are not archived (only their hashes, in the sourcing report).
