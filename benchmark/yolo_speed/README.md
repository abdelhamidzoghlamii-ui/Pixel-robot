> **Heat readings in this archive are not trusted** (zone9/10/11 are CPU-core sensors). Speed, RAM and power results remain valid as measured; CPU capping during these runs is unknown. See benchmark/thermal_char/HEAT_EVIDENCE.md.

# YOLO detector speed ladder (2026-09-28)

Status: **complete** (one unattended timed run). Research only: offline frames, no motors, no `main.py`,
no robot-code or deployed-model change.

## What

`yolo_speed.py` times 14 detector configs on 20 fixed frames, one cold config at a time in the ladder's
style (`../strategic_selector/ladder/ladder.py`, SHA-256 `c7e6d3c9…`): cores 4–7 check, thermal gate
z9 ≤ idle + 4 °C, page-cache drop through the `oneshot.sh` handshake, then a fresh worker process doing
5 warm-up + 60 timed frames (the 20 frames cycled). The worker follows `detect_person.detect_scene`
(`85be7d1b…`): same ONNX Runtime default CPU session, preprocessing, confidence filter and NMS.
Configs: stock Ultralytics yolo11 / yolo26 n/s/m exported at 320 and 640 (`prep/export_models.py`,
`prep/export_log.txt`), the deployed `/termux-home/robot/yolo11m.onnx` at 640, and an int8
dynamic-quantized copy of it. `run_yolo_speed.sh` started it from native Termux through
`oneshot.sh` (the live copy is byte-identical to
`../strategic_selector/ladder/oneshot_executed_conversation_run.sh`, `db1f3156…`).

## Interpreter of the timed worker: native Termux Python

Every block in `runs/yolo_speed_20260928T182649Z/blocks.jsonl` records ONNX Runtime 1.25.1; line 1
(yolo11n_320) contains:

```
"ort": "1.25.1", "providers": ["CPUExecutionProvider"]
```

and all 14 rows have `"ort": "1.25.1"`. On this phone only native Termux Python has that version
(`/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime-1.25.1.dist-info`); Debian's
`python3` has no `onnxruntime` and the Debian `yolo_bench` venv has 1.30.0. This matches the script:
`NATIVE_PY = "/data/data/com.termux/files/usr/bin/python"` (`yolo_speed.py` line 31), used to launch
each `--worker`. The result files do not record the interpreter path itself, so the attribution rests on
the version. The parent (scheduler, thermal gate, report) is Debian `python3`, and the worker is
launched from inside proot, so absolute times may differ from the robot's own process (noted by the
Reviewer in rounds 3–5).

## Results

`runs/yolo_speed_20260928T182649Z/report.txt` — median / P95 ms over 60 timed frames, ONNX Runtime default threads:

| Config | pre | infer | post | total | RSS MiB | dets/frame |
|---|---|---|---|---|---:|---|
| yolo11n_320 | 193/198 | 32/52 | 7/7 | 233/251 | 187 | 2.20 |
| yolo26n_320 | 193/199 | 34/51 | 7/7 | 234/251 | 188 | 2.10 |
| yolo26s_320 | 198/203 | 62/89 | 7/7 | 268/292 | 230 | 2.40 |
| yolo11s_320 | 199/204 | 67/86 | 7/7 | 273/295 | 229 | 2.90 |
| yolo26n_640 | 216/222 | 94/111 | 29/29 | 338/358 | 240 | 2.45 |
| yolo11n_640 | 216/221 | 110/122 | 30/30 | 355/369 | 238 | 2.40 |
| yolo11m_320 | 202/208 | 205/229 | 8/8 | 415/441 | 299 | 2.45 |
| yolo26m_320 | 203/209 | 211/232 | 7/8 | 421/443 | 300 | 2.90 |
| yolo26s_640 | 221/227 | 263/299 | 29/30 | 513/554 | 331 | 3.25 |
| yolo11s_640 | 220/224 | 263/301 | 31/32 | 515/550 | 325 | 3.10 |
| yolo11m_640 | 230/238 | 639/671 | 33/36 | 906/942 | 472 | 3.45 |
| yolo26m_640 | 233/241 | 639/666 | 35/37 | 906/935 | 482 | 3.85 |
| deployed_yolo11m_640 | 232/244 | 641/672 | 35/37 | 908/931 | 472 | 3.45 |
| deployed_yolo11m_640_int8dyn | 253/297 | 1786/1900 | 36/45 | 2081/2262 | 436 | 3.25 |

The deployed config matches stock yolo11m_640 in time, RSS and detections. Int8 dynamic quantization is
2.8× slower on this CPU. Preprocessing (~190–250 ms, decoding the ~2 MB JPEGs) is the largest share for
every n model and both s models at 320; at 640 the s models spend more in inference (~263 ms vs ~220 ms). Detection counts are for speed context only; this is not an accuracy benchmark.

## Frames

`frames/frame_01…20.jpg` are 20 of the 38 `bench_photos/` images (mapping and SHA-256 in
`frames/SOURCES.txt`, also `frames/SHA256SUMS`); publishing them was approved by the owner. No frame has
EXIF GPS tags (checked at archive time). A first set captured live with `termux-camera-photo`
(`prep/capture_frames.py`) came out black because the camera was covered; those frames are not archived,
only `frames_black_camera_covered/SHA256SUMS`. `prep/capture_timing.json`: each call took 2.1–3.4 s
(`take_photo` incl. 0.5 s sleep 2.7–3.9 s), and 3 of 20 calls (frames 14, 16, 18) left no usable file
(`ok: false`, file missing or ≤ 1000 bytes).

## Deployed model provenance

`reviews/review_request5.md` records: "Stock vs deployed yolo11m: 245 vs 245 initializers, 20133752
params each, sorted per-tensor |w| sums max rel diff 0.0; raw outputs on frame_06/frame_14: max abs diff
0.00122 / 0.00061". The deployed `yolo11m.onnx` therefore appears to be the stock COCO yolo11m, which
conflicts with `docs/STATUS.md` ("fine-tuned, not stock", citing DECISIONS #97). Reported to Doc Keeper; not resolved here.

## Contents

| Path | What |
|---|---|
| `yolo_speed.py`, `run_yolo_speed.sh` | runner and run script (the executed versions, `3c3f3bce…`, `5fbd8858…`) |
| `runs/yolo_speed_20260928T182649Z/` | `blocks.jsonl` (per-frame rows), `report.txt`, `run_*.json` (input hashes), empty worker stderr logs |
| `runs/*.stdout.txt`, `runs/*.stderr.txt`, `runs/oneshot_console_*.log` | run output and the `oneshot` console |
| `frames/` | the 20 benchmark frames, `SOURCES.txt`, `SHA256SUMS` |
| `frames_black_camera_covered/SHA256SUMS` | hashes of the unused black frames |
| `models/SHA256SUMS`, `models/SIZES.txt` | the 13 exported ONNX files (weights not in Git) |
| `prep/` | frame capture script and timing, export script and log |
| `reviews/` | the five review rounds: requests, final verdicts, stdout, stderr |

[RUN_INDEX.md](RUN_INDEX.md) lists the run; [ARTIFACTS.md](ARTIFACTS.md) hashes every file.

## Review status

Codex CLI `gpt-6-sol`, read-only sandbox, medium effort, fresh session per round. Rounds 1–4
**CHANGES REQUIRED** (untracked scripts; resume provenance, duplicate rows, model hash timing,
incomplete-block and torn-line handling); round 5 **APPROVE WITH NOTES — required changes: none**
(`reviews/review5_final.md`). The run used the round-5 candidate (hashes in `review_request5.md` equal
`run_*.json`). The run results themselves were not reviewed.

## Evidence gaps

- The interpreter is inferred from the recorded ONNX Runtime version, not recorded directly.
- The worker runs under proot; absolute latencies may differ from the robot's native process.
- 60 timed frames cycle 20 images, so each image is timed three times.
- Model files, `.pt` weights, venv and black frames are not in Git; their hashes are in `ARTIFACTS.md`.
