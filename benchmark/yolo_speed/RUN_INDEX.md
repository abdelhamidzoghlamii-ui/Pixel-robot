# YOLO speed — run index

| Step | When (UTC) | Output | Status |
|---|---|---|---|
| frame capture with `termux-camera-photo` (camera covered) | 2026-09-28 ~10:51 | `prep/capture_frames.py`, `prep/capture_timing.json`, `frames_black_camera_covered/SHA256SUMS` | frames unusable (black); timings kept |
| 20 frames from `bench_photos/` | 2026-09-28 10:57 | `frames/` | complete |
| ONNX exports + int8 copy | 2026-09-28 10:49–10:51 | `prep/export_models.py`, `prep/export_log.txt`, `models/SHA256SUMS` | complete |
| review rounds 1–5 | 2026-09-28 16:59–17:15 | `reviews/` | round 5 APPROVE WITH NOTES |
| timed run, 14 configs (oneshot 18:21:42Z, 5 min idle) | 2026-09-28 18:26:49–18:46Z | `runs/yolo_speed_20260928T182649Z/`, `runs/*.stdout.txt`, `runs/oneshot_console_20260928T182142Z.log` | complete, no resume, stderr empty |
