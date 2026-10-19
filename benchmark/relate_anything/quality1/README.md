# RelateAnything M2 Quality & Gemma A/B Desk Test (`M2-Q1`)

**Status labels:**
- **Accuracy VALID (temperature 0); timings NOT measurements (agent resident); grading pending (Local AI, blind).**
- Motors stay off. No robot files are modified.

## Purpose

Evaluates on every `*.jpg` in `bench_photos/` and `test_photos/` (shrunk to 640 long side, saved at JPEG quality 85, and re-opened):
1. Whether RelateAnything M2 (`relsgg-vits16`, fp32, via `relate.Head` with SHA-256 verification and without thread pinning) produces accurate object-relation lines (`sheet_m2.pdf`).
2. Whether adding M2 relation lines (`Y+M2`) helps Gemma 4 E2B Q4_0 (`setup_q4`) describe a scene better than YOLO detections alone (`Y-ONLY`), presented blind as `X` and `Y` (`sheet_ab.pdf`).

Grading is performed blind by Local AI from `sheet_m2.pdf` and `sheet_ab.pdf`. The harness does not grade replies.

## Prerequisites

1. Ensure the robot default server (`Gemma 4 E2B Q4_0`) is running on `127.0.0.1:8080` in native Termux:
   ```bash
   cd ~/robot && python3 server_manager.py setup_q4
   ```
2. Use the Termux Python interpreter (`/data/data/com.termux/files/usr/bin/python3`), which provides `onnxruntime`, `numpy`, and `PIL`.

## How to Run

Run offline unit tests (fakes only, no network or model files required):
```bash
/data/data/com.termux/files/usr/bin/python3 -B benchmark/relate_anything/quality1/test_quality1.py -v
```

Smoke run (first 3 sorted photos):
```bash
/data/data/com.termux/files/usr/bin/python3 -B benchmark/relate_anything/quality1/quality1.py --limit 3
```

Full run (all sorted photos in `bench_photos/` and `test_photos/`):
```bash
/data/data/com.termux/files/usr/bin/python3 -B benchmark/relate_anything/quality1/quality1.py
```

## Outputs (`runs/<UTC>/`)

- `frames/` — 640-long-side JPEG (quality 85) frames used by YOLO and M2.
- `results.json` — per-photo hashes, detections, M2 raw top-10 rows and shown text, 2× determinism flags, Gemma prompts, and blind replies `X` and `Y`.
- `key.json` — blind mapping `photo -> {"X": "Y-ONLY"|"Y+M2", "Y": ...}` generated with `random.Random(20261017)` (never included in PDFs or reports).
- `sheet_m2.pdf` — one page per `>= 2`-box photo showing the numbered frame, shown M2 text, and raw top-10 M2 rows, plus one summary page listing `< 2`-box photos.
- `sheet_ab.pdf` — one page per `>= 2`-box photo showing the numbered frame and only the blind replies `X` and `Y`.
- `manifest.json` — base commit, file SHA-256 hashes, model/GGUF hashes, `/props` summary, interpreter versions, photo hashes, and UTC timestamps.
- `run.log` — run progress log.
