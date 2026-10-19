# TASK VIS-0 — Gemma 4 Template Check & Gemma Vision Feasibility (Motors Off)

**Labels:** Template check VALID (deterministic); vision replies VALID as outputs (temperature 0); timings and memory NOT measurements (agent resident); quality grading pending (Local AI)

## Scope

1. **Step A — Template Check (text only, port 8090):**
   - Saves `GET /props` `chat_template` verbatim.
   - Calls `POST /tokenize` with `"with_pieces": true` on `"<start_of_turn>user"`, `"<end_of_turn>"`, `"<|turn>user"`, `"<turn|>"`, and `"<|turn>model"`, recording whether each turn marker is `ONE special token` or `several plain-text pieces`.
   - Calls `POST /apply-template` with `[{"role": "user", "content": "Hi"}]` and saves the rendered prompt verbatim.
   - Runs a deterministic (`temperature: 0, top_k: 1, seed: 0, max 96 tokens, cache_prompt: false`) text A/B comparison across the 10 target frames using the saved `Y-ONLY` YOLO lines from `benchmark/relate_anything/quality1/runs/20261017T213604Z/results.json`:
     - **OLD:** `POST /completion` with the exact Gemma 3 marker string (`<start_of_turn>` / `<end_of_turn>`) constructed by `chat.build_prompt('gemma', ...)`.
     - **NEW:** `POST /chat/completions` with the same instruction + YOLO lines as a single `user` message so `llama-server` applies the embedded GGUF template.
   - Reports `CONFIRMED` / `REFUTED` / `UNCLEAR` on whether the Gemma 3 markers are wrong for `gemma-4-E2B-it-Q4_0.gguf`.

2. **Step B — Vision Feasibility (port 8090):**
   - Evaluates the 10 640 px frames from `benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/`:
     `bench_photos__blocked_1`, `bench_photos__empty_2`, `bench_photos__person_facing_150cm`, `bench_photos__person_sideon_300cm`, `bench_photos__roomsig_couch_1`, `bench_photos__roomsig_refrigerator_1`, `bench_photos__roomsig_toilet_1`, `bench_photos__roomsig_toilet_2`, `test_photos__chiara_close_back`, `test_photos__abdel_close_right`.
   - Sends `POST /chat/completions` with one user message containing the JPEG as a base64 `data:image/jpeg;base64,...` URI (`image_url`) plus:
     `"Describe in one or two short sentences what is in front of the robot. Mention only what you can see."`
   - Tests supported Gemma 4 image token budgets `70`, `140`, and `280` via `--image-min-tokens <N> --image-max-tokens <N>` on a separate `timeout 3600` server instance on `127.0.0.1:8090` (`--threads 4 --threads-batch 4 --parallel 1 --swa-full --cache-ram 0 --ctx-size 4096 --reasoning off -lv 4`, no CPU pinning).
   - If all three budgets complete without crashing, runs an **optional** pass at budget `140` with `"Objects detected by YOLO:"` plus the saved `Y-ONLY` lines appended.

## Build & Weights (Outside Repo)

- **Separate `llama.cpp` build (`~/llama.cpp-vis`):**
  - Source: `https://github.com/ggml-org/llama.cpp` tag `b11716` (`797de63b2fa7396dcc8caca76834c699ac84e1c3`)
  - Flags: `-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENCL=OFF`
  - Compiler: Termux `Clang 21.1.8` (`aarch64-unknown-linux-android24`)
  - `libggml-cpu.so` instruction counts: `sdot = 1117`, `fmla (8h) = 456`
- **Model GGUF (`~/models/gemma-4-E2B-it-Q4_0.gguf`):**
  - Source: `ggml-org/gemma-4-E2B-it-GGUF` (rev `b4243c156154b6dca9324415f8c7ccc098b4aed1`)
  - Size: `2841481184` bytes
  - SHA-256: `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52`
- **Vision Projector (`~/models/vis0/mmproj-gemma-4-E2B-it-BF16.gguf`):**
  - Source: `ggml-org/gemma-4-E2B-it-GGUF` (rev `b4243c156154b6dca9324415f8c7ccc098b4aed1`; publisher offers `BF16` and `Q8_0`, not `F16`)
  - Size: `986833664` bytes
  - SHA-256: `711e1e8f43fa0664adbac493129be1e6c25b81af4b4cdea97c7d798b25c0a3a4`

## Usage

Run offline unit tests (no server or network):

```bash
/data/data/com.termux/files/usr/bin/python3 -B benchmark/gemma_vision/vis0/test_vis0.py -v
```

Run 2-frame smoke check (requires port 8080 stopped):

```bash
/data/data/com.termux/files/usr/bin/python3 -B benchmark/gemma_vision/vis0/vis0.py --limit 2
```

Run full 10-frame evaluation:

```bash
/data/data/com.termux/files/usr/bin/python3 -B benchmark/gemma_vision/vis0/vis0.py
```

Each run writes `results.json`, `manifest.json`, `server.log`, `run.log`, and `vis0_sheet.pdf` under `benchmark/gemma_vision/vis0/runs/<UTC>/` and copies `vis0_sheet.pdf` to `/storage/emulated/0/Download/vis0/`.
