# TASK VIS-0 CODER REPORT — Gemma 4 Template Check & Vision Feasibility

**Execution Profile:**
- **Role:** Coder (feasibility harness author and desk-test runner)
- **Platform:** AGY/Antigravity CLI (proot Debian, `/termux-home/robot`, mode `accept-edits`)
- **Model:** `gemini-4-argon-medium` (Gemini 4 Argon (Medium))
- **Ponytail:** lite
- **Reviewer Profile:** Codex CLI (`codex exec`), `gpt-6.1-sol`, effort `medium`, Ponytail `off`, run in isolated copies under `/termux-home/review/vis0/`

---

## 1. Executive Summary

1. **Step A (Template Check): CONFIRMED.**
   - Gemma 3 turn markers (`<start_of_turn>` and `<end_of_turn>`) are **not** special tokens in our deployed GGUF (`~/models/gemma-4-E2B-it-Q4_0.gguf`). They are tokenized into 7 to 8 plain-text pieces (`<` `start` `_` `of` `_` `turn` `>`).
   - Gemma 4 turn markers (`<|turn>` ID `105` and `<turn|>` ID `106`) are single special tokens in the model vocabulary.
   - The GGUF Jinja `chat_template` explicitly formats user turns as `<|turn>user\n{content}<turn|>\n<|turn>model\n`.
   - On text A/B across the 10 M2-Q1 target frames:
     - **OLD** (`/completion` with Gemma 3 prompt markers via `chat.build_prompt`): Degraded in 9/10 frames into broken repetition loops (e.g. `<start_of_turn>user\nWhat is in front of the robot?...`) or hit the 96 token limit without finishing.
     - **NEW** (`/chat/completions` using GGUF chat template): Clean, deterministic, concise 1–2 sentence descriptions stopping at `<turn|>` in 19–40 completion tokens.
   - This directly explains the degraded replies in M2-Q1 and the leaked turn markers in Decision #115.

2. **Step B (Vision Feasibility): VALID AS OUTPUTS, ZERO CRASHES.**
   - Model vision ran on `llama-server` (`b11716`) with `mmproj-gemma-4-E2B-it-BF16.gguf` across all 10 target frames at budgets 70, 140, 280, and the optional 140 + YOLO pass.
   - **Crashes:** `0` crashes across all 40 calls. Zero SIGSEGV, zero process aborts during inference.
   - **Image Token Allocations:**
     - Budget 70: `63` image tokens allocated per 640×482 frame.
     - Budget 140: `154` image tokens allocated per 640×482 frame.
     - Budget 280: `300` image tokens allocated per 640×482 frame.
     - Optional 140 + YOLO: `154` image tokens allocated.
   - **Quality:**
     - Vision replies at temperature 0 produce coherent, grounded scene descriptions (e.g., identifying laptops, tables, couch, refrigerators with open door and contents, washing machine, toilets, people, shirt colors).
     - At budget 70, prompt eval + image decode took ~21s wall per frame. At budget 140, ~54s. At budget 280, ~109s. (All timings and memory are NOT measurements; agent resident).

---

## 2. Artifact & Git Provenance

- **Base commit:** `33c2c78d2f05bac27820d7511588ecb66ccf1f8e`
- **Reviewed Code SHA-256:**
  - `benchmark/gemma_vision/vis0/vis0.py`: `ea3d1e38477b0ff2f85a04ed828da3c60ce7264d805b409f7c0d84e010aec50a` (1,437 lines)
  - `benchmark/gemma_vision/vis0/test_vis0.py`: `0bc938d604278317633175110dd2e7119fcfe98887fe59d547ba9886a987ad21` (462 lines, 15/15 tests passing)
  - `benchmark/gemma_vision/vis0/README.md`: `da2334b84d212bb11c15e78d0fbf84e869c06772a979713dcd94f96df0078fdf` (60 lines)
- **Tracked files:** 100% clean, read-only. `git status` shows no modified tracked files.
- **Untouched directories:** `~/llama.cpp`, `~/llama.cpp-b1609-dotprod`, `~/llama.cpp-upstream`, and `~/models/*` (outside `vis0/`) remained strictly untouched.
- **Run Directory:** `/termux-home/robot/benchmark/gemma_vision/vis0/runs/20261018T010701Z/`
  - `manifest.json` (12,241 bytes)
  - `results.json` (104,809 bytes)
  - `run.log` (7,522 bytes)
  - `server.log` (318,516 bytes)
  - `vis0_sheet.pdf` (1,184,772 bytes, 11 pages, SHA-256 `5aa8cd5bbde16daba5f2e195dca10289dc6a4331453ec8c33fe5ef33636f990f`)
- **Owner Download Sheet:** `/storage/emulated/0/Download/vis0/vis0_sheet.pdf` (identical SHA-256).

---

## 3. Build & Download Specifications

- **llama.cpp Build Directory:** `/data/data/com.termux/files/home/llama.cpp-vis`
  - Repo: `https://github.com/ggml-org/llama.cpp`
  - Tag: `b11716` (commit `797de63b2fa7396dcc8caca76834c699ac84e1c3`)
  - Toolchain: Clang 21.1.8 (native Android aarch64 via Termux)
  - CMake flags: `-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENCL=OFF`
  - Targets: `llama-server`, `llama-mtmd-cli`
  - Library `libggml-cpu.so`: SHA-256 `9e4e11cf979ef78e1aa201ecaed67308520ffa9a6e3f19d028cfb1586b8ab333`
  - Instructions count: `sdot = 1117`, `fmla_8h = 456`
- **Model GGUF:** `/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf`
  - Source: `ggml-org/gemma-4-E2B-it-GGUF` (rev `b4243c156154b6dca9324415f8c7ccc098b4aed1`)
  - Size: 2,841,481,184 bytes, SHA-256 `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52`
- **Multimodal Projector (mmproj):** `/data/data/com.termux/files/home/models/vis0/mmproj-gemma-4-E2B-it-BF16.gguf`
  - Source: `ggml-org/gemma-4-E2B-it-GGUF` (rev `b4243c156154b6dca9324415f8c7ccc098b4aed1`; publisher offers BF16 and Q8_0, no F16)
  - Size: 986,833,664 bytes, SHA-256 `711e1e8f43fa0664adbac493129be1e6c25b81af4b4cdea97c7d798b25c0a3a4`

---

## 4. Test Server Configuration (`ManagedTestServer`)

Wrapped command:
```bash
timeout 3600 /data/data/com.termux/files/home/llama.cpp-vis/build/bin/llama-server \
  -m /data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf \
  --mmproj /data/data/com.termux/files/home/models/vis0/mmproj-gemma-4-E2B-it-BF16.gguf \
  --host 127.0.0.1 --port 8090 --ctx-size 4096 \
  --threads 4 --threads-batch 4 --parallel 1 --swa-full --cache-ram 0 \
  --reasoning off -lv 4 \
  --image-min-tokens <BUDGET> --image-max-tokens <BUDGET>
```
- Port: `8090` only. Port `8080` verified stopped before every launch.
- Flags relied on from `--help`:
  - `--image-min-tokens N` / `--image-max-tokens N`: sets token budget bounds for dynamic vision encoding.
  - `--reasoning off`: disables auto-injection of `<|think|>` channel tokens, which otherwise consume all 96 max tokens.
  - `-lv 4`: enables logging of `mtmd` image token batch size (`n_tokens_batch`) and encode timing.

---

## 5. Review History (Codex `gpt-6.1-sol`, medium effort, Ponytail off)

- **Round 1:** `VERDICT: REVISE` (4 P2 findings: `--out` boundary validation; HTTP error vs socket crash handling; frame stem identity & 640px JPEG dimension verification; server log qualification banner). All addressed.
- **Round 2:** `VERDICT: REVISE` (1 P2 finding: `check_port_8080_stopped` caught `URLError` which caught `HTTPError` 503; needed fail-closed HTTP rejection and socket checks). Addressed with explicit `HTTPError` rejection, `ConnectionRefusedError` requirement, and socket bind check. Unit tests expanded to 15/15.
- **Round 3:** `VERDICT: APPROVED`. All pre/post hashes, line counts, and citations verified against the isolated checkout. No candidate code changed between Round 3 approval and the full run.

---

## 6. Status & Handoff

- Port `8090` test server has fully exited and is down (`8090_DOWN_CONFIRMED`).
- Port `8080` remains down (`8080_DOWN_CONFIRMED`).
- Disk free space: `12 GB` available (`df -h ~` shows 90% used on `/`).
- **Owner Action:** The owner may now restart `setup_q4` on port `8080` if wanted for regular robot operations.
