# VIS-3D: b11716 text and vision benchmark, one test per launch, motors off

Measured 2026-10-23 (owner-run): see [Results](#results-2026-10-23). The owner launches each test from native Termux with no agent resident.
No motors, no camera, no robot mission. Nothing runs after a test; each launch is one server, one block, then exit.

`vis3d.py` imports the VIS-3C runner `../../llm_objective_setting/conversation/chat_mtp_repro/repro.py` (pinned
SHA-256 `29276a35…`, the file that produced the Oct 23 baseline) and through it `chat_mtp.run_block` unchanged:
5 s `coresidency` sampler, 180 s block, 175 s watchdog KILL, no request starts after 150 s, TERM then KILL, cores 4-7
checked after every request. Only the command, the payload and (D) the endpoint are replaced.

## What each test isolates

| Test | Server (b11716, `~/llama.cpp-vis`) | Requests | Question |
|---|---|---|---|
| A | E2B Q4_0, no projector, MTP off | Oct 14 text workload | b11716 text speed vs b1609 OFF (13.35 tok/s) |
| B | A + Q8_0 projector resident | same, no image | A vs B: cost of the resident projector |
| C | B + Q8_0 E2B MTP drafter | same, no image | B vs C: MTP with the projector resident; C vs b1609 ON (18.55 tok/s) differs by build **and** projector |
| D | B + `--reasoning off -lv 4` | 8 fixed frames, Gemma 4 chat template | image latency and answer quality |

Commands (`vis3d.COMMANDS`, test-checked):
- A = the frozen VIS-3C OFF command with only the binary replaced, plus `--metrics`:
  `taskset -c 4-7 ~/llama.cpp-vis/build/bin/llama-server -m ~/models/gemma-4-E2B-it-Q4_0.gguf --port 8080 --ctx-size 4096 --host 127.0.0.1 --threads 4 --threads-batch 4 --parallel 1 --swa-full --cache-ram 0 --metrics`
- B = A + `--mmproj ~/models/vis0/mmproj-gemma-4-E2B-it-Q8_0.gguf --image-min-tokens 70 --image-max-tokens 70`
- C = B + `--model-draft ~/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3` (the VIS-3C ON flags)
- D = B + `--reasoning off -lv 4`

`--metrics` only enables `/metrics`, whose `llamacpp:spec_decode_num_drafts_total` is b11716's exact count of
speculative verification steps (`server-context.cpp`: `n_draft_verif_steps += 1` per verification). b1609 has no such
counter: its ms/pass is the proxy decode ms / (generated − accepted), labelled `ms/pass*` and PROXY in the report.

## Text workload (A, B, C): the Oct 14 payload, kept for speed only

Before starting, `repro.frozen_problems()` must return nothing: prompt texts, ids, payload, model and drafter hashes and
block constants equal the archived Oct 14 `run.json` (SHA-256 `8cc59ac3…`). Every request is
`{"prompt": "<start_of_turn>user\n<text><end_of_turn>\n<start_of_turn>model\n", "n_predict": 512, "temperature": 0.7, "stop": ["<end_of_turn>"], "stream": false}`
to `/completion`. This legacy Gemma 3 template is wrong for Gemma 4 (VIS-0: plain-text markers, 9/10 degraded
replies). It is kept only for an apples-to-apples speed comparison; A–C replies are not judged for quality.

## Vision workload (D): Gemma 4 template

- Endpoint `/v1/chat/completions`; the server applies the GGUF Gemma 4 template (`<|turn>user … <turn|>`).
  `--reasoning off` sets `enable_thinking=false`.
- After the first reply the runner records `/apply-template` for the text part. A run is NOT VALID unless that output
  starts with `<|turn>user\n`, contains `<turn|>` and has no `<start_of_turn>`.
- Request: one user message `[image_url (data:image/jpeg;base64), text]` with the VIS-0 photo-only instruction
  "Describe in one or two short sentences what is in front of the robot. Mention only what you can see."
  Settings: temperature 0, top_k 1, seed 0, max_tokens 96, cache_prompt false. Payloads are encoded before the block.
- Budget: 70 image tokens (min = max). Existing evidence:
  - VIS-0 and VIS-1 logged 63 tokens per 482×640 frame.
  - VIS-1 V2 (Q8, 70): encode about 2.1–2.7 s, wall 8.7–13.6 s with a YOLO list.
  - VIS-1 V3 (Q8, 140): 154 tokens, encode about 7 s.
  - VIS-0 grading: 140 was "slightly more detail", 280 not better.
  - 70 keeps 8 frames well inside the 150 s start window.
- YOLO lines are not added: VIS-0 found the model copies YOLO's labelling errors.
- Frames: 8 git-tracked M2-Q1 frames, `relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__*`.
  `references.json` holds each frame's source photo, source and frame SHA-256 (test-checked against that run's
  `results.json`; the runner refuses a changed frame) and a reference description, key objects and robot context.
  The **Coder drafted these references by viewing the frames; the owner or Local AI confirms or corrects them before
  grading**. Filenames are not ground truth (for example, `empty_2` shows a wardrobe, suitcases and clothes).
- One block, no warm-up image: the report shows the first image separately and the median of images 2–8.

What D measures and how:

| Quantity | Source | Kind |
|---|---|---|
| load | spawn → `/health` ok | measured (runner clock) |
| image encode | `-lv 4` log: "encoding mtmd batch" → "decoding image batch" timestamps | measured (server log timestamps) |
| image tokens, image-token decode | `-lv 4` log: `n_tokens_batch`, "image decoded … in N ms" | logged by the server |
| prompt | server `prompt_ms` (encode + image decode + text prefill) | server timing |
| text prefill | prompt − encode − image decode | **derived** |
| generation | `predicted_n` / `predicted_ms` | server timing |
| wall | client request time | measured |
| TTFT | — | not measured (not streamed) |
| memory, thermal | VmHWM, 5 s PSS, MemAvailable, skin, status, caps | as VIS-3C |

`-lv 4` adds about 25 log lines per request and none per token (VIS-1 V2 log), but D's decode speed is not compared
with B. Image timings are attributed by order of the server's "processing task" sections. If their count differs
from the number of replies, nothing is attributed and the run is NOT VALID. Empty replies and replies cut at
`max_tokens` are labelled `EMPTY` / `INCOMPLETE (length)`; unanswered frames are listed as `NO RESPONSE`.

## Identity checks (refused or NOT VALID otherwise)

- Hashed before every launch:
  - model `8e30dff3…`, drafter `9eba8199…` (as VIS-3C);
  - Q8 projector `9406f99c…` (`vis1/mmproj.json`; ggml-org rev `b4243c15`);
  - `llama-server` and the seven b11716 libraries (pinned in `BUILD_SHA256`; equal to the VIS-2 manifest).
- `run.json` also records:
  - the build's `.git/HEAD` (expected `797de63b…`, tag b11716) and its CMake flags;
  - environment variables `LLAMA_*`, `GGML_*`, `OMP_*`, `KMP_*`, `LD_LIBRARY_PATH` (any of these set: refused) and
    `LD_PRELOAD` (termux-exec, recorded only).
- After the first reply:
  - `/props` `modalities.vision` and `/slots` `speculative` must equal the test's intent;
  - every mapped `libggml*`/`libllama*`/`libmtmd*` must come from `~/llama.cpp-vis/build/bin`.
- From the server log: "loaded multimodal model '<Q8 path>'" iff a projector is intended; "loading draft model
  '<drafter>'" iff MTP is intended.
- MTP is reported only if the server also drafted (`draft_n` > 0) and counted verification steps; drafting without
  the drafter is NOT VALID.
- Also checked as in VIS-3C: all requests answered in the block; Android status 0 at start; server on cores 4-7 at
  start and end; no agent and port 8080 free after the stop; battery Discharging in every sample.

## Owner commands (native Termux; one test per command; nothing follows automatically)

Preflight requirements (checked by `oneshot.sh` and `vis3d.py`; a failed check starts nothing):
- native Termux, root via `su`;
- no claude/agy/node/codex process; no llama-server, chat.py or benchmark runner; port 8080 free;
- the single-run lock is free;
- battery ≥ 80 % and Discharging (charger unplugged), checked twice;
- Android thermal status 0, VIRTUAL-SKIN readable, battery below 45 °C, the 5 s CPU-zone log fresh;
- cores 4-7 allowed (screen on, Termux in front);
- the frozen workload and all pinned hashes as above.

The screen timeout is set to maximum and restored and read back at exit.

```bash
bash ~/robot/benchmark/gemma_vision/vis3d/oneshot.sh --test A
bash ~/robot/benchmark/gemma_vision/vis3d/oneshot.sh --test B
bash ~/robot/benchmark/gemma_vision/vis3d/oneshot.sh --test C
bash ~/robot/benchmark/gemma_vision/vis3d/oneshot.sh --test D
```

Each takes about 2–3 min: 6 s logger start, about 40 s hashing, load, then the requests (A/B about 1 min, C under
1 min, D about 1.5 min). For matched pairs start the next test at a skin within 1 °C of the previous one; the report
shows both.

Folders: `~/robot_runs/vis3d/<UTC>_<test>/` holds:
- `run.json` (command, payload example, prompt texts or frame hashes, weight, build and code hashes, build info, env);
- `prompts.jsonl` (each reply with its server timings and probes);
- `server_block00_<test>.log`;
- `result.json` (block record with the 5 s samples, start and end conditions);
- `report.txt`.

Launcher console and `thermal.log` go in `~/robot_runs/vis3d/`. The VIS-3C folders are only read.

## Report-only (reads folders, starts nothing, no root)

```bash
python ~/robot/benchmark/gemma_vision/vis3d/vis3d.py --report ~/robot_runs/vis3d/<UTC>_A ~/robot_runs/vis3d/<UTC>_B ~/robot_runs/vis3d/<UTC>_C
python ~/robot/benchmark/gemma_vision/vis3d/vis3d.py --report ~/robot_runs/vis3d/<UTC>_D
```

The report always includes the VIS-3C baseline rows (`20261023T124846Z_OFF`, `20261023T132032Z_ON`) recomputed by
`repro.summary`.

- **Throughput** follows the Oct 14 rule: token-weighted generation tok/s over replies of at least 10 tokens.
  `pooled` uses all tokens.
- **Comparisons:** B/A, C/B, A/OFF and C/ON, each with three separate lines:
  - `configuration`: what differs; A/B and B/C may differ only by their flags;
  - `conditions`: MATCHED or UNMATCHED (VIS-3C rule: start skin within 1 °C, status 0, battery within 10 %, equal
    CPU caps);
  - both validity labels.

  A ratio is an observation, not a cause. In particular, C/ON mixes build and projector residency.
- **Test D:** per-image table, first image vs median of the rest, resources, and a grading sheet. The grading sheet
  gives each frame's reference, key objects and reply, with blanks for key objects named, hallucinated objects and
  robot-context errors.

## Limitations and stop conditions

- **Single short blocks:**
  - Each test is one 180 s block; thermal equilibrium is not established (#93).
  - Sampling at temperature 0.7 without a seed makes A–C reply lengths differ between runs (as on Oct 14).
  - A ratio from one pair is one observation; repeat only by a new owner decision.
- **Defaults not pinned by the payload:** b11716 sampler defaults are recorded (`/props`) but not forced equal to
  b1609's. VIS-3 found a small EOG tokenizer difference. Compare `prompt tok` (305 on b1609).
- **Not measured:**
  - MTP with images (not part of this sequence);
  - TTFT;
  - per-request power (only the 5 s battery samples).
- **Stop and report rather than rerun** if any of these happens:
  - a preflight refusal that is not the operator's to fix (hash or frozen-workload mismatch);
  - NOT VALID for projector, drafter or library identity;
  - `/props` or `/slots` disagreeing with the command;
  - a server exit before the end of the block (the runner does not query LMK: check logcat by hand);
  - D replies empty or cut on more than 2 of 8 frames;
  - Android status above 0 at a start.

## Results (2026-10-23)

Owner-run from native Termux, no agent resident; one test per launch; source = the reviewed frozen candidate
(`vis3d.py` `cd4995b2…`, `references.json` `7d3d5b77…`, `repro.py` `29276a35…`, recorded in each `run.json`).
Build `~/llama.cpp-vis` at `797de63b…` (labelled b11716; shallow clone, build number not verified locally), not the
`server_manager.py` binary. Run index: [RUN_INDEX.md](RUN_INDEX.md); hashes: [ARTIFACTS.md](ARTIFACTS.md); DECISIONS #143.

Text (A–C, Oct 14 workload with the legacy Gemma 3 markers, replies not graded):

| Run | Test | Prompts | gen tok/s | accept | steps | gen tok | load s | start skin °C | bat % | Label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `runs/20261023T151754Z_A` | A | 13/13 | 13.22 | n/a | 0 | 657 | 5.2 | 25.30 | 82 | VALID |
| `runs/20261023T152217Z_B` (B1) | B | 13/13 | 13.53 | n/a | 0 | 486 | 5.7 | 28.62 | 81 | VALID |
| `runs/20261023T161842Z_B` (B2) | B | 13/13 | 13.46 | n/a | 0 | 546 | 6.4 | 24.23 | 99 | VALID |
| `runs/20261023T162349Z_C` | C | 13/13 | 18.11 | 35.2 % | 341 | 713 | 6.8 | 27.95 | 98 | VALID |

Both B runs are kept. The official A/B/C comparison uses **B2** (`20261023T161842Z_B`): B2/A 1.018, C/B2 1.346,
A/OFF 0.990, C/ON 0.976 (with B1: B1/A 1.023, C/B1 1.339). Every one of these comparisons is `conditions: UNMATCHED`
(start skin and/or battery); intended configuration differences held. The A/OFF and C/ON lines are in the saved
per-run reports. The combined A/B/C reports were shown on the owner's console and not saved; `reports/` holds
report-only regenerations (`vis3d.py --report`, 2026-10-23T19:03:41Z, archival task, no inference) from the unchanged
run folders: [A/B2/C](reports/REGENERATED_20261023T190341Z_combined_A_B2_C.txt) (official) and
[A/B1/C](reports/REGENERATED_20261023T190341Z_combined_A_B1_C.txt). A ratio is an observation, not a cause: these
runs do not show that the projector or the monitoring changes throughput, and projector and MTP memory cost is NOT
ESTABLISHED (single-run PSS samples vary).

Vision (D, `runs/20261023T163445Z_D`, VALID): 8/8 frames answered and complete, 63 image tokens each; first image wall
5.24 s; images 2–8 median wall 4.42 s, encode 1456 ms, image-token decode 847 ms, prompt 2823 ms, generation
13.79 tok/s; load 5.5 s; server VmHWM 3839 MiB, peak 5 s PSS 3708 MiB, min MemAvailable 2196 MiB; skin
26.8 → 31.4 °C, status 0. TTFT and per-request power not measured. Reply quality is NOT scored: references are
Coder-drafted and not owner-confirmed, and there is no rubric yet. Qualitative notes (Local AI) are in #143; one reply
(`roomsig_couch_1`) omitted the close couch and the partly visible person. Obstacle or person detection reliability,
real-time navigation use, sustained heat and co-residency with RobotCam, YOLO and the mission loop are not established.

One earlier launch was refused before any server start (`not started: battery 79 % < 80 %`, console
`runs/oneshot_console_20261023T150730Z.log`): REFUSED, not a measurement.

Review: AGY `gemini-3.8-flash-high`, Ponytail off; round 1 INVALID, round 2 APPROVED WITH NOTES (records in
`reports/`; validity of each round in [RUN_INDEX.md](RUN_INDEX.md)).
