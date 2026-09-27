# Frozen Qwen vision benchmark — sourcing and verified images

Prepared 2026-09-14. Coder: Codex CLI / gpt-6-astra / High / Ponytail lite.
Reviewer (updated human instruction): AGY / gemini-3.1-pro-high / effort High / Ponytail off (current CLI --help exposes --effort).
No sweep, model inference, production edits, commits or pushes performed for this task.

## Repository and build

- Local HEAD: 888562ec693991eabc25b234d41b700e20ff3658.
- Fresh git ls-remote origin HEAD: f0f9cde7d5a9fcdacbbdaa25c890b55c4370f27a.
  The expected 888562e matches local HEAD, not the remote; no checkout or fetch was performed.
- Existing changes to main.py/server_manager.py and existing untracked probe artifacts were preserved.
- Executed only llama-server --version: 0.4.0-dev (build 2351, commit 790cf51a).
- Installed source HEAD: 790cf51aabd61763486050dec7451d9147cb7c61.

## Matching projectors

All five requested model variants have an available F16 VL projector. There are no
unavailable-projector variants in this set. The three 4B language-model quantizations
share the same 4B vision projector; three duplicate copies are unnecessary.
All local model and projector hashes were verified against the pinned publisher LFS hashes.

### Qwen3.5-0.8B — Q4_K_M

- [Pinned projector download](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/mmproj-F16.gguf)
- Local path: `/data/data/com.termux/files/home/models/qwen35/mmproj-Qwen3.5-0.8B-F16.gguf`.
- Bytes: 204987232; downloaded and rehashed.
- SHA-256: `56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453`.

### Qwen3.5-2B — Q4_K_M

- [Pinned projector download](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/mmproj-F16.gguf)
- Local path: `/data/data/com.termux/files/home/models/qwen35/mmproj-F16.gguf`.
- Bytes: 668227264; existing file reused and rehashed.
- SHA-256: `7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7`.

### Qwen3.5-4B — Q4_K_M, Q5_K_M and Q6_K

- [Pinned projector download](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/mmproj-F16.gguf)
- Local path: `/data/data/com.termux/files/home/models/qwen35/mmproj-Qwen3.5-4B-F16.gguf`.
- Bytes: 672423616; downloaded and rehashed.
- SHA-256: `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864`.

## License check against docs/LICENSES.md

Each pinned Unsloth model card declares Apache-2.0 and identifies the corresponding
Qwen upstream model. The full upstream LICENSE files were opened and read at these revisions:

- [Qwen3.5-0.8B LICENSE](https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17/LICENSE) — Apache-2.0.
- [Qwen3.5-2B LICENSE](https://huggingface.co/Qwen/Qwen3.5-2B/resolve/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE) — Apache-2.0.
- [Qwen3.5-4B LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE) — Apache-2.0.

Apache-2.0 is allowed by docs/LICENSES.md. These weights remain external prototype
assets under ~/models; no weights or third-party source are copied into the repository
or an APK. The APK dependency ledger is unchanged. Redistribution requires the Apache
license text, retained attribution, modification notices, and any applicable NOTICE.

## Verified images

The correct directory is ~/robot/bench_photos; ~/bench_photos does not exist.
All four files are readable JPEGs, 4080 x 3072, EXIF orientation 6. Inspected as raw
encoded pixels without EXIF rotation, exactly the orientation passed to mtmd.

labels.csv has target_object/category/detector labels, but NO position column.
Object identity was cross-checked visually; the positions below are a frozen visual
supplement, not positions claimed to come from CSV. No source labels were modified.

| Photo | Relevant CSV evidence | Visually verified expected answer |
|---|---|---|
| person_facing_060cm.jpg | target person; detected person;chair;tv | PERSON=L, TV=L (partial TV) |
| roomsig_refrigerator_1.jpg | target refrigerator | REFRIGERATOR=C |
| roomsig_toilet_3.jpg | target toilet | TOILET=C |
| empty_2.jpg | category empty; suitcase;umbrella;bowl;backpack | NONE of the four queried classes |

Position is the center of the visible bounding box in horizontal thirds. The person
and partial TV lie in the left third of the raw sideways image. Refrigerator and
toilet centers lie in the center third. Empty does not mean no objects at all.
This small set contains no positive right-third case; it is not a comprehensive accuracy evaluation.
The person filename is not distance ground truth (CSV says an estimated 180 cm);
distance is not scored.

### Frozen image SHA-256

- `person_facing_060cm.jpg`: `6e7882317f0620a383a2da279d58a3f82c357a978bc47221947af3460e67bcdb`
- `roomsig_refrigerator_1.jpg`: `ba8abe530b043baf8d0d1b10ddfb0e2493f562cdd8a9817044c5b128bda0317f`
- `roomsig_toilet_3.jpg`: `dca6d9a6228670a13547b808a75fe3f59aa9f9528d2c696e08a191ea5ddaa77e`
- `empty_2.jpg`: `68c2ed00e7635d6884377b4c34a90617e238e63749318dc602d9f49423246f2a`
- `labels.csv`: `841f3d93f6b1f0d0ca43f2214ca073ad4e2672bc5857b634ef101621b456f70b`

## Protocol and output interpretation

- 5 models x 4 requested image budgets x 2 passes x 4 photos = 160 attempted requests.
- Budget sets --image-min-tokens and --image-max-tokens together. Output cap is separately
  fixed at 64 tokens. Image resize rounding can produce a different actual image-token
  count (prior 1024 logs show 972); decoding batch counts and full logs are retained.
- /completion receives prompt: {prompt_string, multimodal_data: [base64 JPEG]}.
  LLAMA_MEDIA_MARKER=[img-1], one marker and one image, ChatML user/assistant wrapper,
  non-thinking assistant prefill, Qwen stop tokens, temperature 0, seed 123, cache_prompt false.
- This is a visual identification task with a short frozen prompt; it does not import the
  robot navigation prompt. It is not a reproduction of the earlier probe's full prompt latency.
- b2351 /props must report modalities.vision=true and the expected media_marker before
  inference. Merely starting a PID is not projector-loaded evidence.
- A fresh server is launched for EACH photo in BOTH passes. This isolates pending work after
  client timeout and avoids unequal prompt reuse. Default server startup warmup remains;
  there is no client warmup. Startup time is recorded separately from HTTP latency.
- Unbounded has no HTTP deadline. The 10s pass uses a total wall deadline through response
  body receipt and parsing, not merely a socket inactivity timeout. The client connection closes
  and that exact child is stopped after every call. Late answers are not recovered or credited.
- HTTP body completion, 2xx status, inference response shape, timeout and process liveness
  are separate fields. HTTP errors/crashes become rows and remaining cells continue.
- Correctness requires a complete successful response and exact set equality of all queried
  object-position pairs. Whitespace/case normalize; duplicates, unknown classes, prose,
  wrong positions and missing/extra objects fail. Raw answers remain available.
- MemAvailable is KiB from /proc/meminfo. Baseline before launch, loaded before request,
  request minimum and lifecycle minimum are recorded. Lifecycle samples are every 250 ms
  through cleanup; short peaks may be missed. A 10s row covers only its observed request
  interval, not the memory a full unbounded inference would have needed.
- 'fits=YES' requires all four unbounded image calls to return successful inference responses
  with verified vision and a surviving child. 'UNPROVEN' covers failures/incomplete evidence;
  it is not an OOM diagnosis. No arbitrary RAM threshold is used.
- Summary unbounded latency is the median of successful HTTP calls only, with its sample
  count. Accuracy and 10s-success counts use denominator 4, so failures are not discarded.
- Missing, zero-byte or hash-mismatched models/projectors are recorded as skipped.
- Scratch port 8088 only. Existing llama-server processes cause an early refusal; no other
  process is stopped. Each owned child receives SIGTERM, then SIGKILL only if still alive
  after 10s, and is reaped before another launch. Failed cleanup stops the run.
- Fixed 60s cooldown after preflight hashing and between models; 2s between requests.
  This is a rest interval, not a claim of thermal equilibrium or a cleared filesystem cache.
- Battery must be unplugged/discharging and 50-80% at each request per DECISIONS #92.
  Power violations stop with partial JSON. Ctrl+C/SIGTERM also stop the child and save partial
  results; an interrupted pending row remains marked running rather than fabricated as success.
- Only standard-library modules; no robot, detector, camera, motor, or production launcher imports.

### Source evidence

- Installed tools/server/README.md lines 501-509 documents the multimodal prompt object.
- Installed tools/server/server-context.cpp lines 4618-4623 exposes vision capability and marker.
- Installed tools/mtmd/clip.cpp contains the warning that Qwen grounding needs at least 1024 image tokens.
- [Pinned Qwen ChatML/non-thinking template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/chat_template.jinja).

## Human run — Codex CLOSED

Stop other model servers first using the existing human-controlled procedure; use native Termux
with battery in the documented band. Then run this command (the output directory must be new):

```bash
cd ~/robot
python qwen_vision_bench.py --output ~/qwen-vision-bench-20260914
```

Results: results.json (atomic checkpoints), summary.txt, and one server log per attempted image.
The unbounded pass can wait indefinitely for a hung live server; Ctrl+C preserves partial results.

## Verification before review

- Native Python --check passed: b2351 version, all five model hashes, three projector hashes,
  all four image hashes, and labels.csv hash. No models loaded.
- qwen_vision_bench_test.py uses only fake local HTTP child processes: request shape, strict
  labels, 500 response while alive, partial-body timeout while alive, startup exit,
  text-only capability rejection, memory fields, child cleanup, missing/zero-byte skips,
  atomic JSON and alive-versus-success summary. No model sweep performed.

## Doc Keeper evidence

New external prototype assets: 0.8B and 4B F16 projectors downloaded and hash verified;
2B projector reused. All requested VL variants can now be attempted by the human.
Canonical STATUS/DECISIONS were not edited. No runtime performance or fit result is claimed.
