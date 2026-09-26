# Coder reports, verbatim

Extracted unchanged from Claude Code session transcript `861a0a6f-957a-4f4c-89f0-a59cee9d75e3.jsonl` (assistant text messages at the UTC timestamps below). Only this header and the separators were added.

---

## Message at 2026-09-24T21:10:37Z

I've finished verifying every candidate. Before downloading anything, I need one decision from you.

| Candidate | Source @ revision | Licence | Version / params | Download | Runtime | Works on aarch64 CPU here? |
|---|---|---|---|---|---|---|
| Laya | HF `convaiinnovations/laya` @ `55cf4c4ebb4e…` + PyPI `laya` | Apache-2.0 | **0.3.20, newer than 0.3.2** (18 releases since); 421.3 M params | ~0.85 GB (root `model.safetensors`, 842.6 MB) | torch + transformers | Yes, expected (0.3.2 already ran) |
| laya-micro | GitHub `osamabinIaggin/laya-micro` @ `686fc466…` (not on Hugging Face) | MIT | toolkit, no release; 322 M, then vocabulary-pruned | ~0.68 GB (Laya's `multilingual/` subfolder) + ~1.2 GB of build outputs | build steps need torch, onnx and onnxscript; the result runs on onnxruntime | Likely: onnxruntime has aarch64 wheels. **The model has to be built on the phone** (prune → export to ONNX → int8) |
| Decision-1.0-Kai-0.6B | HF `llm-semantic-router/…` @ `7185f514…` | Apache-2.0 (+Gemma tokenizer terms) | ~0.6 B | 2.17 GiB | torch with ROCm, transformers **4.57.6** | **No.** Code: `"This runtime requires a real ROCm CUDA device; no CPU fallback"` |
| Decision-1.0-Lex-0.6B | @ `ee8e74d9…` | Apache-2.0 | ~0.6 B | 2.17 GiB | same as Kai | **No** (same guard) |
| Decision-1.0-Eos-0.8B | @ `3c2d6326…` | Apache-2.0 | 0.8 B (Qwen3.5) | 1.43 GiB | torch with ROCm, transformers 5.17, FLA 0.5.2 | **No.** Code: `"A BF16-capable GPU is required; AMD ROCm uses device="cuda:0""` |
| Decision-1.0-Sol-2B | @ `0665a411…` | Apache-2.0 | 2 B (Qwen3.5) | 3.54 GiB | ROCm Docker image, Triton + FLA | **No.** Code: `"The bound profile requires a ROCm CUDA device"` |
| Decision-1.0-Nox-4B | @ `0bb83350…` | Apache-2.0 | 4 B (Qwen3.5) | 7.87 GiB | same as Sol | **No** (same guard) |
| system-one-open | GitHub `mithalouni/system-one-open` @ `77f1f7cc…` | MIT | trained weights **not published** (kept on the author's Modal cloud storage; "HF upload pending") | 0 (uses the installed GGUF, sha `cded614c…`) | its own code needs transformers and a GPU (via Modal). With the GGUF it can only run **zero-shot** through llama.cpp, reading the probabilities of the answer-letter tokens | Probably yes on build 1609: it already runs Gemma 4 E2B and supports `n_probs`/`logit_bias`. No rebuild needed |
| Von 1.1 | `wfzyx/von` | — | — | installed | — | revision `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd` |

**Storage:** 28 GB free, so the budget is 18 GB to keep 10 GB free.
- Everything together needs about 27 GB: 17.2 GiB of Decision weights, about 3.7 GB for Laya and laya-micro, and roughly 1.1 GB for each separate venv.
- That's over budget, so per your instruction I'm stopping to ask what to drop.

**My recommendation: drop all five Decision-1.0 models.**
- Every one of them refuses to run without an AMD ROCm GPU, checked in the loader code itself.
- Downloading them would only reproduce a failure we already know is coming.
- Without them, the total is about 5 GB.

---

## Message at 2026-09-24T21:23:53Z

Three candidates now load and make a decision offline on the phone: **Laya 0.3.20**, **laya-micro** (after building it on the phone), and **system-one-open** in zero-shot form on the installed GGUF. The five Decision-1.0 models weren't downloaded, as you chose, because every one of them refuses to run without an AMD ROCm GPU.

Every run used the same toy state with 4 options (`search_here`, `call_chiara`, `travel_next`, `finish`), cores 4–7, and 4 threads. Peak memory was measured from outside the process (`wait4`). **This was a smoke test, not a benchmark.**

| Candidate | Source | Revision | Size | Runtime | Loads | Load ms | First-call ms | Peak RSS |
|---|---|---|---|---|---|---:|---:|---:|
| **Laya 0.3.20** (latest; **newer than 0.3.2**, 18 releases since) | HF `convaiinnovations/laya` + PyPI `laya` | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | 842.6 MB weights | torch 2.14.0+cpu, transformers 5.17 | **Y** | 14451 | 1364 | 2810 MiB |
| **laya-micro** (int8 ONNX, built on phone) | GitHub `osamabinIaggin/laya-micro` (not on Hugging Face) + Laya `multilingual/` | code `686fc466…`; weights `55cf4c4e…` | 678 MB download → 164 MB int8 graph | build: torch + onnx 1.23 + onnxscript 0.7.2; run: onnxruntime 1.30.0 only, no torch | **Y** | 1406 | 534 | **600 MiB** |
| **system-one-open** (zero-shot) | GitHub `mithalouni/system-one-open` + installed GGUF | code `77f1f7cc…`; GGUF sha `cded614c9b24…a7` | 0 new (3.46 GB GGUF already installed) | llama.cpp **1609 (e1a1abb7)**, run from proot | **Y** | 4798 | 8268 | 3506 MiB |
| Decision-1.0-Kai-0.6B | HF `llm-semantic-router` | `7185f514…` | 2.17 GiB | ROCm torch, transformers 4.57.6 | not tested (dropped) | – | – | – |
| Decision-1.0-Lex-0.6B | same | `ee8e74d9…` | 2.17 GiB | same | not tested (dropped) | – | – | – |
| Decision-1.0-Eos-0.8B | same | `3c2d6326…` | 1.43 GiB | ROCm torch, FLA 0.5.2 | not tested (dropped) | – | – | – |
| Decision-1.0-Sol-2B | same | `0665a411…` | 3.54 GiB | ROCm Docker, Triton + FLA | not tested (dropped) | – | – | – |
| Decision-1.0-Nox-4B | same | `0bb83350…` | 7.87 GiB | same | not tested (dropped) | – | – | – |
| Von 1.1 (already installed) | `wfzyx/von` | `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd` | – | – | – | – | – | – |

Licences: the Laya checkpoint is Apache-2.0; the laya-micro and system-one-open code is MIT; the Decision models are Apache-2.0, plus Gemma tokenizer terms for Kai and Lex.

**What each model chose on the toy state:**
- **Laya:** `search_here`, with a near-flat distribution: search_here 0.296, travel_next 0.281, call_chiara 0.235, finish 0.189; confidence 0.0104.
- **laya-micro:** `search_here` 0.883, travel_next 0.081, call_chiara 0.021, finish 0.015; confidence 0.670.
- **system-one-open:** `search_here` 0.9994, call_chiara 0.0004, travel_next 0.0001, finish 0.0.

**system-one-open and llama.cpp**
- The repo itself needs transformers and a GPU on Modal (a cloud GPU service); it doesn't use llama.cpp at all.
- Its trained weights aren't published (the README says "HF upload pending"; the author has no models on Hugging Face). So on the phone it can only run as the **base** Gemma 4 E2B with the repo's own prompt builder, reading the probabilities of the answer letters A–D from one forward pass.
- **Build 1609 works for this, and nothing was rebuilt.** It already supports Gemma 4, returns token probabilities, and runs from proot.
- The four answer letters took essentially all the probability. One caveat: I didn't independently check whether this build reports probabilities before or after sampling.

**Hashes and pinning**
- Every download was pinned to a revision and hashed. Hashes are in `sha256.txt` in each candidate folder, plus `build_sha256.txt` for the laya-micro build outputs.
- Laya `model.safetensors` is `891102d3…` and multilingual is `9d628fd9…`; both match Hugging Face.
- Each candidate has its own venv and cache under `/termux-home/sel-candidates/`. The Von 1.1, Von 1.2 and Laya environments were not touched. Free space is now 23 GB.

**Failures and warnings, verbatim**
1. **laya-micro on Hugging Face:** `{'error': 'Invalid username or password.'}` (HTTP 401). It's a GitHub repo, not a Hugging Face one, and was found there.
2. **Decision-1.0, from the code (not run):**
   - Kai/Lex: `ValueError("This runtime requires a real ROCm CUDA device; no CPU fallback")`
   - Eos: `RuntimeError('A BF16-capable GPU is required; AMD ROCm uses device="cuda:0"')`
   - Sol/Nox: `RuntimeError('The bound profile requires a ROCm CUDA device')`
   - The model cards say the same: "CPU and MPS are not supported".
3. **Laya 0.3.20 at load:** `RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected entries as uncalibrated.`
4. **onnxruntime (harmless in proot):** `GPU device discovery failed: Error: std::error_code with category name: generic, value: 13, message: Permission denied, filesystem path: "/sys/class/drm", context: Iterating over DRM sysfs devices`
5. **laya-micro caveats (not crashes):**
   - `toy_tokenizer_parity_vs_stock: false`. The pruned vocabulary tokenizes our toy text differently from the stock tokenizer, so its outputs on our domain are out of distribution unless it's re-pruned on our own text.
   - Pruning produced a vocabulary of 7,779 tokens, while the README reports 15,188.
   - Quantization reported `max logit drift 2.0339, argmax matches`.
   - On this phone int8 was **slower** than fp32 (2290 vs 1920 ms per call).
6. **Nothing else failed.** All smoke runs and build steps exited 0.

As instructed, no reviewer was run for this step. Nothing in the repo changed: no commit, no push, `docs/` untouched.
