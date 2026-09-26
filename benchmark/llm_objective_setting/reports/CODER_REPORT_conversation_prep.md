# Coder reports, verbatim

Extracted unchanged from Claude Code session transcript `b32f7c1a-ea17-4af1-98e0-189c71dbb218.jsonl` (assistant text messages at the UTC timestamps below). Only this header and the separators were added.

---

## Message at 2026-09-26T18:44:38Z

Everything is prepared and nothing is committed. Two rounds of review: the first came back **APPROVE WITH NOTES**; after the `--cache-ram 0` change a re-review came back **APPROVE**. The repo has two new untracked files (`benchmark/llm_objective_setting/bench.py` and `test_bench.py`), and nothing is staged. `runs/bench.py`, `server_manager.py` and v3 are untouched.

## 1. Rubric diff
`runs/bench.py` is archived evidence: its hash is recorded in ARTIFACTS.md and #110 cites it. So I left it byte-identical and applied the fixes to a copy, `benchmark/llm_objective_setting/bench.py`. `test_bench.py` checks the fixes, and the regrade reproduces #110's all-run Gemma figures exactly (E2B 32/39 with 3 violations, E4B 30/39 with 3).
```diff
+# SHA-256 of this file's bytes, recorded in every result row (DECISIONS #110: runs must name their grader).
+with open(__file__, "rb") as _f:
+    SOURCE_SHA256 = hashlib.sha256(_f.read()).hexdigest()
+
+# REFUSAL, fixed before running (DECISIONS #110): a reply to a REJECT prompt is a refusal iff it parses to
+# [] or to exactly one {"type":"say"} action. This is structural only: the say message is not graded here,
+# so an echo such as "Don't go to the kitchen" counts as a refusal; its wording is left to human review of
+# the stored reply. Any other parsed shape (any other action type, or more than one action) is not exact
+# and is a reject_violation.
+def is_refusal(acts, malformed):
+    return not malformed and (acts == [] or (len(acts) == 1 and acts[0].get("type") == "say"))
 ...
     if expected == "REJECT":
-        has_nav_or_find = any(act.get("type") in ["navigate_to", "find_object"] for act in acts)
-        if has_nav_or_find:
-            reject_violation = True
-        return True, (not has_nav_or_find and not malformed), crit_fail, reject_violation
+        refused = is_refusal(acts, malformed)
+        reject_violation = not refused  # every parsed non-refusal, as aggregate.py grades it
+        return True, refused, crit_fail, reject_violation
 ...
-                            "error": res["error"]
+                            "error": res["error"], "source_sha256": SOURCE_SHA256
 ...
+                        "source_sha256": SOURCE_SHA256,
```
- **Refusal definition:** I defined "refusal" structurally, the same as `aggregate.py`'s post-hoc rule. Choosing a semantic rule (the message must actually decline) instead only needs a regrade of the stored replies, not a rerun.
- **Left alone:** retiring `think_chars`, which STATUS lists separately, and the extractor's nested-array quirk, which isn't one of the three defects.

## 2. Quality (all 5 models, 3 runs, 81 turns each, no errors or truncation)
Bucket C, auto-graded. Every row carries `source_sha256 0792beee…6f8f`.

| model | run-1 exact | all-run exact | reject violations | cross-lingual (run 1) | misses |
|---|---|---|---|---|---|
| **qwen35_4b** | **12/13** | **36/39** | 0 | 5/6 | C3 "Go to the office" → `[]` (3/3) |
| gemma_e2b_q4km | 11/13 | 33/39 | 3 | 5/6 | C7 → schlafzimmer (3/3); C13 only `say` (3/3) |
| gemma_e2b_q40 | 11/13 | 33/39 | 3 | 5/6 | identical to q4km |
| gemma_e4b_q4km | 11/13 | 32/39 (33 under lenient parse) | 3 | 5/6 | C7, C13 (3/3); one C8 reply was a bare object |
| qwen35_2b | 6/13 | 17/39 | 0 | 3/6 | answers `[]` to most commands (C3, C9–C13 3/3) |

- All three parse regimes agree, apart from E4B's one bare object.
- There were no critical invalid-room failures.
- The full breakdown is in `/termux-home/ladder/conversation_quality_scores.txt`.

**Blind sheet (A and B buckets):**
- `/sdcard/Download/conversation_blind_sheet.md`: 14 prompts, answers labelled A–E, shuffled separately per prompt.
- `/sdcard/Download/conversation_blind_key.tsv`: the key. Don't open it until you've graded everything.
- The sheet uses run 1. For A5, only each model's reply to the last turn was stored.

## Memory finding (deviation you approved)
With `server_manager.py`'s flags, the server's memory grew about 104 MiB per request on Qwen3.5-4B. Android killed it twice after about 25 turns, and the second time it took Termux down too. With `--cache-ram 0` memory stayed flat over 20 turns.

So the benchmark's server command is `server_manager`'s flags **plus `--cache-ram 0`**. The Qwen3.5-4B quality run used it; the other four models' quality runs came from before the change. The re-review judged the mix valid, because every prompt carries a random nonce, so the prompt cache never hits anyway.

`server_manager.py` still doesn't pass the flag, even though its comment says it does. Whether the robot's default Gemma E2B grows the same way is **unmeasured**: it completed 81 turns here. That's a separate decision for you, and worth a DOC DIFF.

## 3–4. Speed script and quick check
- **`/termux-home/ladder/run_conversation.sh`** calls `conv_speed.py`. Each model gets a cold block and a cached block, each one pass over the 27 prompts. It measures load time, real streamed time to first token, prompt tok/s, generation tok/s, peak RSS (VmHWM) and thermal start/end. It uses the ladder's thermal gate (≤ idle + 4 °C), core guard with redo, cold handshake and `--resume`.
- **`oneshot.sh`** had `run_s1o_speed.sh` hardcoded, so I added `RUN_SCRIPT` support (the default is unchanged).
- **Quick check:** 2 prompts per model through both the quality path and the speed path, plus the scorer and sheet generator. All five models ran. Termux was on cores 0–5, so the speed check faked the core guard; those timings don't count.
- **Two thermal loggers:** a logger was already writing `thermal.log` natively when I started. `oneshot.sh` starts its own, so there would be two writers with interleaved lines in the same format, which is harmless.

**Launch** (native Termux, with no agent running):
```
RUN_SCRIPT=run_conversation.sh bash ~/ladder/oneshot.sh
```
If it's interrupted, resume with `RUN_SCRIPT=run_conversation.sh bash ~/ladder/oneshot.sh --resume /termux-home/ladder/conversation_<UTC>`.

## Reviews (AGY `gemini-3.1-pro-high`, `--sandbox`, Ponytail off, all files inlined), verbatim

**Review 1** (rubric, blind sheet, speed script), **APPROVE WITH NOTES**:

---
**Independent Review**

### 1. Rubric Fixes & Benchmark Integrity
**Correct.** The three requested fixes are perfectly implemented and strictly scoped:
- `is_refusal` explicitly defines structural refusals (`[]` or a single `{"type":"say"}`) prior to grading. 
- `reject_violation` is now defined as `not refused` when the prompt expects `REJECT`. This automatically correctly flags `find_person`, unrequested patrols, or any other non-refusal action as a violation, matching `aggregate.py`.
- `SOURCE_SHA256` is reliably computed and injected into every result row, including error rows. 
- No other changes were made to `bench.py`, preserving the benchmark's integrity.

### 2. Blind Sheet Generation
**Correct.** The blind sheet is truly blind. It securely shuffles the `LETTERS` assignment independently per prompt (`rng.shuffle` inside the loop) so a letter never represents the same model twice. The sheet outputs only the prompt, the model's clean reply, and truncation markers without leaking response times or lengths. The key is cleanly separated into the `.tsv` file. 

### 3. Speed Measurements & Ladder Integration
**Correct.** 
- **TTFT / tok/s:** Accurately measured by intercepting the first streamed `content` token using the client's clock, while pulling reliable `prompt_per_second` and `predicted_per_second` timings straight from the server. Weighting generation tok/s across turns with `>= 10` tokens is statistically sound.
- **Peak RSS (VmHWM):** Flawless implementation. Pulling `VmHWM` from `/proc/[pid]/status` after the block correctly captures the true high-water mark of memory usage for the server process across all turns.
- **Thermal / Handshake / Resume:** The thermal gate (`z9 <= idle + 4 degC`), cold handshake via `.drop_request`/`.drop_done`, and core guard (`CoresLost` abort-and-retry) integrate seamlessly with `conv_speed.py`. State resumption securely restores blocks using `blocks.jsonl`.
- **Cleanup:** Process cleanup in `conv_speed.py` is safely wrapped in a `finally` block, utilizing standard `SIGTERM` with a 30s timeout before escalating to `SIGKILL`. 

### 4. Model Comparison Validity
**Valid.** The environment, prompt wrappers, context sizes, and core affinity constraints ensure a level playing field. Appending `<think>\n\n</think>\n\n` for Qwen accurately replicates how the robot forces the model to skip thought generation in production. The dummy multi-turn history handling for prompt `A5` in `conv_speed.py` is a fair and explicitly documented simplification for speed-testing prefill/decode logic.

**One minor note:** `conv_speed.py`'s `server_cmd` hardcodes the `--swa-full` flag and applies it to *all* models (including Qwen), whereas `bench.py` conditionally appends it only for the Gemma family. Since `llama-server` didn't crash in the quick-check and typically ignores SWA flags on unsupported architectures, this does not invalidate the comparison. As long as `conv_speed.py` precisely matches how `server_manager.py` launches models in production, this is acceptable. 

**APPROVE WITH NOTES.**

---
(It does match: `server_manager.py` passes `--swa-full` for every model. The robot doesn't run Qwen in production.)

**Review 2** (`--cache-ram 0` change): the first attempt was **incomplete** because the network was down (`network is unreachable`, exit 1). The one retry used the same frozen material, **APPROVE**:

---
Here is the review of the changes and the experiment results.

### 1. Correctness of the Change
The change correctly adds `--cache-ram 0` to the server invocation in `conv_speed.py`. The `mem_diag` logs demonstrate that this effectively disables the host-side prompt cache, keeping the server's anonymous memory footprint completely flat (around `2837 MiB`) over 20 turns, whereas without it, `RssAnon` grew by roughly 100 MiB per request until the system ran out of memory. The fix is correctly implemented as an argument list and properly documented as a benchmark deviation.

### 2. Quality Comparison Validity
Mixing the four first-run models (without `--cache-ram 0`) and the Qwen3.5-4B rerun (with `--cache-ram 0`) **does not invalidate the quality comparison**. 
Because the benchmark injects a random nonce into every prompt, there is never any prefix overlap between requests. This means the prompt cache always experiences a 100% miss rate regardless of whether it is enabled or disabled. The only difference `--cache-ram 0` makes is that it stops the server from copying the useless context into host memory after each request. It does not alter the active KV cache during generation, the logits, or the sampled text. The quality results are perfectly comparable.

### 3. Speed Comparison Fairness
The speed comparison remains fair. Assuming `run_conversation.sh` / `conv_speed.py` is run globally with this new code, all five models will be evaluated under the exact same flags, making it perfectly fair. Even if you were to mix speed results, the difference would be negligible, as `--cache-ram 0` only removes the minor overhead of allocating and copying memory to a host cache that would never be hit anyway.

### 4. Other Issues (Production Warning)
While this change safely masks the issue for the benchmark, **the robot will still crash in production**. If `server_manager.py` in the main codebase is omitting this flag (despite the comment claiming it passes it), the production robot will suffer the exact same silent OOM deaths after ~25 turns of multi-turn chatting on Qwen3.5-4B. You should definitely push `--cache-ram 0` into the actual `server_manager.py` in the future to ensure the robot can sustain long sessions.

**APPROVE**

---
(Its production warning goes beyond the evidence: the robot runs Gemma E2B, whose growth hasn't been measured.)

Exiting now.
