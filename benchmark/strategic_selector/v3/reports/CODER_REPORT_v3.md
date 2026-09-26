# Coder reports, verbatim

Extracted unchanged from Claude Code session transcript `861a0a6f-957a-4f4c-89f0-a59cee9d75e3.jsonl` (assistant text messages at the UTC timestamps below). Only this header and the separators were added.

---

## Message at 2026-09-25T00:13:31Z

I've built harness v3 and run the full development set once for all five candidates. The reviewer's verdict is **FAIL / INCOMPLETE**. Its "critical" finding is contradicted by the result files, and it says it couldn't find the result JSONs even though they were in the workspace, so as it stands this is an incomplete review, not a real pass or fail.

**Development results, 66 cases, default frame `filtered_text`** (preferred / acceptable counts; reversed means the same cases with the option order reversed):

| Candidate | Preferred | Acceptable | Preferred (reversed) | Acceptable (reversed) | Invalid | Order flips | Load ms | First call ms | Warm median ms | Warm P95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **s1o** (zero-shot Gemma 4 E2B) | **46** | **64** | 38 | 55 | 0 | 16/66 | 7717 | 29139 | 27174 | 30903 | 3666 |
| laya_en (0.3.20) | 34 | 54 | 25 | 36 | 0 | 19/66 | 28488 | 5266 | 3498 | 4700 | 2811 |
| von11 | 33 | 49 | 28 | 46 | 0 | **55/66** | 23329 | 7974 | 2806 | 5213 | 3363 |
| laya_multi (stock fp32) | 26 | 37 | 21 | 34 | 0 | 17/66 | 33386 | 1105 | 1083 | 1327 | 2585 |
| laya_micro (int8, 15,188 vocab) | 22 | 32 | 31 | 43 | 0 | 29/66 | 2028 | 1966 | 1306 | 1536 | **540** |

- Per-family results are in `/termux-home/v3-runs/dev-full.report.md`.
- **No candidate got any `possible_person` case right**, in canonical order. Only s1o and von11 ever got `all_rooms_called` and `localization_lost` right.
- **Timings are provisional**, as you said. The s1o times are dominated by llama.cpp 1609 evaluating prompts at about 14 tokens/s with prompt caching off.
- **laya_micro's tokenizer matched the stock tokenizer on 0/132 calls.** The pruned vocabulary splits common words, for example `▁Mission` becomes `▁Mis` + `sion`. So its inputs are out of distribution, which explains why it scores below stock laya_multi.

**Held-out set:** 66 cases, seed 7919, SHA-256 `96d41308eb0d2fa0c3bf30c1370eb93d1b4b7afbfd13cd433370cfd5cc3b2601`. It was never loaded, printed or scored. The runner refuses it without `--unseal-heldout`, and a test checks that.

### Item 7: laya-micro
- **Latency:** the quantizer times random inputs of 3 items × 259 tokens × 28 options (laya-micro's own 3-question example). The smoke test sent 1 × 171 × 4. On the same graphs, fp32 took 408 vs 1922 ms and int8 471 vs 2039 ms. int8 isn't faster than fp32 on this CPU at either size.
- **Vocabulary:** the README quickstart says `--seed-top 4096`, but laya-micro's own `docs/METHOD.md` gives `seed_top 4,096 -> 7,779` and `8,192 -> 15,188 (the shipped checkpoint)`. Rebuilding with 8192 gave exactly 15,188 with parity 2805/2805, and the harness uses that build.

### Item 8: s1o
- **Source:** build 1609's `populate_token_probs` uses `get_token_probabilities()`, which is a softmax over the raw `llama_get_logits_ith` logits, whenever `post_sampling_probs=false`. Only `true` reads the sampler chain.
- **Measured:** with `false`, the A–F probabilities are bit-identical (difference 0.000000) at T=0, at T=1.7 with top_k=2, and with a +5 logit bias on B. With `true` and a neutral sampler chain they match within 5e-6; with `true` and top_k=2 only two tokens come back.
- **Configuration:** the adapter sends `post_sampling_probs:false`, `n_probs:1000` and `cache_prompt:false`. It renormalises over the option letters, and raises an error if a letter falls outside the top 1000.

### Harness: `benchmark/strategic_selector/v3/` (untracked, not committed)
| File | SHA-256 |
|---|---|
| adapters.py | `ae7dd0a026771572458cdeeeb27f65891ffddb8e126b049994428834c0902632` |
| cases.py | `892a946dab0ff0973224b1fceb01400661f7a0f40f22892ace306d2dd4678781` |
| run.py | `caeb2a3c173a5dc77e404828814915e39bd41b7fa607a4388781c4cad11a9199` |
| measure.py | `e3fcbe3f93675935b7387d9dada15ff846928579841728a0083d9416f7cc2b83` |
| report.py | `1fc7ff87cc78bc66f4c1181aa046449d84d74aa40fca2fae8f604905b48d7ad6` |
| test_v3.py | `06c51a153bf5e898333014fdbe1b5868d45be3205f7a420d455251bae5af64f1` |
| s1o_probs_evidence.py | `1f176e9430d13f863d47c21002946fd38c2ffd56f254f029e851a2b820a58194` |
| CASES.json | `fddf3a31f59b3a00dbab16d6e25d37686c74d20171930da395feafd449c0fbe2` |
| dev.jsonl (66) | `a4e0f0bf318a6d4b068e03dd9325b44c917687e9d8334b1d9c872b3a700b8420` |
| heldout.jsonl (66, sealed) | `96d41308eb0d2fa0c3bf30c1370eb93d1b4b7afbfd13cd433370cfd5cc3b2601` |

- The SHAs were recorded before the run and verified unchanged afterwards, and `test_v3.py` passes.
- **v2 label defect:** `hall_hint` variations 3–5 label a room that was just searched as the preferred answer. v2's own 3 variations never hit this. v3 skips those 3 variations and keeps v2's labels for all the others.
- Run outputs are in `/termux-home/v3-runs/dev-full/`.

### Review (AGY, gemini-3.1-pro-high, Ponytail off, `--sandbox`)
Status SUCCESS, 229 s, no denied actions. stdout and stderr were kept separately (stderr was empty), and the frozen workspace was unchanged afterwards. Verbatim:

> An independent review of the benchmark harness v3 has been completed.
>
> **VERDICT: FAIL / INCOMPLETE**
> The harness contains a catastrophic logical flaw in its primary robustness metric, rendering the current leaderboard completely misleading. Additionally, the review is strictly `INCOMPLETE` because the raw `dev-full/*.json` files were not provided in the environment, preventing a full independent recomputation of the reverse splits and warm quantiles.
>
> Below are the detailed findings, categorized by severity.
>
> ### 1. CRITICAL: `order_flips` Metric is Inverted (Severity: Critical)
> **Evidence:** `v3/run.py` (lines 53-55), `v3/test_v3.py` (line 33)
> The metric designed to measure position-bias actually *rewards* it and penalizes semantic robustness.
> - In `run.py`, `order_flips` is calculated as `sum(normal[c] != reverse[c])`.
> - `r["choice"]` stores the selected option **key** (e.g., `"A"`, `"B"`), not the semantic action description.
> - When the order of options is reversed, a perfectly robust model that selects the same semantic action (e.g., "Search Room") will select a **different key**. Thus, `normal[c] != reverse[c]` evaluates to `True`, heavily penalizing the robust model with a high flip count.
> - Conversely, a completely broken, position-biased model that blindly always chooses `"A"` will have `normal[c] == reverse[c]` evaluate to `True`, yielding **0 order flips**.
> - **Proof:** The author codified this bug in `test_v3.py` by asserting that the `Mock` adapter has `0` order flips. The `Mock` adapter literally hashes the option keys (scoring `"A"` identically every time) and ignores the text. It is 100% position-biased, yet the test suite enforces that it receives a perfect `0` order flip score.
> - **Impact:** `von11` appears to be the most robust model in reality (tracking the semantic action to its new key in 55/66 cases), but the report penalizes it as the worst (55 flips). `s1o` and `laya` variants are heavily position-biased (blindly picking the same letter ~50/66 times), yet the report praises them for having low "flips" (16-29).
>
> ### 2. METHODOLOGICAL THREAT: Held-out Set is Not Sealed (Severity: High)
> **Evidence:** `v3/cases.py` (lines 20-22), `v3/test_v3.py` (lines 21-23)
> The held-out dataset is claimed to be "sealed," but it is generated deterministically from the open-source v2 logic using a hardcoded seed (`SPLITS = {"heldout": 7919}`). Any user, model, or training run with access to `cases.py` can trivially reconstruct the exact byte-for-byte held-out set. A true seal requires keeping the generator script, the generator logic, or the seed strictly private.
>
> ### 3. METHODOLOGICAL THREAT: Tokenizer Parity Fails 100% of the Time (Severity: High)
> **Evidence:** `v3/adapters.py` (lines 75-76), `v3/report.py` output
> The report notes that `laya_micro` achieved a tokenizer parity vs. stock of **0/132 calls**.
> - The coder rebuilt the pruned tokenizer and claimed it achieved 2805/2805 parity in isolated testing.
> - However, the parity check in `adapters.py` relies on `json.dumps(state, ensure_ascii=False)`. This likely introduces whitespace, escaping, or structural formatting differences that the native C++ runtime tokenizes differently than the stock HuggingFace tokenizer.
> - Either the pruning *did* catastrophically break the tokenizer for this specific prompt domain, or the inline check is flawed. In either case, the `laya_micro` accuracy results cannot be trusted to represent the true pruned model's theoretical capacity until this 100% failure rate is fixed.
>
> ### 4. DATA PRESENTATION: Misleading Per-Family Table (Severity: Medium)
> **Evidence:** `v3/report.py` (line 28)
> The header of the per-family table reads `Per family (n)` where `n=6` cases. However, the cells are populated as `preferred / acceptable`, rather than `preferred / n`.
> - For instance, if a model gets `1/3` (like `laya_multi` on `hall_hint`), it means it made 1 preferred choice and 3 acceptable choices. It implicitly means it completely failed the remaining 3 cases (0 acceptable). Visually, this formatting heavily implies a score of "1 out of 3 cases", making the table highly confusing and prone to misinterpretation by decision-makers.
>
> ### 5. MISSING ARTIFACTS: Recomputation Incomplete (Severity: Medium)
> **Evidence:** Missing `dev-full/*.json` files.
> Because the full JSON trace files were not provided in the environment or the payload, I cannot mathematically recompute the reverse metrics, the true order flips, or the warm `p50` quantiles to check for floating-point or sorting bugs. I can only verify the provided tables against each other.
>
> ---
>
> ### Recomputed Numbers (Normal Order Only)
> Using the provided `Per family` table, I manually re-summed the total `normal` preferred and acceptable counts. They **perfectly match** the overall summary table provided in the report, verifying that the aggregation math from family to overall totals is correct:
>
> *   **laya_en**: Preferred: 34, Acceptable: 54
> *   **laya_micro**: Preferred: 22, Acceptable: 32
> *   **laya_multi**: Preferred: 26, Acceptable: 37
> *   **s1o**: Preferred: 46, Acceptable: 64
> *   **von11**: Preferred: 33, Acceptable: 49
>
> ### Items 7 and 8 Verification
> *   **Item 7 (laya_micro):** Supported by evidence. `shape_timing.stdout.txt` proves that the latency gap (408 ms vs 1922 ms) is purely due to the batch and sequence shape difference (1x171x4 vs. 3x259x28). The vocab size discrepancy (7,779 vs 15,188) is properly explained by the shift from `--seed-top 4096` to `8192` in the final shipped checkpoint.
> *   **Item 8 (s1o):** Supported by evidence. `s1o_probs_evidence.stdout.txt` proves that when `post_sampling_probs=False`, the llama server ignores sampling parameters like Temperature and Top-K, returning identical probabilities across runs. The adapter correctly isolates the pre-sampling raw softmax over the vocabulary. (Note: `s1o` is being evaluated zero-shot on a base model, which poses a minor methodological threat regarding instruction adherence, but its high A-letter mass suggests it conforms well enough).

### My notes on the review (checked read-only, no code changed)
- **Finding 1 is contradicted by the data.** Choices are stored as meaning-level keys, not letters. For example, `s1o.json` row 0: canonical offered `['search_here', …, 'travel:toilet']` and chose `search_here`; reversed offered `['travel:toilet', …, 'search_here']` and also chose `search_here`. So a flip means the model chose a different action, which is what a flip should mean. s1o's letters are mapped back to option keys by position before scoring. The mock adapter scores the key names, so its choice doesn't depend on option order, and 0 flips is correct for it. The chosen positions are also spread out: s1o's canonical picks fall at positions 0/1/2/3/4 = 20/10/32/3/1, not "always A".
- **Finding 3:** the parity failure is real, and it isn't caused by my check. States are plain strings, not `json.dumps` output, and both tokenizers are the same library. The first difference is `▁Mission` → `▁Mis` + `sion`.
- **Finding 5 is wrong:** `dev-full/*.json` were in the workspace (about 470 KB each). The reviewer didn't read them, so it never recomputed the reversed-order numbers, flips or warm medians.
- **Findings 2 and 4 are fair.** The held-out set is sealed procedurally, not cryptographically: anyone with `cases.py` can regenerate it. The per-family cells are "preferred/acceptable", which reads ambiguously.

I've made no changes, and any change would need a new review. Nothing is committed or pushed, and `docs/` is untouched.

Decision for you: should I re-run the reviewer with the per-candidate summaries and a sample of rows included in the prompt, so it can actually recompute? And should I fix the report layout (finding 4), which would then need that new review anyway?

---

## Message at 2026-09-25T03:14:23Z

All four items are done, and the re-review passed. It confirms that the previous review's "inverted flips" finding was wrong.

**1. Per-family table fixed.** `report.py` now prints each cell as `P x/n · A y/n`. That was the only code change; its new SHA-256 is `f10d94b1193513d00fa5db7eb1c93aeef558e4b0265be57c66a108fe2ec4a009`. The results weren't re-run, `test_v3.py` passes, and the new report is `/termux-home/v3-runs/dev-full.report.v2.md`.

| Family | laya_en | laya_micro | laya_multi | s1o | von11 |
|---|---|---|---|---|---|
| new_room | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 5/6 · A 5/6 | P 6/6 · A 6/6 | P 0/6 · A 0/6 |
| room_finished | P 3/6 · A 6/6 | P 0/6 · A 0/6 | P 1/6 · A 1/6 | P 3/6 · A 6/6 | P 3/6 · A 6/6 |
| hall_hint | P 2/6 · A 6/6 | P 0/6 · A 0/6 | P 1/6 · A 3/6 | P 2/6 · A 6/6 | P 2/6 · A 6/6 |
| target_confirmed | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 |
| possible_person | P 0/6 · A 6/6 | P 0/6 · A 6/6 | P 0/6 · A 6/6 | P 0/6 · A 6/6 | P 0/6 · A 0/6 |
| all_rooms_first | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 2/6 · A 2/6 |
| all_rooms_called | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 4/6 · A 4/6 | P 6/6 · A 6/6 |
| localization_lost | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 6/6 · A 6/6 | P 5/6 · A 5/6 |
| route_blocked | P 6/6 · A 6/6 | P 4/6 · A 4/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 |
| repeat_search | P 1/6 · A 6/6 | P 0/6 · A 4/6 | P 1/6 · A 4/6 | P 1/6 · A 6/6 | P 1/6 · A 6/6 |
| heard_from_room | P 4/6 · A 6/6 | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 6/6 · A 6/6 | P 2/6 · A 6/6 |

**A correction of my own:** the old ambiguous table misled me too. I previously wrote that no candidate got any `possible_person` case right. In fact the four non-Von candidates are **acceptable on 6/6** (all chose `search_here`), none chose the preferred `call_chiara`, and von11 is 0/6.

**2. possible_person dump:** `/termux-home/v3-runs/possible_person_dump.md`. It covers 6 cases × 2 orders, with the state, the offered options in order, the labels, and each candidate's choice and full distribution. It was generated read-only from the run files.
- The four non-Von candidates chose `search_here` in both orders, usually with high confidence (s1o about 0.999, laya_multi about 0.99).
- von11 chose a travel option: `travel:bedroom` in canonical order and `travel:kitchen` reversed.

**3. Re-review (AGY, gemini-3.1-pro-high, Ponytail off, `--sandbox`)**
- Status SUCCESS, 77 s, no denied actions, stderr empty. The prompt was 113 KB with all evidence inline.
- The frozen workspace and the harness were verified unchanged afterwards.
- My own recount of flips from the result files matches the summaries and the reviewer: von11 55, laya_en 19, laya_multi 17, laya_micro 29, s1o 16.

Verbatim:

> # REVIEW VERDICT: PASS
>
> ## Numbered Findings
>
> **Finding 1: order_flips calculation and semantic keys (Correction of previous review)**
> - **Severity**: None
> - **Evidence**: I recomputed the `order_flips` directly from the `CHOICES_ALL_CASES` dictionaries for all candidates by counting cases where the normal choice differs from the reverse choice. The computed counts match the `SUMMARY` JSON precisely:
>   - `von11`: 55
>   - `laya_en`: 19
>   - `laya_multi`: 17
>   - `laya_micro`: 29
>   - `s1o`: 16
> - **Analysis**: The `choice` values in the logs are literal semantic keys (e.g., `"search_here"`, `"travel:living_room"`), not positional letters like "A" or "B". Because they are semantic keys, a difference between the normal and reverse choices means the model genuinely decided on a different real-world action when the prompt order changed. Finding 1 from the previous review was invalid. The developer is completely correct.
>
> **Finding 2: Row-level consistency**
> - **Severity**: None
> - **Evidence**: I cross-referenced the 10 raw `ROW` outputs per candidate with their respective labels and distributions. In every instance, the `argmax` accurately reflects the highest probability key in the `distribution` dictionary, the `choice` matches the `argmax`, and the `preferred_match` / `acceptable_match` flags correctly indicate whether the chosen semantic string is in the label's lists. No inconsistencies exist.
>
> **Finding 3: Per-family table accuracy**
> - **Severity**: None
> - **Evidence**: I validated the new per-family Markdown table output from `report.py` against the underlying `SUMMARY` JSON counts for `von11` and `laya_en`. The values format correctly. For instance, `laya_en`'s `new_room` JSON states `preferred: 6`, `acceptable: 6`, which perfectly maps to `P 6/6 · A 6/6` in the table. `von11` maps perfectly to `P 0/6 · A 0/6` for the same family.
>
> **Finding 4: possible_person label judgment**
> - **Severity**: Info
> - **Evidence**: The robot is in an unsearched room with a detected but unidentified person 60 cm directly ahead. The current label marks `call_chiara` as preferred and allows `search_here` as acceptable.
> - **Judgment**: The label is **right**.
>   - `call_chiara` is the most intelligent, context-aware action: it safely addresses the person immediately without moving, attempting an ID via voice.
>   - `search_here` represents rigid, script-following behavior (pattern matching "unsearched room") that triggers a methodical sweep. While suboptimal because it ignores the person right in front of the robot, it safely keeps the robot in the room.
>   - Leaving the room entirely (as `von11` did) is a total failure.
>   - The label correctly differentiates deep contextual understanding (preferred) from basic procedural compliance (acceptable). I would not change this label. Note that `s1o` and the `laya` models all defaulted to `search_here`, earning an acceptable score but missing the preferred score, which is a highly useful diagnostic signal.
>
> **Finding 5: Methodology and remaining threats**
> - **Severity**: Info
> - **Evidence**:
>   1. The code change in `report.py` strictly altered presentation (table string formatting). The underlying data extraction and scoring logic were untouched. Therefore, not re-running the models is safe and appropriate.
>   2. The held-out split remains procedurally sealed. As noted by the human, the threat vector here is the internal team overfitting/tuning on the test set, not a cryptographically secure outer threat. Proceeding with the dev set for development and keeping the held-out set sealed is standard and correct methodology. There are no remaining threats to using these dev results.

**My note on the review:** I disagree with "no remaining threats". These are still open:
- laya_micro's tokenizer matched stock on 0 of 132 calls;
- s1o is zero-shot base Gemma, not the trained system-one-open model;
- the timings are provisional;
- v2's `hall_hint` label defect (v3 skips variations 3–5);
- only one frame (`filtered_text`) has been run.

**4. Held-out:** recorded in `/termux-home/v3-runs/HELDOUT_SEAL_NOTE.md`. It stays procedurally sealed with no code change: 66 cases, SHA-256 `96d41308…`, never loaded, printed or scored.

Nothing is committed or pushed, and `docs/` is untouched.

---

## Message at 2026-09-25T06:27:52Z

The review ran: **PASS WITH CAVEATS**, status SUCCESS in 50 s, no denied actions, stderr empty. The frozen workspace and the harness were verified unchanged afterwards.

## Finalists: speed and accuracy on the 24 judgment cases (provisional timings)

**Gate: laya_en passes and von11 passes, both provisionally. s1o is out: its best warm median is about 24.9 s, far over 3000 ms.**

The biggest caveat first. **Android moved Termux out of the full-CPU cpuset (0–7) into `/foreground` (0–5).** When that happens, `taskset -c 4-7` silently runs on cores 4–5 only, and cores 6–7 can't be used at all. That's why two von11 settings (6–7 at 3 and 4 threads) failed to launch. Earlier runs recorded no cpuset evidence, so the effective cores behind every timing below are uncertain. `measure.py` now records the allowed CPUs at the start and end of each run, and refuses to start if the requested cores aren't all available. For the clean run, keep Termux as the top app, screen on.

### laya_en
| Variant | Acceptable (rev) | Preferred (rev) | Flips | Warm median ms | P95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| torch, cores 4–7, 4 threads (dev-full) | 24 (7) | 10 (2) | 18/24 | 3647.0 | 5012.8 | 2811 |
| **ONNX fp32, 4–7, 4 threads** | **24 (7)** | **10 (2)** | 18/24 | **2092.3** | **2721.7** | **1646** |
| ONNX, 4–7, 3 threads | 24 (7) | 10 (2) | 18/24 | 3090.8 | 3265.1 | 1645 |
| ONNX, 4–7, 2 threads | 24 (7) | 10 (2) | 18/24 | 3246.8 | 3895.0 | 1645 |
| ONNX, 6–7, 2 threads | 24 (7) | 10 (2) | 18/24 | 2979.2 | 3765.5 | 1645 |
| ONNX, 6–7, 3 threads | 24 (7) | 10 (2) | 18/24 | 3781.0 | 4516.2 | 1642 |
| ONNX, 6–7, 4 threads | 24 (7) | 10 (2) | 18/24 | 4303.1 | 5591.8 | 1639 |

- **Parity gate passed:** argmax identical on **48/48** decisions (24 cases × 2 orders), max probability difference **0.0001**, which is torch Laya's 4-decimal rounding. The export's logit drift was 0.00001.
- **One adaptation was needed:** laya-micro's `runtime.py` reads `mask_token_id` from the encoder config, and the English config doesn't have it. I added it through an overlay checkpoint: symlinks to the real files, plus `mask_token_id` 50284, which is the tokenizer's own `[MASK]` and the id torch Laya uses.
- **Order sensitivity:** laya_en is acceptable on **24/24 in canonical order but only 7/24 reversed**.

### s1o (out)
| Variant | Acceptable (rev) | Preferred (rev) | Flips | Warm median ms | P95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| original prompt order (dev-full) | 24 (20) | 12 (9) | 11/24 | 28296.9 | 34173.2 | 3666 |
| instruction first, no cache | 24 (24) | 12 (12) | 11/24 | 27905.9 | 28916.5 | 3583 |
| instruction first, prefix cache | 24 (24) | 12 (12) | 11/24 | **24880.3** | 25917.8 | 3648 |

- **4a:** prompts are 323–340 tokens (median 329). Prompt evaluation runs at a median of 11.7 tokens/s.
- **4b:** llama-bench on the 329-token prompt:

  | Cores | Threads | ubatch | tokens/s |
  |---|---:|---:|---:|
  | 4–7 | 4 | 512 | **12.27** (best, the current setting) |
  | 4–7 | 4 | 128 | 11.13 |
  | 0–7 | 8 | 512 | 11.19 |
  | 6–7 | 2 | 512 | 8.71 |
  | 4–7 | 2 | 512 | 7.77 |

- **4c:** the instruction prefix is 41 tokens, and the server log confirms each scored call evaluated 288 of its 329 tokens. **The probabilities are not unchanged.** The top choice matched on 48/48 calls, but probabilities differ by up to **0.179**, and by more than 0.01 on 10 of 48 calls.
  - Diagnostic on the worst case: uncached runs are bit-identical to each other (0.567 for the top option), and so are cached runs from the same starting state (0.622). But the harness's cached run gave 0.746 for the same prompt, so cached results depend on what the server processed before.
- **4d:** llama.cpp was not rebuilt.

### von11
| Variant | Acceptable (rev) | Preferred (rev) | Flips | Warm median ms | P95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| 5a: plain (dev-full, no rerun) | 24 (24) | 8 (6) | 24/24 | 2921.3 | 5267.8 | 3363 |
| **plain, 4–7, 4 threads** | **24 (24)** | **8 (6)** | 24/24 | **2753.3** | **2945.3** | 3360 |
| plain, 4–7, 3 threads | 24 (24) | 8 (6) | 24/24 | 3157.0 | 3331.1 | 3078 |
| plain, 4–7, 2 threads | 24 (24) | 8 (6) | 24/24 | 3651.3 | 4472.2 | 3316 |
| plain, 6–7, 2 threads | 24 (24) | 8 (6) | 24/24 | 4029.5 | 5056.4 | 3063 |
| plain, 6–7, 3 and 4 threads | — | — | — | not measured (cpuset) | — | — |
| 5b: permutation averaging (5 rotations, batched) | 24 (24) | 7 (7) | 24/24 | 13771.7 | 28048.8 | 3371 |
| 5c: von-1.0 NLI backend | 24 (24) | 6 (6) | **0/24** | 8812.9 | 13653.7 | 1845 |

- **5b, batching:** von-sdk 1.1.1 has no public batch API. But the underlying model's `forward` accepts a padded batch, so each decision's rotations ran as **one forward pass**. That matched sequential `evaluate_choice` within 5.0e-5 on 8 decisions.
- **5b, flips:** permutation averaging still flips on 24/24. Cyclic rotations of the reversed list are different permutations from those of the canonical list, so averaging them doesn't make the result reversal-invariant.
- **5c:** yes, a second backend exists. BertaBackend `von-1.0` is an NLI cross-encoder loaded from the cached `d8bb5e07` snapshot. It's order-invariant (0 flips), but slower and less often preferred.
- **5d:** the best Von variant is **plain von11** (all three are 24/24 acceptable; plain has the most preferred and is fastest). It gets 2753 ms at 4 threads on cores 4–7.

### repeat_search: dump and judgment
The dump is at `/termux-home/v3-runs/repeat_search_dump.md`: 6 cases × 2 orders, with state, options, labels, and each finalist's choice and full distribution.

My judgment: **keep the label, but it grades a rule the instruction never states.**
- "Preferred = the nearest unsearched room" is a sensible efficiency tie-break, and the distances are in the state. But the instruction only asks for "the most useful next action" and says nothing about distance.
- The acceptable set (any unsearched room, with call and ask excluded) is what the instruction actually supports. All three finalists are 6/6 acceptable in canonical order.
- In canonical order the finalists mostly choose the **first travel option listed**, not the nearest one. So the preferred misses reflect position bias more than a labelling error.
- If preferred matters for the decision, the instruction should state the distance priority.

### Harness changes (untracked, not committed)
The diff against the pre-task harness is `/termux-home/v3-runs/fin/harness_finalists.diff`, 458 lines:
- **`run.py`:** a `--families` option, defaulting to the 4 judgment families.
- **`report.py`:** recomputes everything from the logged rows for the chosen families.
- **`measure.py`:** core set and threads from `V3_CORES`/`V3_THREADS`, new venv mappings, and the cpuset guard and recording.
- **`adapters.py`:** threads from `V3_THREADS`, plus new `LayaEnOnnx`, `Von11Perm`, `Von10Nli` and `S1OInstrFirst` adapters.
- **`test_v3.py`:** now also checks the default family filter. All tests pass.

| File | SHA-256 |
|---|---|
| adapters.py | `23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc` |
| measure.py | `7040be22ec0bf2ef8dadb5b46a52e10402ab84d1ffd38eab0020430519705242` |
| report.py | `46c34d87d228ac6418e5c457c703494626b49ce2404083fcc74811a0731092c2` |
| run.py | `a491c3646ad1f581d872bef548155b39172db880957f16789aa743d9f6374c48` |
| test_v3.py | `696d1f6772df9607453e79999a7be0285f21f837cbdc1351d371426906d051eb` |
| cases.py / s1o_probs_evidence.py | unchanged |
| heldout.jsonl | `96d41308…` (still sealed, never loaded) |

Other artifacts:
- laya_en ONNX: `laya_en.onnx` `bb76cb05…`, `.onnx.data` `48774636…`
- overlay encoder config: `40076416…`

### Review (AGY, gemini-3.1-pro-high, Ponytail off, `--sandbox`), verbatim

> **VERDICT: PASS WITH CAVEATS**
>
> **1. Harness Diff Evaluation (Severity: None - Correct implementation)**
> *   **`--families` filter**: Correctly implemented in `run.py` and `report.py`. The `report.py` script properly reconstructs the aggregate statistics from the underlying row data for only the requested families. It correctly identifies and excludes the first call of the entire run for warm statistics, even when rows are filtered.
> *   **Adapters**: `LayaEnOnnx`, `Von11Perm`, `Von10Nli`, and `S1OInstrFirst` correctly implement the required mechanics: proper mask configuration for the ONNX overlay, correct cyclic permutation mapping/averaging and batched forward passes for Von11Perm, proper model instantiation for the Von NLI backend, and correct prompt prefixing/caching for S1O.
> *   **`measure.py` cpuset guard**: Soundly implemented. It leverages `/proc/self/status` to read `Cpus_allowed_list` and will definitively abort if the requested core set is truncated, protecting the integrity of the upcoming clean runs against Android's cgroup shifting.
>
> **2. laya_en Parity Gate (Severity: None - Passed)**
> *   **Evidence**: The argmax perfectly matched 48/48 decisions across both orderings. A maximum probability difference of 0.000100 is completely valid and expected for FP32 ONNX evaluation versus Torch, falling well within acceptable floating-point mathematical drift. Parity is confirmed.
>
> **3. s1o Speed & Caching (Severity: Critical - Model Eliminated)**
> *   **Evidence**: The coder's conclusion is fully supported. The scratch diagnostic definitively proves that prefix caching in `llama.cpp` build 1609 is not stateless/invariant; identical text prompts yield significantly different output probabilities depending on whether the prefix was cached from a previous slot state.
> *   **Gate Status**: Even with caching, the fastest median response (`s1o_if_prefix`) is 24880.3 ms, which is ~24.9 seconds. This vastly exceeds the 3000 ms requirement. **s1o is definitively out.**
>
> **4. Von Custom Tests (Severity: Low - Informational)**
> *   **Table Verification**: The numbers in 5a precisely match the `dev-full` table for `von11` (Canonical: 24 Acceptable, 8 Preferred. Reversed: 24 Acceptable, 6 Preferred. 24/24 Flips).
> *   **Permutation Averaging**: The implementation is mathematically faithful to the spec (cyclic rotations generated, mapped back to keys, distributions averaged, and choice derived via argmax). The coder's note regarding reversal invariance is mathematically correct: cyclic rotations of a reversed list are *disjoint* permutations from the cyclic rotations of a canonical list, which is why 24/24 flips still occur under cyclic averaging.
> *   **Second Backend**: `Von10Nli` utilizing the Berta cross-encoder is a legitimate alternative backend and executed properly, though it is slower and less accurate.
> *   **Winner Rule & Gate**: Under the winner rule (Acceptable > Preferred > Speed), the standard **`von11` (c47_t4)** is the best Von variant. It achieves 24/24 acceptable in both canonical and reversed orders, and achieves a warm median of 2753.3 ms, provisionally passing the 3000 ms gate.
>
> **5. repeat_search Label Judgment (Severity: None - Sound reasoning)**
> *   **Evidence**: The coder correctly identified that the `preferred` label in this family strictly correlates with Euclidean distance to the target room, which is present in the state text but completely omitted from the prompt's instruction text.
> *   **Judgment Soundness**: The coder's logic is sound. It is unreasonable to penalize a model's "acceptable" correctness for failing to guess an unstated distance-optimization objective. Treating the physically nearest room as a "preferred" tie-breaker while retaining any valid unsearched room as "acceptable" is the exact intended use case for a dual-tier label system. The labels should be kept as-is.
>
> **6. Gate Status per Finalist & Threats**
> *   **s1o**: **FAIL**. Eliminated on speed.
> *   **laya_en**: **PASS WITH CAVEATS**. Provisionally passes speed (2092.3 ms on ONNX c47/t4).
>     *   **Threat - Severe Order Bias**: While it hits 24/24 acceptable canonically, this accuracy catastrophically collapses to **7/24 acceptable** when options are reversed. Its canonical success is highly reliant on positional bias (selecting the first option) rather than semantic understanding.
> *   **von11**: **PASS**. Provisionally passes speed (2753.3 ms on c47/t4).
>     *   **Advantage**: It is highly robust to option ordering. While it flips choices 24/24 times on reverse, it reliably flips to *another acceptable choice*, maintaining a perfect 24/24 acceptable rate on reversed lists.
> *   **Threat - The Cpuset Caveat**: The discovery that Android silently moved Termux from the `top-app` cpuset (cores 0-7) to `foreground` (cores 0-5) means the provisional timings for cores 4-7 likely ran under-resourced on fewer, weaker cores. Thanks to the new `measure.py` guard, the final clean runs will be forced onto the correct X1/A78 cores. Consequently, clean run timings for `von11` and `laya_en` on 4-7 are likely to be notably *faster* than these provisional timings, solidifying `von11`'s pass status against the 3000 ms gate.

**My notes on the review:**
- Its claim that clean-run timings will be "notably faster" is a guess. I don't know which runs actually lost cores 6–7, so the clean run has to show it.
- "Euclidean distance" isn't quite right; the labels use the route distances in the state. It doesn't change the conclusion.
- One point for the decision: laya_en is ahead on preferred canonically (10 vs 8), but it depends on option order (7/24 acceptable reversed). von11 is acceptable 24/24 in both orders, but its choice among the acceptable rooms changes with order in every case.

Nothing is committed or pushed, `docs/` is untouched, and the held-out set is still sealed.
