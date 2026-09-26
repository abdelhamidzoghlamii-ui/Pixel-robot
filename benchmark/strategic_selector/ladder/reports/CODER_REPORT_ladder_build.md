# Coder reports, verbatim

Extracted unchanged from Claude Code session transcript `cf7a68f1-edbb-4162-b820-2cd2b8c20018.jsonl` (assistant text messages at the UTC timestamps below). Only this header and the separators were added.

---

## Message at 2026-09-25T20:48:29Z

The ladder runner is built and was tested on 3 toy cases with laya_en and von11. **The reviewer's verdict is Fail**, on two thread-scaling bugs. I haven't fixed anything: edits were frozen for the review, and any change needs a new review. All four frozen files still match their SHA-256 hashes after the review. The real ladder has not been run.

**Two things you should know first:**
- **No cold load via root.** The phone isn't rooted: `su` inside Debian is proot's fake root, and `drop_caches` gives "Permission denied". The report prints that error. As a stand-in, the runner evicts only the model's weight files from the page cache and labels the result "weights-cold". Python libraries stay cached, so this understates a true cold start.
- **Only two models were run end to end.** The tokenise/forward timing paths for s1o, laya_multi, laya_micro, von12 and von10 NLI have not been exercised yet. The worker refuses to report timings if either timer never fired during warm-up, so a missed path fails loudly rather than giving silent zeros.

**Files** (all in `/termux-home/ladder/`)
- `ladder.py` — the runner: cases, blocks, cpuset guard, stats, winners, hints, report
- `ladder_worker.py` — per-venv worker; reuses the v3 adapters and the jevlike worker's token, memory and output code
- `test_ladder.py` — checks for the fit, P95, case validation and hint rules; passes
- `toy_cases.jsonl` — 3 cases I wrote (levels 1, 3 and 5; phrasings direct, indirect, narrative)
- `toy_run/` — `report.txt`, `decisions.jsonl` (every decision), `results.json`, `logs/`, plus the run's stdout and stderr
- `toy_run_1/` and `toy_run_2/` — two earlier toy runs, kept because they led to the two noise gates below

The review copy is in `/termux-home/ladder-review-20260925T204237Z/`: the request, the frozen files and hashes, `review.stdout.json` and `review.stderr.txt` (empty).

**How to run it:** `python3 /termux-home/ladder/ladder.py CASES.jsonl --out DIR [--models laya_en,von11] [--levels 1,2] [--threads 2,3,4]`

Your cases file needs one JSON object per line with `id`, `level` (1–5), `phrasing`, `situation`, `options` (a list, or a key→description map) and `answer`. `acceptable` and `instruction` are optional; without an instruction, the default is "Choose the single best next action for the robot."

**What I changed before the review:** the first toy run printed "run at 2 threads". It was based on a 4-thread decision that took 1952 ms in one block and 554 ms in another, with identical code. I added two gates:
- The thread hint is withheld when the same decision, repeated at the same thread count, differs by more than 1.3×.
- Thread-count differences smaller than max(15%, that measured noise) are reported as "no measurable difference".

**Toy report** (final run, exit 0, stderr empty, no leftover processes)
```
LADDER REPORT  2026-09-25T20:39:59+00:00
cases /data/data/com.termux/files/home/ladder/toy_cases.jsonl sha256 2ebe1f9fd4eac5d9  n=3  levels [1, 3, 5]
models laya_en, von11  threads [2, 3, 4]  cores 4-7  device RAM 7468 MiB
timing: perf_counter in the worker; decision = wall time of adapter.decide; post = total - tokenise - forward

== laya_en (ONNX) [CANDIDATE] ==
accuracy        correct 4/6 (67%)  acceptable 4/6 (67%)  (both orders)
  by level      L1 2/2 (100%)  L3 2/2 (100%)  L5 0/2 (0%)
  by phrasing   direct 2/2 (100%)  indirect 2/2 (100%)  narrative 0/2 (0%)
order flips     1/3  ['toy_3']
decision ms     median 546  P95 1044  (main block, 4 threads, n=6)
  split median  tokenise 0.3  forward 545.3  post 0.3   [tokenise=runtime.build_sequence, forward=onnxruntime run]
tokens/s        156   tokens by level L1 49  L3 85  L5 168   limit 512, hit 0x
latency fit     ms = 32.4 + 6.011 x tokens   R2 1.00  (n=6, tokens 49-168)
threads         2: 585 ms  3: 713 ms  4: 554 ms  (written order, fresh cached worker, n=3 each)   torch threads seen {'main': None, 't2': None, 't3': None, 't4': None}
                repeat check: worst toy_1 325 ms (main) vs 343 ms (t4) = 1.06x
                choice differs from main block in 0 decisions, max prob diff 0.0000
load (s)        phase         cold  cached
                file_read     1.27    0.39
                import        0.45    0.27
                init          2.26    2.19
                warmup        0.23    0.21
                total         4.21    3.06
                spawn_to_ready    4.54    3.39
                weight files 1610 MiB; cold = weights-cold (posix_fadvise DONTNEED on the weight files; libraries stay cached)
                drop_caches not possible: bash: line 1: /proc/sys/vm/drop_caches: Permission denied
RAM             peak 1538 MiB (VmHWM, worker)   largest single process 1539.0 MiB (wait4)
HINTS
  - per-token cost is 97% of predicted latency at level 5 (fit 32 ms + 6.01 ms/token x 168 tokens at level 5, R2 1.00) -> shorter input will help
  - model forward is 100% of decision time (median tokenise 0.3 / forward 545.3 / post 0.3 ms; shares 0%/100%/0%) -> only fewer tokens or a faster model will help; glue code is not the bottleneck
  - 2 vs 4 threads differ by 5%, within the 15% noise floor (2 thr 585 ms, 3 thr 713 ms, 4 thr 554 ms (median of 3 each)) -> 2 threads frees cores at no measurable cost
  - weights-cold load is 1.2 s slower than cached (4.2 vs 3.1 s; file read 1.3 vs 0.4 s) -> keep the worker resident or the weights in page cache
  - init is 71% of cached load (2.2 of 3.1 s) -> a faster-loading format or a resident worker will help
  - choice changed under reversed order in 1/3 cases (33%) -> order bias; average over orders or fix the order
  - accuracy falls from 100% at level 1 (2/2) to 0% at level 5 (0/2) -> the model breaks down at the harder levels
  - sensitive to phrasing: 'direct' 2/2 vs 'narrative' 0/2 correct (100% gap) -> normalise phrasing before the model

== von11 [CANDIDATE] ==
accuracy        correct 4/6 (67%)  acceptable 5/6 (83%)  (both orders)
  by level      L1 2/2 (100%)  L3 1/2 (50%)  L5 1/2 (50%)
  by phrasing   direct 2/2 (100%)  indirect 1/2 (50%)  narrative 1/2 (50%)
order flips     2/3  ['toy_2', 'toy_3']
decision ms     median 804  P95 1329  (main block, 4 threads, n=6)
  split median  tokenise 2.0  forward 802.3  post 0.6   [tokenise=OptionMarkerModel.tokenizer calls, forward=OptionMarkerModel forward hooks]
tokens/s        102   tokens by level L1 46  L3 82  L5 165   limit 8192, hit 0x
latency fit     ms = 338.4 + 5.814 x tokens   R2 0.98  (n=6, tokens 46-165)
threads         2: 979 ms  3: 938 ms  4: 851 ms  (written order, fresh cached worker, n=3 each)   torch threads seen {'main': 4, 't2': 2, 't3': 3, 't4': 4}
                repeat check: worst toy_3 1244 ms (main) vs 1348 ms (t4) = 1.08x
                choice differs from main block in 0 decisions, max prob diff 0.0000
load (s)        phase         cold  cached
                file_read     2.49    1.86
                import       19.51   15.83
                init          4.79    2.83
                warmup        0.68    0.49
                total        27.47   21.01
                spawn_to_ready   27.73   21.23
                weight files 3021 MiB; cold = weights-cold (posix_fadvise DONTNEED on the weight files; libraries stay cached)
                drop_caches not possible: bash: line 1: /proc/sys/vm/drop_caches: Permission denied
RAM             peak 3441 MiB (VmHWM, worker)   largest single process 3441.7 MiB (wait4)
HINTS
  - per-token cost is 74% of predicted latency at level 5 (fit 338 ms + 5.81 ms/token x 165 tokens at level 5, R2 0.98) -> shorter input will help
  - model forward is 100% of decision time (median tokenise 2.0 / forward 802.3 / post 0.6 ms; shares 0%/100%/0%) -> only fewer tokens or a faster model will help; glue code is not the bottleneck
  - 2 vs 4 threads differ by 13%, within the 15% noise floor (2 thr 979 ms, 3 thr 938 ms, 4 thr 851 ms (median of 3 each)) -> 2 threads frees cores at no measurable cost
  - weights-cold load is 6.5 s slower than cached (27.5 vs 21.0 s; file read 2.5 vs 1.9 s) -> keep the worker resident or the weights in page cache
  - import is 75% of cached load (15.8 of 21.0 s) -> library import dominates; a resident worker avoids it
  - choice changed under reversed order in 2/3 cases (67%) -> order bias; average over orders or fix the order
  - accuracy falls from 100% at level 1 (2/2) to 50% at level 5 (1/2) -> the model breaks down at the harder levels
  - sensitive to phrasing: 'direct' 2/2 vs 'indirect' 1/2 correct (50% gap) -> normalise phrasing before the model
  - peak RAM 3441 MiB is 46% of device RAM (7468 MiB) -> only one such model can stay resident

== CATEGORY WINNERS ==
accuracy         von11        von11 67%/83%acc  laya_en 67%/67%acc
order stability  laya_en      laya_en 33% flips  von11 67% flips
decision speed   laya_en      laya_en 546 ms  von11 804 ms
cached load      laya_en      laya_en 3.1 s  von11 21.0 s
cold load        laya_en      laya_en 4.2 s  von11 27.5 s
RAM              laya_en      laya_en 1538 MiB  von11 3441 MiB
(ties keep the listed model order; accuracy ties break on acceptable)
```

**Review, verbatim** (AGY 1.2.11, `gemini-3.1-pro-high`, `--sandbox`, no `--dangerously-skip-permissions`; status SUCCESS; stderr empty)

---

Here is the independent correctness review of the benchmark runner.

### Findings

**1. MAJOR: Thread scaling hint ignores intermediate thread counts**
* **Location**: `ladder.py:304-316`
* **What is wrong**: The logic strictly extracts the minimum and maximum tested thread counts (`lo, hi = min(sc), max(sc)`) and calculates the speedup (`gain`) between those two endpoints, completely ignoring any intermediate thread counts. 
* **Concrete failure scenario**: If running at 2 threads takes 500 ms, 3 threads takes 200 ms, and 4 threads takes 480 ms, the code compares only 2 and 4. It will conclude that the 4% difference is within the noise floor and output "2 threads frees cores at no measurable cost", hiding the fact that 3 threads is 60% faster and is the actual optimal configuration.
* **Suggested fix**: Iterate over all values in `sc` to find the thread count with the lowest `median_ms`. Compare that optimal thread count against the lowest thread count to advise whether the speedup justifies the extra cores.

**2. MAJOR: Falsely blaming thread counts for baseline non-determinism**
* **Location**: `ladder.py:228-231` and `ladder.py:318`
* **What is wrong**: `thread_choice_diffs` counts differences between `main_written` and *all* scaling blocks (which includes `t4`). The `main` block and the `t4` block execute the exact same workload at the exact same thread count (e.g., 4 threads).
* **Concrete failure scenario**: If the model has standard run-to-run non-determinism (due to float sampling, RNG, or unseeded softmax), `t4` might produce a different choice or probability than `main`. The script will output "choice changed with thread count... results depend on thread count; fix it", which will mislead the user into debugging thread scaling for a generic determinism issue.
* **Suggested fix**: Exclude the block that matches `main`'s thread count from the `thread_choice_diffs` loop (e.g., `for b in blocks if b["block"] != "main" and b["threads"] != main["threads"]`).

**3. MINOR: Tokens definition skews the latency-vs-length fit for KV-cached models**
* **Location**: `ladder.py:248`, `ladder.py:201` and `ladder_worker.py:164`
* **What is wrong**: The `tokens` counter is built by summing the length of all encoded option sequences (`sum(map(len, seqs))`). For most adapter setups, each option sequence includes the prompt (so a prompt of length P and options of length O results in `(P+O1) + (P+O2)`). If the model uses a KV cache and processes the prompt only once, this double counts the prompt tokens in the x-axis of the fit.
* **Concrete failure scenario**: A model reads a 500-token prompt once and evaluates four 10-token options. Total tokens processed is 540, but the instrumentation reports `(500+10)*4 = 2040` tokens. This causes the computed `per_token_ms` slope to be artificially diluted (~4x smaller than reality), which could trigger the "shortening input saves little" hint erroneously.
* **Suggested fix**: Clarify if the fit should track total *processed* tokens or *sequence length*. Consider using `longest` (max sequence length) as the x-axis for the latency fit, or deduplicate the prompt length if `jw.encoded` exposes it.

**4. NIT: Potential ZeroDivisionError on disturbed timings check**
* **Location**: `ladder.py:235`
* **What is wrong**: `min(x[1], x[2])` is used as the denominator when calculating the ratio for `repeat_worst`.
* **Concrete failure scenario**: On a fast stub/dummy model, or if the OS `perf_counter` clock resolution is unusually coarse, a decision could evaluate in exactly `0.0` ms. The script will immediately crash with a `ZeroDivisionError`.
* **Suggested fix**: Add a small epsilon to the denominator or use `max(1e-5, min(x[1], x[2]))`.

---

### Per-Hint Verdict Table

| Hint Rule | Follows from evidence? | Why |
| :--- | :---: | :--- |
| `no latency-vs-length hint` | **Yes** | Explicitly checks for lack of token spread before fitting. |
| `latency is poorly explained` | **Yes** | R2 < 0.5 correctly spots variance unconnected to input length. |
| `no measurable per-token cost` | **Yes** | A negative or zero slope mathematically proves no penalty for length. |
| `shorter input will help` | **Yes** | Share >= 50% cleanly isolates length as the dominant latency driver. |
| `shortening input saves little` | **Yes** | Share < 50% cleanly isolates fixed costs as the dominant latency driver. |
| `tokenisation is X% ... pre-tokenise` | **Yes** | >= 20% is a sensible threshold to warn about abnormal tokenizer overhead. |
| `post-processing is X% ... glue code` | **Yes** | >= 20% is a sensible threshold for non-inference overhead. |
| `model forward is X% ...` | **Yes** | >= 80% correctly indicates healthy GPU/inference utilization. |
| `no thread-count hint (disturbed)` | **Yes** | > 1.3x difference on identical configurations reliably detects thermal/scheduler noise. |
| `X vs Y threads differ by Z% (noise)` | **Partly** | Correctly avoids noise traps, but only looks at the absolute minimum and maximum threads, missing optimal intermediate points. |
| `X threads is Y% faster than Z` | **Partly** | Same flaw as above. |
| `choice changed with thread count` | **No** | Falsely attributes baseline non-determinism (between identical thread-count runs like `main` and `t4`) to thread scaling. |
| `cold load is X slower than cached` | **Yes** | > 1000ms and > 25% are robust thresholds for resident-worker advice. |
| `phase is X% of cached load` | **Yes** | >= 50% effectively identifies the specific initialization bottleneck. |
| `first call takes X ms vs median` | **Yes** | >= 2x and 200ms reliably identifies uninitialized lazy components. |
| `choice changed under reversed order`| **Yes** | 10% flips across cases mathematically proves tie-breaker / order bias beyond chance. |
| `accuracy falls from X to Y` | **Yes** | 30% drop explicitly catches degradation on harder conceptual levels. |
| `sensitive to phrasing` | **Yes** | 25% gap effectively highlights prompt sensitivity. |
| `truncated decisions hit limit` | **Yes** | Any value > 0 is data loss the user must know about. |
| `peak RAM is X% of device RAM` | **Yes** | >= 40% is the exact mathematical boundary where 2+ resident models become unsafe. |

### Overall Verdict
**Fail.** 
While the timer instrumentation (especially the avoidance of double-counting via the `Timed` wrapper) and the thermal noise guard (`repeat_worst`) are very well designed, the script fails on evaluating its own thread-scaling experiments. It falsely attributes non-determinism to thread counts, and its thread scaling logic blindly skips intermediate data points. Resolve the two major logic bugs and you will have a highly reliable tool.

---

**My read of the findings** (Coder's view, separate from the review):
- **1 and 2 — agree, both real.** The thread hint should rank every thread count, not just the lowest and highest. The main block and the t4 block are the same configuration, so a difference between them is run-to-run non-determinism, not a thread effect. It should be reported under its own label, not blamed on threads.
- **3 — I think the premise doesn't hold here.** None of these models keeps a cache across options. Six of the seven score all options in one sequence, so the token sum is just that sequence. von10 NLI really does run one full premise+option pair per choice, so the sum is what it processes. Padding in its batch is the only gap.
- **4 — agree.** It's a one-line guard.
- **One the reviewer rated "yes" that I disagree with:** the level and phrasing hints fire on as few as 2 decisions per group. The toy report's "100% vs 0%" comes from 2/2 vs 0/2. They need a minimum sample size before firing.

Do you want me to fix 1, 2 and 4 and add the minimum sample size, then run a new review? After that, the next step would be a smoke run of the five unexercised models on the toy cases, then your real ladder file.

---

## Message at 2026-09-25T21:26:26Z

The six fixes are in, the round-2 review passed, and the smoke run on the other five models finished cleanly: exit 0, stderr empty, no leftover workers or `llama-server`. All seven models have now been through the ladder. No commit and no push (this task said none, and the reviewer's "Ready to commit" is a review finding, not permission).

**What changed**
1. **Thread hint** ranks every thread count and compares the fastest with the lowest, gated on the measured noise.
2. **Main vs t4:** a difference between these same-thread runs is reported as "run-to-run non-determinism". Choice changes at other thread counts are blamed on threads only if the same-thread repeats matched; otherwise the hint says the two can't be separated.
3. **Token definition** is printed in each model's report section.
4. **Epsilon guard** on the repeat ratio. The new test also caught the same 0.0 ms problem in tokens/s and the time-split shares, so those are guarded too. A negative fitted fixed term is capped at a 100% per-token share, with the term still shown.
5. **Level, phrasing and order-flip hints** need at least 10 decisions per group; for flips that means 10 cases, since each case is one written-vs-reversed comparison. Below that they print the counts and "too few to judge".
6. **Cold load:** before each cold block the runner prints your `su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` line and waits for Enter. It then checks `/proc/meminfo` and labels the load "full-cold" only if the page cache fell below half its size. Otherwise, or if you type `s`, it uses weights-cold and prints the reason. With no terminal, as in my unattended smoke run, it doesn't wait and falls back.

**Not yet exercised:** a real full-cold load, because it needs you to run the `su` line in native Termux. In the toy rerun I pressed Enter without dropping the cache; the runner caught it ("page cache did not drop (3281 -> 3281 MiB)") and fell back. The smoke run had no terminal, so all five cold loads there are weights-cold.

**Round-2 review, verbatim** (AGY 1.2.11, `gemini-3.1-pro-high`, `--sandbox`, no `--dangerously-skip-permissions`; status SUCCESS; stderr empty; frozen files unchanged afterwards)

---

Here is the independent correctness review of the round-2 candidate.

### 1. Fix Item Verification

| Item | Verdict | Evidence |
| :--- | :--- | :--- |
| **1. Thread hint ranking** | **Fixed** | Ranks thread counts from fastest to slowest, identifies the best (with tie-breaker for fewer threads), and compares the fastest against the lowest thread count. Noise-gated correctly using `max(0.15, rw["ratio"] - 1)` (`ladder.py:335-348`, `test_ladder.py:37-40`). |
| **2. Non-determinism vs threads** | **Fixed** | Effectively separates the block of repeats run at `threads_main` into `same` (non-determinism) and compares against `other` (thread effect). Correctly suppresses the thread-effect hint if same-thread non-determinism is detected (`ladder.py:255-257`, `349-361`). |
| **3. Token definition** | **Fixed** | Explicit `TOKEN_DEF` mapping added to the worker, correctly populated for all seven adapters, returned in the `ready` payload, and printed in the report block (`ladder_worker.py:35-43`, `ladder.py:442`). |
| **4. Epsilon guard** | **Fixed** | Division points now safely wrapped with `max(1e-6, ...)`: `ratio` lambda, `tok_per_s`, `split_share`, and the fit's `share` calculation (`ladder.py:261`, `272-274`, `308`). |
| **5. Min sample size** | **Fixed** | Added `MIN_N = 10`. Order flips, level, and phrasing hints explicitly branch to output "too few to judge" alongside the counts when $n < 10$. The logic for phrasing elegantly drops undersampled phrasings from the comparison while still checking the well-sampled ones (`ladder.py:380-399`). |
| **6. True cold load** | **Fixed** | Uses `sys.stdin.isatty()` to prompt the user to drop the cache in native Termux. Verifies the drop using `/proc/meminfo`, correctly falls back to `weights-cold` (posix_fadvise) if skipped or if the cache drop was negligible, and logs the explicit reason (`ladder.py:114-131`). |

---

### 2. Findings

**None.**
The implementation is mathematically and logically flawless. The handling of the same-thread repeats vs other-thread counts is excellent, as is the logic for filtering out undersampled phrasings (`v["correct"][1] >= MIN_N`) before determining the "best" and "worst" phrasings.

---

### 3. Per-Hint Verdict Table

| Hint Category | Does advice follow from evidence? | Can it fire on insufficient or noisy data? |
| :--- | :--- | :--- |
| **Latency Fit & Phase Shares** | Yes. Highlights whether token count or glue code is the bottleneck using strict math (R2, slope, phase medians). | **No.** Requires $N \ge 2$. High $R^2$ guarantees a deterministic relationship even at small $N$, as model compute times have negligible variance. |
| **Thread Scaling** | Yes. Recommends the lowest thread count unless a higher count provides a gain above the measured noise floor. | **No.** Protected by the `noise` gate (derived dynamically from same-thread repeat jitter, floor 15%) and utilizes medians to absorb OS jitter. |
| **Run-to-Run Non-Determinism** | Yes. Exposes non-reproducibility and prevents it from being falsely attributed to a thread effect. | **No.** A single change in choice between identical setups is absolute proof of non-determinism. |
| **Load Times (Cold vs Cached)** | Yes. Pinpoints whether eviction, library imports, or weights format initialization dominates startup time. | **No.** While these rely on single measurements, model load times are massive (seconds) and highly deterministic. |
| **Warm-up Call** | Yes. Warns when the first call is $\ge 2\times$ slower than the median. | **No.** Gated by a 200 ms minimum gap to avoid firing on trivial overheads. |
| **Order Bias (Flips)** | Yes. Recommends mitigation if reversed prompt order changes $\ge 10\%$ of decisions. | **No.** Explicitly gated by the new `MIN_N = 10` requirement. |
| **Level Degradation** | Yes. Highlights models that break down on harder situations ($\ge 30\%$ drop in accuracy from easiest to hardest level). | **No.** Explicitly gated by `MIN_N = 10` for both the minimum and maximum levels being compared. |
| **Phrasing Sensitivity** | Yes. Exposes prompt sensitivity ($\ge 25\%$ accuracy gap between the best and worst phrasings). | **No.** Excludes any phrasing with $< 10$ decisions before running the comparison. |
| **Truncation & Peak RAM** | Yes. Factual observations on maximum sequence length and VmHWM footprint. | **No.** Absolute threshold measurements. |

---

### 4. Overall Verdict

**PASS**
Every requested fix was implemented completely and robustly. The analysis logic correctly bridges the gap between raw metrics and actionable advice, and the epsilon guards effectively prevent runtime math errors on trivial edge cases. Ready to commit.

---

**Where I disagree with the review:** it's overconfident about hints firing on small samples, and I'd weigh that before the real run.
- **Load hints come from a single load each.** The review calls load times "highly deterministic", but they aren't. In the toy rerun, von11's library import took 16.0 s cold and 18.0 s cached, so the cold one was faster. A cold-vs-cached gap under about 2 s shouldn't be trusted from one sample.
- **Thread medians are from 3 decisions each.** In the smoke run, laya_multi's "4 threads is 16% faster than 2" only just clears the 15% floor. With your real case set the medians will be much steadier.

**Smoke run** (full report: `/termux-home/ladder/smoke_run/report.txt`)

| Model | Correct (both orders) | Median decision | Tok/s | Cached / cold load | Peak RAM |
|---|---|---|---|---|---|
| s1o | 6/6 | 6055 ms | 18 | 6.1 / 9.6 s | 3545 MiB |
| laya_multi | 3/6 | 366 ms | 229 | 18.5 / 26.6 s | 2650 MiB |
| laya_micro | 3/6 | 290 ms | 348 | 2.7 / 3.1 s | 524 MiB |
| von12 | 6/6 | 830 ms | 99 | 19.4 / 25.7 s | 3442 MiB |
| von10 NLI | 6/6 | 2090 ms | 122 | 17.7 / 22.5 s | 1818 MiB |

Cold loads here are weights-cold. With 3 toy cases, the accuracy column only shows the models ran, not how good they are.

What the smoke run showed about the runner:
- **Every model's timing hooks fired.** The worker refuses to run if tokenise or forward time is zero after warm-up, and none refused. Forward is essentially all of decision time for every model, and tokenisation is at most 9 ms (s1o, whose tokenizer is an HTTP call).
- **The disturbance gate caught a real case.** s1o's repeated 4-thread decision took 6018 ms in one block and 8140 ms in the other (1.35×), so its thread hint was withheld.
- **s1o's fit has a negative fixed term** (−407 ms, with 3 distinct token counts), so its per-token share shows as the capped 100%. Your real case set should give a proper spread.
- **s1o's import shows 0.00 s:** it imports nothing heavy. Its init includes starting `llama-server` and loading the model.
- **Nothing changed across runs or thread counts:** no model changed a choice between repeats or thread counts, and no input hit its token limit.

**Files** (`/termux-home/ladder/`)
- `ladder.py`, `ladder_worker.py`, `test_ladder.py` (passes), `toy_cases.jsonl`
- `toy_run/` and `toy_run.pty.txt` — toy run on laya_en and von11 with the new code, including the prompt transcript
- `smoke_run/`, `smoke_run.stdout.txt`, `smoke_run.stderr.txt` (empty) — the five-model smoke run
- `old_runs/` — the earlier toy runs
- Review copies: `/termux-home/ladder-review-20260925T204237Z/` (round 1) and `/termux-home/ladder-review-20260925T211413Z/` (round 2). Each holds the request, frozen files, hashes and the raw review JSON.

**Next step:** give me your ladder JSONL and run the runner from a real Termux session, so you can drop the cache at each prompt:
`python3 /termux-home/ladder/ladder.py YOUR_CASES.jsonl --out /termux-home/ladder/real_run`

With all seven models it pauses seven times, once per model.
