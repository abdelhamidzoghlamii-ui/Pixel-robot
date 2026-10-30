# MEM-1 — memory-context benchmark

**GitHub archive status: prepared benchmark; no run artifacts archived here.**
Owner report (2026-10-29): the MEM-1 primary benchmark was completed on the phone
and its artifact manifest was owner-verified (`MEM1_ALL_HASHES_OK`). Those
artifacts are still phone-local. This report is not a GitHub-archived or
independently audited measurement; precise metrics, validation labels and the
scope of completed experiments require the original artifacts. The run index
below lists only archived runs.

## Purpose

How does a remembered-context block of a given token length affect the robot's Gemma speed (prompt processing,
generation, wall time), recall accuracy, RAM and heat? How much does prefix caching help when the instructions and
memory stay unchanged between questions? Local AI uses the answer to decide whether a memory system is practical.
This is research only and changes no deployed behavior.

## Runtime under test

The robot's deployed text runtime, as `server_manager.start_setup('setup_q4')` launches it, read from the code:

| Item | Value | Checked by |
|---|---|---|
| Binary | `~/llama.cpp-b1609-dotprod/build/bin/llama-server` (b1609, commit `e1a1abb7`, ARM dotprod, #114) | SHA-256 pinned (launcher `77eb35a0…`, `libggml-cpu.so` `a4ef0d97…`); every regular file of the bin folder hashed into `run.json`, plus `libggml-cpu.so` (a symlink, which that skips) under its own name |
| Model | `~/models/gemma-4-E2B-it-Q4_0.gguf` | SHA-256 pinned `8e30dff3…` (same as #137/#142) |
| Flags | `--threads 4 --threads-batch 4 --parallel 1 --swa-full --cache-ram 0 --host 127.0.0.1`, `--ctx-size` 2048 (4096 only in the capacity/long launches) | `mem1.command(2048) == taskset + coresidency.server_cmd()` (test) |
| CPU | `taskset -c 4-7` for every condition, as the chat benchmarks (#137, #142); cores checked before and after every request | `Cpus_allowed_list` at block start and end |
| Not used | MTP drafter, vision projector | test |
| Template | Gemma 4 turns `<\|turn>…<turn\|>` from `chat.build_prompt('gemma', …)` (TPL-FIX, `70f7508`); thinking off; never the Gemma 3 markers | every block, before any generation: `/apply-template` (GGUF template, `enable_thinking: false`) must render byte-identical prompts, or the launch stops |
| Decoding | temperature 0, top_k 1, seed 0, n_predict 48, stop `<turn\|>`, not streamed | `run.json` |

## Frozen set (`memory_set.json`, SHA-256 `8feb74f9…`, pinned in `mem1.py`)

Invented facts only (no user history). Built by `build_set.py` with `llama-tokenize` from the original b1609 build
(same commit `e1a1abb7` and vocabulary; vocabulary load only, no inference). Every prompt is
`system: instructions + memory`, `user: question`. Memory entries are `- Jan 5: <fact>` lines, oldest first,
with all dates in 2026 stated once in the header. Two tasks share the memory, the questions and the prime and differ
only in the instructions: **recall** (scored, structured; primary, capacity, long) and **freetext** (unscored prose;
`--exp freetext`).

| Target memory tokens | 0 | 128 | 256 | 512 | 1024 | 2048 (long only) |
|---|---:|---:|---:|---:|---:|---:|
| Achieved (tokenizer) | 0 | 121 | 249 | 519 | 1029 | 2055 |
| Deviation | 0 | −7 (−5.5 %) | −7 (−2.7 %) | +7 (+1.4 %) | +5 (+0.5 %) | +7 (+0.3 %) |
| Filler distractors | 0 | 1 | 8 | 24 | 54 | 113 |
| Recall prompt tokens | 138–140 | 259–261 | 387–389 | 657–659 | 1167–1169 | 2193–2195 |
| Recall shared prefix (instructions + memory) | 125 | 246 | 374 | 644 | 1154 | 2180 |
| Freetext prompt tokens | 107–109 | 228–230 | 356–358 | 626–628 | 1136–1138 | 2162–2164 |
| Freetext shared prefix | 94 | 215 | 343 | 613 | 1123 | 2149 |

Memory tokens = prompt tokens minus the same question's memory-0 prompt; the difference is identical for all six
questions and both tasks (test). Every prompt plus 48 output tokens fits its context; a prompt that does not fit is
refused before anything starts (memory 2048 at ctx 2048 is refused). The server's own `/tokenize` must return the
frozen ids (count and SHA-256) of the launch's task in every block, or the launch stops.

**Memory-length comparability.** Six probe entries sit at fixed relative positions in every non-zero memory, exactly
once and in the same order (test); fillers from one frozen shuffled list fill the rest, and each longer memory
contains every shorter memory's fillers. The fillers are controlled distractors about other people (no probe name,
key value, charger, tea, dog or dentist; test), so a longer memory adds distractors and never changes a correct
answer. Memory 0 contains no entry: it is the unknown-answer baseline (every question's right answer is UNKNOWN) and is
not comparable to recall at memory > 0; it is used only for speed and as that baseline.

| Probe | Position | Entry |
|---|---|---|
| begin | first | Nadia's cat is named Pepper. (association) |
| stale | 1/4 | The spare robot charger is kept in the kitchen drawer. (superseded) |
| mid | 1/2 | Oskar's bicycle lock code is 4417. |
| near | right after mid | Oskar's gym locker code is 9032. (near-miss distractor) |
| update | 3/4 | Update: the spare robot charger is now kept on the hallway shelf. (later date supersedes *stale*) |
| end | last | Leo takes his tea with honey and no milk. (preference with a negated distractor) |

**Structured recall (the scored task).** The instructions ask for exactly one line of JSON, `{"answer": "VALUE"}`,
VALUE being the shortest exact name, number, word or phrase from the memory, or `UNKNOWN` when the memory does not
contain the answer. No key value appears in the instructions, a question or the prime (test). The frozen key, in
normalized form:

| Id | Question | accept (CORRECT) | stale | distractor |
|---|---|---|---|---|
| q_begin | What is the name of Nadia's cat? | `pepper` | — | — |
| q_mid | What is Oskar's bicycle lock code? | `4417` | — | `9032` |
| q_end | What does Leo take in his tea? | `honey`, `honey and no milk`, `honey, no milk` | — | `milk`, `honey and milk` |
| q_update | Where is the spare robot charger kept now? | `hallway shelf`, `the hallway shelf`, `on the hallway shelf`, `hallway` | `kitchen drawer`, `the kitchen drawer`, `in the kitchen drawer`, `kitchen` | — |
| q_absent_pet | What is the name of Nadia's dog? | — (only UNKNOWN) | — | `pepper` |
| q_absent_time | At what time is Oskar's dentist appointment? | — (only UNKNOWN) | — | — |

Parsing (`mem1.parse_answer`, no prose is interpreted): **STRICT** = the reply, outer whitespace aside, is exactly one
JSON object whose only key is `answer` with a string value; **FENCED** = that object is the only content of one
Markdown code fence (```` ```json ````); **MALFORMED** = anything else (prose, text around the object, other or
duplicate keys, a non-string value, truncated JSON); **EMPTY** = a blank reply. Normalization before the exact match
(`mem1.normalize`): Unicode NFKC, casefold, whitespace collapsed, one trailing period dropped — nothing else.

Outcomes (`mem1.score`): a fact present in the memory (memory > 0, present questions): **CORRECT** (an accept value),
**MISSED** (UNKNOWN), **STALE** (a stale value), **DISTRACTOR** (a distractor value), **WRONG** (any other value,
e.g. `Pepper and Max`, `not 4417`, `4417 or 9032`). A fact the memory lacks (absent questions, every question at
memory 0): **ABSTAINED** for UNKNOWN, **UNSUPPORTED** for any other value. Unparsed replies keep their format as the
outcome (**MALFORMED**, **EMPTY**) and a failed request is **ERROR**; all three are failures and stay in the
denominator. The four round-7 counterexamples ("Nadia's cat is named Pepper and Max.", "The code is not actually
supposed to be 4417.", "Hallway was wrong; now kitchen.", "I don't know; the appointment is in the morning.") are
MALFORMED as replies and WRONG/UNSUPPORTED/STALE when wrapped as values (test).

Accounting, reported separately per memory length and mode over **all** N scored requests: **format compliance**
(STRICT, FENCED, MALFORMED, EMPTY, ERROR counts; strict % = STRICT / N, parsed % = (STRICT + FENCED) / N) and **recall
accuracy** (every outcome count; accuracy = (CORRECT + ABSTAINED) / N, and parsed-only accuracy = the same over parsed
replies, labelled conditional). What it measures: whether Gemma retrieves one exact value from the supplied memory and
abstains when it is absent, under a fixed output format, at each memory length. What it does not measure: natural
conversational use of memory, paraphrase, reasoning over several facts, or how users would judge a prose answer; the
JSON format itself may change the model's behaviour, which is why compliance is reported on its own.

**Free-text task (`--exp freetext`, unscored).** The earlier instructions ("Answer in one short sentence … reply
exactly: I don't know.") with the same memory and questions. Every reply is kept verbatim in `requests.jsonl`
(`reply`), marked `outcome: UNSCORED`, `review: MANUAL REVIEW REQUIRED`; nothing grades it automatically and it is never
combined with recall accuracy. Its speed and resource figures are measured like primary.

A separate unscored **prime** prompt (`Reply with the single word: ready.`, after the same instructions + memory) is
frozen in the set with its token count and ids for both tasks; it shares exactly the scored questions' prefix
(`lcp_with_scored` = shared prefix, test). Its only role is the cache experiment (below); its reply is recorded and
never scored. For the cache counters the set also freezes, per task and memory length, the token LCP of every pair a
CACHED block can send in a row: warm-up → prime (`prime_prompt.lcp_warmup`), prime → question (`prime_prompt.lcp`)
and question → question (`prompts.<id>.lcp`; ≥ the shared prefix, symmetric, test). `mem1.expected` derives every
request's question, prompt tokens, shared prefix and expected `cache_n` from these tables and the plan (FRESH: 0); the
runner and the report both use it, and the block's no-generation checks refuse a server whose warm-up tokenization
gives another LCP. Adding the tables changed nothing else in the set (checked field by field).

## Experimental matrix (one launch = one line; nothing chains launches)

| Launch | ctx | Memory lengths | Blocks (2 rounds) | Requests |
|---|---:|---|---:|---:|
| `--exp primary --ctx 2048 --smoke` | 2048 | 0, 1024 | 4 (1 round) | 4 primes + 8 scored (2 questions per block). **Pipeline check, NOT a measurement.** |
| `--exp primary --ctx 2048` | 2048 | 0, 128, 256, 512, 1024 | 20 | 20 primes + 120 scored |
| `--exp capacity --ctx 2048` | 2048 | 1024 | 4 | 4 + 24 |
| `--exp capacity --ctx 4096` | 4096 | 1024 | 4 | 4 + 24 |
| `--exp long --ctx 4096` (optional) | 4096 | 2048 | 4 | 4 + 24 |
| `--exp freetext --ctx 2048` (unscored) | 2048 | 0, 128, 256, 512, 1024 | 20 | 20 primes + 120 replies for manual review |

A **block** = one server start, one memory length, one cache mode. Inside a block: load (spawn to `/health`), the
template/token/`n_ctx` checks for the prime and every question (no generation), one fixed warm-up request (`Hi`,
4 tokens, uncached, no memory; excluded from the statistics), the prime, then **all six scored questions**. Every
block of a round asks them in the same order (the set order rotated by the round number, recorded per block in
`run.json`'s `matrix`), so every memory length and both cache modes score exactly the same questions in the same
positions (test over the whole matrix).

Cache modes and request classes (what they mean on this b1609 build):

- **FRESH**: `cache_prompt: false` on every request. In b1609 this sets `n_past = 0`
  (`tools/server/server-context.cpp`, the `cache_prompt` branch), so the whole prompt is processed. The prime is
  class **PRIME**; the six questions are class **FRESH**.
- **CACHED**: `cache_prompt: true` on the prime and every question. The prime is class **FIRST**: the slot holds only
  the warm-up, so only its few shared tokens can be reused — the cold first request with memory, comparable across
  lengths because the prime text is the same. The prime leaves the instructions + memory in the slot, so each scored
  question is class **CACHED** and reuses the shared prefix from the previous request.
- FRESH/CACHED comparisons therefore cover the same questions; a comparison whose two sides end up with different
  valid questions (an invalid request on one side) is NOT COMPARABLE (`QUESTIONS_NOT_AS_PLANNED`).
- The label never comes from the request flag. Every reply is checked against the server's own counters:
  `timings.cache_n` must equal the token LCP with the slot's previous prompt (0 for FRESH), `prompt_n + cache_n`
  must equal `tokens_evaluated` (= the frozen token count), CACHED needs `cache_n >= shared prefix`, FRESH/FIRST
  need `cache_n <` shared prefix, `truncated` must be false (no context shift), and `prompt_ms`, `prompt_n`,
  `cache_n`, `predicted_n`, `predicted_ms` must be numbers. A reply that fails any check is kept but excluded from the
  statistics and makes its block INVALID.

**Order and drift control.** Round 1 interleaves FRESH ascending with CACHED descending
(0F 1024C 128F 512C 256F 256C 512F 128C 1024F 0C); round 2 is the reverse, so every condition's mean position is
the same and a linear drift cancels over a round pair (`--rounds` must be even). Before every block: cores 4–7
(waits), the skin gate (VIRTUAL-SKIN ≤ launch-start skin + 1.0 °C, at most 8 min, else WARM START), then start
conditions. A launch stops before a block at Android status ≥ 3 or battery ≥ 45 °C, and also when either value is
unreadable (fail closed). The launch-start reading must pass every field rule below, or nothing starts.

**Comparison eligibility (`mem1.compare`; `comparisons.json`, and `--report … --json PATH`).** Every comparison is a
record naming the independent variable, the dimensions allowed to differ, the dimensions required to be identical
(with a same/different flag each), the expected and actual question ids and repeats of both sides, completeness,
environmental validity and environmental matching, a verdict and machine-readable reasons:

| Comparison | Independent variable | Allowed to differ | Required identical |
|---|---|---|---|
| `<class> mem L / mem 0` (FRESH, FIRST, CACHED) | memory length | memory text and its tokens | everything in the one `run.json` (model, build, decoding, command, set, code, ctx); planned questions and repeats |
| `mem L CACHED / FRESH` | cache mode | `cache_prompt`, server prefix reuse | the same, per memory length |
| `capacity <class>: ctx 4096 / ctx 2048` (two launches) | context allocation | `ctx`, `--ctx-size`, run folder and times | task, exp, rounds, smoke, decoding, warm-up, prime, set, model, pinned, build and code SHA-256, thresholds, the command except `--ctx-size`, the planned matrix (questions, rounds, modes, memory) |

Verdict **COMPARABLE** only with no reason; otherwise **NOT COMPARABLE**, listing every reason:
`RUN_NOT_COMPLETE`, `SMOKE_LAUNCH`, `CONFIG_DIFFERS` (a missing dimension counts as different), `NO_VALID_REQUESTS`,
`QUESTIONS_DIFFER` (planned sides differ), `QUESTIONS_NOT_AS_PLANNED`, `REPEATS_NOT_AS_PLANNED`, `REPEATS_DIFFER`,
`ENV_INVALID` (a block behind it is INVALID, or none was recorded) and `ENV_UNMATCHED`. A favourable environmental
match never overrides another reason, and ratios (prompt ms, wall s, gen t/s medians) are shown only for COMPARABLE
comparisons. Sides come from the planned matrix, so a side that recorded nothing is still listed. **ENV_UNMATCHED**:
over every observation of the blocks behind the comparison (gate skin, preflight, every sample, postflight), VIRTUAL-SKIN
spans more than 1.0 °C, battery % more than 10 % (unchanged threshold), or the policy4/policy6 caps differ; a missing
value is UNMATCHED, never matched. A NOT COMPARABLE ratio is not evidence that memory length caused a difference.

**Telemetry (asynchronous; `observe`, `sampler_loop`, `telemetry_stop`).** One sampler thread per block, started
with the server, takes an observation at slot k = server start + k × `SAMPLE_PERIOD_S` (15 s) until the server stop
begins. No sensor, `su`, `dumpsys`, file scan or resource analysis runs on the request path: between requests the
runner only checks memory (`telemetry_stop`, before the warm-up and before every request; each ends the block
before that request): the sampler failed, stopped or latched a `su` timeout; no sample begun
in the last `SAFETY_MAX_AGE_S` = 30 s (the slot-0 sample still running at the server start is allowed until then; a
stuck or late sampler is not); the latest sample's safety reads failed (`su`, battery or `dumpsys` return code not 0,
or Android status / battery °C unreadable) — all fail closed; the latest sample at Android status ≥ 3 or battery
≥ 45 °C → safety stop (`safety_stop`, the same rule as the pre-block observation's, return codes included). Until the
slot-0 sample completes, the safety evidence is the pre-block observation taken just before the server start, and
at most 30 s may pass from the server start without a sample. Plus two existing single-file `/proc` reads without root (the runner's cores 4–7 check, the
server's `VmHWM`). The check runs between requests: a request already running (at most `REQUEST_TIMEOUT_S` = 240 s)
is not interrupted. A telemetry `su` call that timed out latches at once: the sampler takes no further observation
(the block then fails closed before its next request), the postflight observation is skipped (the block's `end`
records why; the block is INVALID), and the launch ends after that block (`root_left_running`, also when the sampler
outlived its join or the postflight itself timed out): killing the `su` client does not prove its root shell ended,
and every observation writes the same root output file, so no further observation starts.
A request never waits for a sample. Samples are buffered in memory (at most `MAX_SAMPLES` = 256; more are dropped and
counted, never reconstructed) and written with the block. A preflight observation before the server start and a
postflight one after the stop stay outside the timed interval. One observation is **one** `su` call (timeout
`ROOT_TIMEOUT_S` = 10 s) whose root shell first asks for cores 0–3 and nice 10 for itself (`taskset -p f`, `renice`;
Magisk does not pass the caller's affinity on) and then reads battery sysfs (%, status, °C, current and voltage →
W), `dumpsys thermalservice` (status, VIRTUAL-SKIN), the `thermal-cpufreq-*` cooling devices' `cur_state`/`max_state`
(`cooling`, optional supporting evidence: shell builtins only, no extra process or root call; any failed read or no
device found records `cooling: null`, never state 0, and never affects validity), the server's `/proc/<pid>/smaps_rollup` (PSS, RSS) and reports
its own `Cpus_allowed_list`, nice and CPU ticks. Cheap reads stay in Python: `/proc/meminfo`, the policy4/policy6
`scaling_max_freq` (a sample) or the full `repro.cpufreq` (pre/postflight), and the last 4 KiB of the thermal log
(not the whole file). Nothing is cached: every value is read fresh. The sampler thread asks for cores 0–3 and
nice 10 for itself (`os.sched_setaffinity`/`os.setpriority`, per thread on Linux); what it actually got, and every
refusal, is recorded (`sampler.cpus`, `nice`, `affinity_error`, `nice_error`), never assumed. Gemma's cores 4–7 are
not changed. The sampler waits with `Event.wait` (no busy wait); an overrun skips slots (counted as `missed`) rather
than bunching; the stop joins the thread for at most `ROOT_TIMEOUT_S` + 5 s, and a thread still alive is recorded.
A failed read keeps a bounded diagnostic (a `diag` list entry: operation, kind (`missing`, `non-zero return code`,
`unparseable`, `timeout`, `exception`), return code, root stderr and raw output each cut at 300 characters, UTC and monotonic time);
no credentials or device identifiers are read.

**Schedule (discrepancy recorded).** The owner's task names an "existing approximately 15-second monitoring
interval". The code before this repair actually sampled every 5 s (`coresidency` sampler, 15 s gap bound), took two
more `su` calls after every request, and `oneshot.sh`'s root thermal logger wrote every 5 s (#142). This repair follows
the owner's 15 s: sampler period 15 s with a 20 s gap bound (one missed slot → too sparse → INVALID), no reads
between requests, and the thermal logger at 15 s, pinned to cores 0–3 at nice 10 (it records what it got in
`thermal_logger_sched.txt`, copied into `run.json`). The frequency was not raised to make validation pass.

**Instrumentation (report section "telemetry").** Per block, recomputed from the raw samples and the sampler's own
count: attempts (`sampler.attempts`, counted before each observation, so one that raised or never returned is
included), failed (failed recorded samples plus attempts that left no record), dropped samples, missed slots, actual intervals (median/max), sample duration (median/max), `su` wall time
(total/max), sampler-thread CPU (`time.thread_time`) and root-shell CPU (utime+stime+cutime+cstime of the root
shell, i.e. including its finished children such as `dumpsys`; not system_server), seconds of sampling overlapping requests (and how many requests were touched) and overlapping the model load,
the CPU list and nice obtained by the thread and by the root shells; plus a launch total against the server time.
A cost total covers only the attempts that measured it: when any attempt lacks that figure (failed, raised, never
returned) or the attempt count is missing, the total is printed with `+?` as a lower bound — unknown, never zero.
Runner memory is recorded once per block (`runner_end`). These are the sampler's own costs; system_server's
`dumpsys` work and any thermal or frequency effect on Gemma are not attributable this way.

**Timing isolation.** Load time (`load_s`), server prompt evaluation (`prompt_ms`), generation (`predicted_ms`,
tok/s), client wall time per request (`wall_s`, `t_end`: taken when the reply arrives, before grading and request
checks) and telemetry cost are kept separate. Every report states that measurements were
taken with telemetry running and are **not an unmonitored baseline**; its effect on inference is not quantified (see
the validation plan below). No offline test is evidence of low overhead.

**Validity: one contract (`validate_block`), for the runner and the report.** Every reason is a
`{'code', 'detail'}` record recomputed from the recorded evidence; none = VALID. The report never reads a stored
`problems`, `valid` or `outcome`: `load_run` recomputes requests (`recheck_row`: mode, class, the planned question
for the round and position, the expected prompt tokens, shared prefix and `cache_n` from the frozen set via
`expected` — a row's own copies are only compared, a difference is a problem —, server timings, wall and times
(server prompt + generation time ≤ client wall time ≤ the request's block-time interval), the request UTC,
`truncated` exactly false (missing or null is no evidence), `prompt_n`/`cache_n` integers in 0 … the prompt's tokens
and `predicted_n` an integer in 0 … `n_predict` (48; exactly 48 when `stop_type` is `limit`), `predicted_per_second`
(if present) equal to 1000 × `predicted_n` / `predicted_ms` within 1 % (the server's formula), `prompt_ms` and
`predicted_ms` > 0 whenever `prompt_n` / `predicted_n` > 0 (work takes time), reply → outcome) and blocks (`validate_block`) and keeps the stored labels only for audit (`stored`,
`stored_problems`), listing any disagreement. INVALID can never become COMPARABLE.

| Code | When |
|---|---|
| `WARM_START` | the gate says warm start, or the gate record is missing |
| `ENDED_EARLY` | the block did not end with `all questions done` (refusal, failure, safety or telemetry stop, cores lost) |
| `READING_INVALID` | the preflight or postflight observation fails a field rule below |
| `READING_ORDER` | preflight not within 60 s before the server start; postflight not within 60 s after the stop, or not after the sampler ended (2 ms rounding allowance) |
| `TIMES_INVALID` | `mono_t0`, `server_s`, `sampling_end_s` not finite numbers ≥ 0 in order; the sampling end before the last request's end; requests overlapping, out of position order or begun before the model load ended; a reading whose UTC disagrees with its monotonic time by more than 2 s (relative to the first reading; a clock change invalidates); `load_s` not a number in 0 … 120 s; or no server pid |
| `SAMPLE_INVALID` | a sample fails a field rule (or is not an object; a missing server pid still checks it as a sample); its `mono` is not `mono_t0` + `t` (± 1 s); it took longer than `JOIN_S` = 15 s; it ended after the sampler ended; it began before the previous recorded sample ended (the sampler is serial) |
| `SAMPLES_EMPTY` | no valid sample begun within the sampling interval (a sample begun after the sampling end, a stop race, covers nothing) |
| `SAMPLE_GAP` | valid samples leave a gap > 20 s from the server start to the first, between two, or from the last to the sampling end, or run backwards |
| `SAMPLER_FAILED` | the sampler raised (recorded, re-raised) |
| `SAMPLER_STATE` | the sampler was still alive after the join, ended before the sampling end or after its bounded join (`server_s` + 15 s), its attempt count is not recorded + dropped samples (an observation raised or never returned), or its state is missing |
| `SAMPLES_LOST` | any sample dropped (buffer full) or the count missing |
| `SERVER_CORES` | the server's `Cpus_allowed_list` at start or end is not `4-7` |
| `AGENTS_PRESENT` | an agent process was found just before or just after the block (`agents.before`/`after`, `repro.agents`; the runner stops the launch if one is found before a block), or the record is missing |
| `CHECKS_FAILED` | `checks_problems`: the block's questions are not exactly its round's planned questions (smoke-aware); the recorded GGUF-template results are not `true` for exactly the prime and those questions; the server token counts are not the frozen counts (ints) for exactly those prompts; a recorded request has no check; the server `n_ctx` is not the launch's ctx as an int (an unknown `n_ctx` also refuses the block live); no chat-template hash; no warm-up evidence (uncached, 1–4 generated tokens, client wall time); or the block's own `ctx` is not the launch's |
| `REQUESTS_INVALID` | any request of the block is invalid after recomputation, or records another round, memory length or mode than its block |

Field rules (`observation_problems`; every field required, numbers finite and never a bool):

| Field | Valid only if |
|---|---|
| timestamps | `mono` a finite number ≥ 0, `utc` ISO 8601 with a zone; samples: `t` ≥ 0 and `t_end` ≥ `t` |
| reads | `root_rc`, `battery_rc`, `dump_rc` exactly the integer 0 |
| battery | % an integer 0–100; °C in −10 … < 45; status `Discharging`; samples: W within ±30 |
| Android thermal | status exactly 0 |
| VIRTUAL-SKIN | in (0, 80) °C |
| thermal log | no `thermal_error`; the stamp `at` well-formed; `thermal_age_s` a number in −2 … 30 s; and `utc` − `at` equal to that age within 2 s (freshness recomputed from the evidence, not from the absence of an error label) |
| CPU caps | policy4 and policy6 positive integers (anything else reads as missing). A cap that changes during the block is Android's thermal management: recorded and reported (`cap_limit`), never INVALID |
| samples: memory | MemAvailable and swap used in 0 … 65536 MiB (no `meminfo_error`); `smaps_rc` 0, `pss_pid` = this block's server, PSS an integer in (0, 16 GiB] |

**Strict numbers and JSON.** Artifacts are read with `strict_loads` (NaN, Infinity, −Infinity and overflowing
literals such as `1e999` are a parse error: the line is listed with its raw text and left out) and written with
`dumps` (a non-finite float becomes the string `'nan'`/`'inf'`, kept as evidence that fails validation; never written
as a number). An integer beyond ±2⁶³ is also a parse error (and written as its digits, a string); `num` never
raises on any int. `result.json` skipped entries must be objects with an integer block. Request timings must be
finite and ≥ 0; request arithmetic (`prompt_n` + `cache_n`) runs only on checked numbers; a reply that is not a string
is MALFORMED; report figures that are not numbers print as their short repr, never formatted. Malformed evidence is
INVALID or INCOMPLETE with a reason, not a crash.

Limits: the 20 s and 60 s bounds, the plausibility ranges and the 15 s period are tunable constants in `mem1.py`
(`SAMPLE_MAX_GAP_S`, `ENV_MAX_GAP_S`, `BATTERY_MIN_C`, `SKIN_RANGE_C`, `SAMPLE_PERIOD_S`) and unmeasured on the phone;
the rules are fail closed, so a transient `su` failure costs a block, not a false VALID; drift faster than 15 s is
seen only by `time_in_state` per block.

**Completeness and partial artifacts.** `run.json` records the planned `matrix` (every block with its scored
questions). The report reads every artifact defensively (`load_run`): a missing or unreadable `run.json`, or one
missing required keys, makes the report **INVALID** (no comparisons); a torn last line (a write cut off by a kill), a
malformed line, a record missing required keys or a duplicated block record is listed with its raw text and left out;
requests whose block has no usable record (a block interrupted mid-execution) are listed as orphans and kept out of
every aggregate; a truncated or partial `result.json` is listed and treated as absent. Nothing missing is filled in.
The recorded `matrix` must equal the matrix the configured experiment
produces (`schedule(exp, rounds, smoke)` with the frozen questions); a shortened or altered matrix is listed and the
run is INCOMPLETE (it is kept as recorded, never replaced). Provenance is required: `run.json` must carry `decoding`, `warmup`, `thresholds` and `telemetry` exactly as this
code records them (`configuration()`), `model_sha256`/`pinned_sha256` equal to the pinned model, server and CPU
library, `build_sha256` containing the pinned server and `libggml-cpu.so`, `code_sha256` with one well-formed hash
for exactly the benchmark's source files (`code_files()`), `command` exactly this benchmark's server command at the
run's ctx (`command(ctx)`: binary, model, threads, cache flags, cores 4–7; no drafter or projector), a task
matching its experiment and the frozen prime; anything missing or different makes `run.json` INVALID (no
comparison uses it, within or across launches). Requests recorded twice for one (block, position) are listed and left
out; when `artifacts.sha256` exists, every file it lists must still match (a file changed after the run is listed).
`exp` must be configured at its `ctx`, the command's `--ctx-size` must be that ctx, and `rounds` 1 for
a smoke launch, else even in 2 … 64; otherwise `run.json` is INVALID. Then each planned block is COMPLETED (right round/length/mode, finished, one prime, exactly the planned scored
questions answered), FAILED, SKIPPED (cores lost before it; listed in `result.json`) or MISSING (never reached or never
recorded). The run is COMPLETE only if every block completed, all planned scored requests were answered, `result.json`
says `all blocks done` and no artifact problem or orphan exists; otherwise the header says INCOMPLETE with the
reasons and every comparison is NOT COMPARABLE (`RUN_NOT_COMPLETE`). A complete launch whose `result.json`
records agent processes at its end (`agents_end` not empty) or port 8080 not free (`port_free_end` not true) is
labelled COMPLETE, TIMINGS NOT ELIGIBLE, and every comparison it enters (within the launch and capacity) is NOT
COMPARABLE with `AGENTS_PRESENT` / `PORT_NOT_FREE`: an agent seen at the end cannot be shown not to have overlapped
inference. A `run.json` with a wrong task, a `prime`
without an id, an empty or non-string `command`, or another set's SHA-256 is INVALID; shallow block records
(`start: {}`, missing times, non-object samples) are reported as INVALID with their reasons. Only these recognized
artifact cases are handled; any other exception (a programming error) is raised, not hidden (test).

## Integrity, observed performance and comparison eligibility

Three separate judgements; none implies another.

1. **Measurement integrity** (`validate_block`): VALID or INVALID per block, from requests, telemetry, timestamps,
   memory readings and safety evidence (the codes above). A missing, malformed or non-positive cap reading is
   INVALID; a cap that *changes* is not (smoke `20261028T011149Z`: policy6 2.850 → 0.851 GHz with Android status 0).
2. **Observed performance under Android thermal management** (descriptive): the timing table, with each block's
   policy4/policy6 cap range and its frequency limit (`cap_limit`): **LIMITED** (a cap reading below that policy's
   `hw_max`), **UNRESTRICTED** (every reading at `hw_max`), **UNKNOWN** (no sample, a cap missing, not positive or
   above `hw_max`, or `hw_max` missing or different between pre- and postflight). Readings are 15 s apart, so
   UNRESTRICTED means "at every reading", not "never capped"; an external 3 s diagnostic saw caps change between
   readings. Cooling-device states are shown as the highest recorded state per device with how many readings
   recorded it, or `cooling not recorded` (never 0). These timings are what the phone delivered, not a causal
   effect of memory length.
3. **Controlled comparison eligibility** (`compare`): unchanged. ENV_UNMATCHED (skin span > 1.0 °C, battery > 10 %,
   any cap difference, any missing value) keeps a comparison NOT COMPARABLE; each record also lists both sides'
   frequency limits (`frequency_limits`). Only COMPARABLE ratios attribute a difference to memory length, cache mode
   or ctx.

**Historical runs.** The report recomputes validity under the current rules and never writes into a run folder.
For a run recorded by another `mem1.py` it prints `provenance: recorded by mem1.py <sha>, reported by <sha>`; where
the stored block labels differ (the 01:11 smoke's cap-change reasons), the existing "stored validity differs" issue
marks the run NOT COMPLETE, so every comparison stays NOT COMPARABLE. A smoke stays SMOKE - NOT A MEASUREMENT. Runs
without cooling evidence show `cooling not recorded`.

**Risk: few or no eligible comparisons.** In the smoke, skin rose 29.2 → 37.3 °C over four blocks (about 2 min) and
policy4/policy6 caps changed inside three of them. A comparison needs every reading of all its blocks (four in the
full primary run, spread over the launch) within 1.0 °C of skin and at identical caps. The full launch can therefore
produce few or no COMPARABLE comparisons; that is a finding about this phone's thermal behavior, not something to
relax. A narrowly scoped next step needs its own review: gate each block on caps back at `hw_max` (and a tighter
skin return) before it starts, and measure in a pilot how many blocks stay UNRESTRICTED. The experimental matrix is
unchanged.

**Sessions (`--session`, `--resume`; primary only in `oneshot.sh`).** The same single run, folder and 20-block
matrix, executed in owner-operated sessions of `SESSION_BLOCKS` = 4 consecutive blocks (five sessions); nothing else
of the method changes (order, rounds, questions, model, timings, telemetry, safety stops, validity, the 1.0 °C
comparison rule, cap matching). The launch without `--session` is unchanged.

- Session 1 (`--exp primary --ctx 2048 --session`) writes `run.json` as usual plus `session_blocks: 4`, runs blocks
  0–3 and stops cleanly. Each later session (`--resume <folder name>`) runs the next four recorded-order blocks.
- **Gate reference:** every session gates on the launch-start VIRTUAL-SKIN in `run.json` (`start.skin`), never on its
  own start reading (which must still pass every field rule, or nothing starts).
- **Deferral, never WARM START or SKIPPED:** when the gate is not passed after 8 min, the preflight reading taken
  right after it is above the launch-start skin + 1.0 °C (or has no skin), or cores 4–7 are lost before the server
  start, the session ends and that block, not started, is the first one of the next session. Agent processes
  and the Android-status / battery stops end a session before a block as they end a launch.
- **Resume refuses**, before anything is written, unless: `run.json` is valid (every existing provenance rule) with
  `session_blocks` 4; there is no `result.json`; every artifact reads without an issue and no request lacks its
  block record; the recorded blocks are exactly 0…n−1 with blocks left, every one COMPLETED (a block that ended early
  can neither be finished nor repeated: that run stays INCOMPLETE; an INVALID but completed block does not stop it,
  as in a single launch) and none with a telemetry `su` timeout or a sampler that outlived its join
  (`root_left_running`: a root shell may still run, so no later session may observe either), no earlier session
  whose start reading or last preflight reading (`start`, `preflight` in its end record) had a `su` timeout or was
  interrupted while it ran (`OBSERVING`; fail closed, same reason), and the server logs are exactly theirs (a block started but not recorded, e.g. Ctrl-C
  mid-block, can never be repeated); every earlier session has
  readable start and end records; and the pinned model/server/library, `build_sha256`, `code_sha256` (every
  benchmark source, `oneshot.sh` included), the skin-gate threshold and the gate's 8 min are exactly as recorded.
- **Evidence is never overwritten:** `blocks.jsonl` and `requests.jsonl` are appended; `server_block*.log`,
  `sessionNN.json` (planned blocks, reference skin, logger scheduling, the hashes it verified; a resume writes it after
  its provenance check and before its start reading), `sessionNN_end.json` (end reason, recorded blocks, its start
  reading and its last preflight reading, each `OBSERVING` if a signal came while it ran, agents and port 8080 at its
  end; also written, as
  `interrupted: …`, when a signal, error or refused start reading ends it) and `sessionNN_report.txt` are created exclusively.
- **Reporting:** a session after which the run can never continue (those two cases) says so in its end reason
  (`RUN CANNOT BE RESUMED (INCOMPLETE): …`) and prints no resume command. A session that leaves blocks writes no
  `result.json`; its `sessionNN_report.txt` (and `--report`) says
  INCOMPLETE with the missing blocks. Only after the last planned block are `result.json` (its `sessions`; agents at
  any session end and port 8080 not free at any session end make the run TIMINGS NOT ELIGIBLE), `report.txt`,
  `comparisons.json` and `artifacts.sha256` written. The report lists every session; a missing or unreadable session
  record makes the run INCOMPLETE.

## Measurements

Per request (`requests.jsonl`): expected and evaluated tokens, `cache_n`, `prompt_n`, server `prompt_ms`,
`predicted_n`, `predicted_ms`, tok/s, client wall time and UTC, `ttft_proxy_s` (= server `prompt_ms`; the request is
not streamed, so **no true TTFT is measured**), stop type, truncation, the reply verbatim, for recall the parsed
`format`, `answer` and `outcome` (freetext: `UNSCORED`), scored or not (prime), problems, the full server timings.
Per block (`blocks.jsonl`): load s (first block: page cache not dropped), checks, server `VmRSS`/`VmHWM`/`VmSwap`
(own child, `/proc`), every telemetry sample (PSS from `smaps_rollup`, MemAvailable, swap, battery W, thermal,
caps, cost and timing), the sampler state, block times (`mono_t0`, `server_s`, `sampling_end_s`), the pre- and
postflight observations, gate, problems (the report recomputes them). Peak PSS, minimum MemAvailable and maximum swap
are computed in the report from the samples. Per launch (`run.json`): task, command, decoding, schedule,
set/model/build/code SHA-256, thresholds, telemetry configuration, the thermal logger's obtained scheduling, start
observation, the planned `matrix` and the prime; `result.json` adds the skipped blocks and the coverage (`complete`, per-block state,
reasons); `comparisons.json` holds the comparison records. The report gives median and nearest-rank P95 by memory
length and class, format compliance and recall accuracy (or the UNSCORED note), block validity and the comparison
verdicts. With two blocks per condition, P95 of 12 values is close to the maximum.


## Prerequisites and safety

Native Termux (prompt `~ $`), Magisk root, no Claude/Codex/AGY process, no other llama-server or benchmark runner,
port 8080 free, charger unplugged, battery ≥ 80 %, MemAvailable ≥ 3000 MiB, Android status 0, battery < 45 °C,
screen on with Termux in front (the launcher keeps the screen on and restores its timeout). No motors, firmware,
serial port, camera, microphone, LiDAR, `main.py`, `run_mission.py` or robot app is touched; no deployed default is
changed. The runner stops only llama-server processes it started itself (process group of its own child);
`oneshot.sh` stops only its own logger and cleans its own lock.

## Commands (owner, native Termux)

```bash
bash ~/robot/benchmark/memory_context/oneshot.sh --exp primary --ctx 2048 --smoke   # first: pipeline check
bash ~/robot/benchmark/memory_context/oneshot.sh --exp primary --ctx 2048
bash ~/robot/benchmark/memory_context/oneshot.sh --exp capacity --ctx 2048
bash ~/robot/benchmark/memory_context/oneshot.sh --exp capacity --ctx 4096
bash ~/robot/benchmark/memory_context/oneshot.sh --exp long --ctx 4096              # optional
bash ~/robot/benchmark/memory_context/oneshot.sh --exp freetext --ctx 2048          # unscored, manual review
```

Primary in five sessions of four blocks (instead of the single primary launch; same run):

```bash
bash ~/robot/benchmark/memory_context/oneshot.sh --exp primary --ctx 2048 --session   # session 1: blocks 0-3
bash ~/robot/benchmark/memory_context/oneshot.sh --resume <UTC>_primary_ctx2048        # sessions 2-5, one at a time
```

Report only (starts nothing; any folders; two capacity launches also give the 4096/2048 comparison):

```bash
python3 -B ~/robot/benchmark/memory_context/mem1.py --report ~/robot_runs/mem1/<run> [<run> ...] [--json PATH]
```

Runs are written to `~/robot_runs/mem1/<UTC>_<exp>_ctx<ctx>[_smoke]/` (`run.json`, `blocks.jsonl`,
`requests.jsonl`, `server_block*.log`, `result.json`, `report.txt`, `comparisons.json`, `artifacts.sha256`); the launcher's console log
and `thermal.log` (CPU-zone readings, not a heat state, #126) are in `~/robot_runs/mem1/`.

## Files

| File | Role |
|---|---|
| `mem1.py` | runner and report; reuses `chat_mtp_repro/repro.py` (cpufreq, top_share; its `conditions` is no longer used), `chat_mtp/chat_mtp.py` (spawn/stop, refusals), `coresidency.py` (root, dumpsys parsing, cores, HTTP; its 5 s sampler is no longer used) and `power_map.py` (skin gate) unchanged |
| `oneshot.sh` | owner launcher: `chat_mtp_repro/oneshot.sh` with this runner (diff: paths, argument check incl. `--exp freetext`, runner list) |
| `build_set.py` | builds `memory_set.json`; `python3 -B build_set.py --check` rebuilds it in memory and compares (vocabulary-only tokenizer, about 4 min) |
| `memory_set.json` | frozen memories, questions, answer key, both tasks' instructions and token counts |
| `test_mem1.py` | offline tests (fakes only; every phone interface — `su`, dumpsys, sysfs/proc, processes, sockets, the server — is guarded and fails the test if reached, in proot or native Termux): `python3 -B -m unittest -v test_mem1`, from this folder |
| `RUN_INDEX.md`, `ARTIFACTS.md` | run index and SHA-256 catalog, empty until runs are archived |

`.gitignore` makes only these eight files trackable here; anything else created in this folder stays ignored. Archiving runs into the repository later needs its own exception.

## Limitations and open measurement risks

- No true TTFT (non-streamed); `ttft_proxy_s` is the server's prompt time.
- Thermal drift can leave few or no COMPARABLE comparisons (see "Risk" above). Cap classification and cooling
  states are 15 s snapshots; the cooling read is unverified on the phone (a `null` shows it failed).
- FIRST has two observations per memory length per launch; P95s rest on 10–12 values.
- Sessions: the round-pair order balances a *linear* drift; changes between sessions (recharging, ambient, hours
  apart) are not linear and coincide with block positions (sessions 0–3, 4–7, … split the 10-block rounds). The
  unchanged ENV_UNMATCHED rule (skin span 1.0 °C, battery span 10 %, identical caps) still applies to blocks from
  different sessions, so recharging between sessions can by itself leave comparisons NOT COMPARABLE. A session
  interrupted mid-block (or killed without its end record) cannot be resumed; the run stays INCOMPLETE.
- Battery % falls during a 20-block launch; the 10 % match rule (kept, over every reading) may mark comparisons
  NOT COMPARABLE (ENV_UNMATCHED) by itself. The smoke launch shows block durations; Local AI decides whether the run length needs changing
  (a new review).
- **Telemetry overhead is not measured.** The sampler is asynchronous, its thread and root shells ask for cores 0–3
  at nice 10, and its own cost is reported, but `dumpsys` work in system_server, the binder and the root daemon are
  not pinned or attributable, and a pinned reader still adds heat and power draw on a shared SoC. No result is an
  unmonitored baseline and no interference-free claim is made; offline tests prove only the structure (no sensor work
  on the request path, bounded waits, fail closed).
- The root-side pin (`taskset -p f`, `renice` inside the `su` shell) is unverified on the phone: the recorded
  `root_cpus`/`root_nice` show whether it took effect.
- A 15 s period sees drift at 15 s granularity; a missed slot makes the block INVALID by design.
- MEM-1's own safety checks run between requests: a running request (≤ 240 s) is not interrupted, so a stop can come
  up to one request late; Android's and the kernel's own thermal protections are independent of MEM-1 and untouched.
- A timed-out `su` call: Python kills the `su` client, but nothing proves the root shell and its children ended. The
  launch ends after that block (no further observation shares the root output file), but the leftover shell may run
  on until it finishes. Unchanged dependencies keep two unbounded waits: `chat_mtp.stop` waits without a timeout after
  SIGKILL, and `oneshot.sh`'s persistent root thermal logger has no per-read timeout (its cleanup waits about 20 s and
  keeps the lock if the logger is still alive).
- A wall-clock change during a block (NTP) makes it INVALID (UTC must follow the monotonic clock within 2 s).
- Exact cache counters (`cache_n` = LCP) are from the b1609 source, not yet observed on the phone for these prompts;
  the smoke launch is the first observation. A mismatch invalidates rows instead of passing silently.
- Structured recall measures exact single-value retrieval under a JSON output format; a correct value phrased outside
  the frozen accept list (e.g. `Pepper the cat`) is WRONG by design, and format failures count against accuracy
  (compliance is reported separately so the two can be told apart). Natural conversation is only in the unscored
  free-text task, which needs a manual review.
- The fail-closed validity rules (field rules, 60 s / 20 s bounds) are unmeasured on the phone: real `su` or
  sampler hiccups may invalidate many blocks; the smoke launch shows how often.
- The page cache is not dropped: load times are warm-cache loads.
- Synthetic, English, six questions: recall results describe this set, not real conversations.
- RSS/PSS describe a server restarted per block, not a long-lived server accumulating turns.

## Telemetry-overhead validation plan (NOT YET EXECUTED; owner-operated, needs its own authorization)

Purpose: quantify how much the measurement telemetry changes Gemma's measurements. Nothing below has been run, and the
arms below need a code change (an experiment mode) that is not part of this candidate.

- **Constant safety floor in every arm.** Every arm keeps the same safety monitoring: the thermal logger at 15 s and a
  safety-only `su` read (battery sysfs + `dumpsys thermalservice`) every 15 s with the same `telemetry_stop` rules
  (30 s staleness, failed reads, status ≥ 3, battery ≥ 45 °C) and the same pre/postflight observations. The arms
  differ only in the measurement reads added to it (`smaps_rollup`, `/proc/meminfo`, the two caps):
  **F15** = added at every 15 s slot (as shipped), **F60** = added at every fourth slot (60 s), **FLOOR** = never.
  The floor's own cost is therefore not measured; a fully unmonitored arm is used only if the owner judges it safe,
  otherwise the result is stated as "relative to the safety floor", never as zero interference.
- **Validity per arm.** Safety coverage keeps the 20 s gap limit in every arm; the measurement-field coverage limit
  is the arm's period + 5 s (F15 20 s, F60 65 s; FLOOR none), so the 60 s arm is not excluded by construction. FLOOR
  and F60 blocks are overhead-experiment data only, never MEM-1 results.
- **Order.** A balanced three-arm (Williams) design: the six sequences F15-F60-FLOOR, F60-FLOOR-F15,
  FLOOR-F15-F60, FLOOR-F60-F15, F60-F15-FLOOR, F15-FLOOR-F60, one per session, so every arm is equally often in every
  position and after every other arm; repetitions in whole replicates of six sessions.
- **Frozen across arms.** Model, build, command, ctx 2048, one memory length and mode, prompts, question order, cache
  priming (warm-up, prime), repetitions, server restart per block (empty cache at start), page cache not dropped (the
  first block of a session is a discarded warm-up), the same skin gate and cooldown, one charge-state band per
  session; the obtained sampler and root-shell affinity/nice recorded per block (a block where pinning failed is
  reported, not silently pooled).
- **Measures.** Per request: server prompt ms, generation tok/s, client wall time. Per block: CPU utilization from
  `/proc/stat` busy jiffies (user + nice + system + irq + softirq) deltas for cpus 0–3 and 4–7 between pre- and
  postflight, plus the sampler thread and root-shell CPU from the telemetry section; `time_in_state` is frequency
  residency, reported separately, not utilization; skin and battery change from pre- to postflight.
- **Tolerance and analysis.** Pre-specified practical tolerance: a difference of at most 2 % in median prompt ms and
  median gen tok/s (owner may set another before the run). The unit of analysis is the session (requests within a
  block are not independent): per session and arm one median; arm differences paired within sessions; bootstrap
  intervals resample whole six-sequence replicates (each a balanced set of six sessions, one per sequence), so the
  sequence balance is kept in every resample; sessions are never resampled individually. An interval inside ±tolerance = "within tolerance", one crossing it
  = "not resolved", never "no effect". Utilization = busy / (busy + idle + iowait) jiffies per CPU group over the
  block (steal and guest are 0 on the phone and reported if not); idle and iowait are reported separately.
- **Exclusions.** Every block failing validation is excluded and counted per arm (and why); paired data from the same
  session are kept and reported, since monitoring frequency can change how often failures are detected. A session
  with an arm missing enters a sensitivity analysis only (primary analysis: complete sessions); detection-rate
  differences between arms are reported as a possible selection bias, not corrected away. The discarded warm-up of a
  session is one full block at F15 in every session (the same monitoring exposure whatever the sequence), followed
  by the standard skin gate and cooldown; carryover is then estimated by sequence position, not assumed away.
- **Size.** The first replicate (6 sessions) is a pilot: its spread sets the number of replicates needed to resolve the
  tolerance; pre/postflight temperatures can miss transient differences, so the in-block safety samples (equal in
  every arm) are compared too.
