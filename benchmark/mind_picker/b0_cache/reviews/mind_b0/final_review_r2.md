This was a **source-only static review** of this packet. No tools, skills or sub-agents were used. The owner directed this fallback because Codex’s read-only sandbox could not execute commands in proot during #144 R3. Tests, phone artifacts, comparisons and checksum checks are **supplied evidence**, not independently reproduced results. Static review cannot prove runtime behaviour.

**The round-1 P2 is partially resolved.** The requested missing-`completion_probabilities` case now preserves R1–R4, records the exception in `run.json` through `info`, fails C1/C2/C4/C5, and writes the report and manifest. The offline test is present, and the supplied output reports all six tests passing. The broader requirement to retain results and always finalize after exceptions remains incomplete.

**P2 — Interrupts can discard accumulated results, and cleanup exceptions can bypass finalization.**

All citations below refer to `benchmark/mind_picker/b0_cache/b0.py` (**344 lines**), using section E’s full numbered source.

- **b0.py:133:** `    except Exception as e:`
- **b0.py:313:** `            rows, info = run_sequence(post, s, proc.pid, emit)`
- **b0.py:314:** `        except BaseException as e:  # recorded; the report and manifest are still written, then a stop re-raises`

A `KeyboardInterrupt` during `sequence()` escapes its `Exception` handler. Consequently, the assignment at line 313 never completes: `main()` retains its initially empty `rows` and `info`. Previously emitted requests can remain in `requests.jsonl`, but the resulting report loses those collected rows and their metadata. Recording the runner exception does not preserve them.

- **b0.py:319:** `                os.killpg(proc.pid, signal.SIGTERM)`
- **b0.py:324:** `                    proc.wait()`
- **b0.py:327:** `    print(finish(out, run, rows, info) + f'\n{out}')`

Cleanup executes outside the preceding exception handler’s protection. For example, the process can exit between `poll()` and `killpg()`, causing `ProcessLookupError`; an interrupt during `wait()` can also escape. Either path skips `finish()`, leaving the report and manifest unwritten. These are control-flow gaps beyond unavoidable filesystem failures.

Preserve the sequence’s mutable results where the outer handler can access them, record cleanup failures, and ensure finalization is attempted before re-raising an interrupt. Extend offline coverage to these paths.

No P1 or additional P3 findings.

| Criterion | Implementation versus confirmed definition | Assessment of supplied evidence |
|---|---|---|
| **C1** | Correctly uses `min(LCP, n−1)` for R2/R3/R5/R8. | **PASS supported:** cache/reference **391/391, 363/363, 370/370, 405/405**. |
| **C2** | Correctly tokenizes BOS + system opening + T(1), excluding the tail. | **PASS supported:** R5 cache **370**, reference **370**. |
| **C3** | Reports the post-separate-chat rebuild cost informationally. | **INFO supported:** R7 reprocesses **428**, caches **6**, total **434**. |
| **C4** | Compares all five prescribed prompt pairs using five-way renormalized probabilities, matching tops and maximum absolute difference. | **PASS supported:** all five tops match; max difference **0.0161785887**, reported **0.016179 ≤ 0.02**. Probabilities are within tolerance, not identical. |
| **C5** | Checks one recorded PID/start-time pair, 13 rows, zero recorded request errors, no recorded exception, exit zero and no SIGKILL. The exception-path defect above remains. | **PASS supported for this completed run:** PID **21342**, start **56010578**, **13** requests, **0** errors, exit **0**, no SIGKILL, exception **None**. |

Run **20261102T110006Z is VALID supplied evidence for this counters-and-probabilities experiment**. Its successful results are not invalidated by the unexercised failure-path defect. Actual R8 length is **449 tokens**, within 400–700; the five supplied action IDs are single and distinct; cached/fresh prompt hashes match for every required pair.

The supplied counters show growing picker calls reprocessing **36, 52 and 44** tokens, shared chat reprocessing **15**, and the picker after separate chat reprocessing **428**. Token counts per part are **294** for system/catalogue, **48** for mission/memory, **19** for tail, **10** for speech and **23** for companion system. Timings are correctly labelled **NOT MEASUREMENTS**.

The supplied criteria comparison reports no result changes from run 1; C5 adds the new PID/start-time and `exception None`. The requested ignore exception and README interpretations are present, and the frozen-set hash is unchanged in the supplied records.

**Final verdict: CHANGES REQUESTED** — complete failure-path preservation and finalization. This is a review finding only and does not authorize a commit.