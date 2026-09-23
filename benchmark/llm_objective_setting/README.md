# Pixel 7 on-device LLM conversation and objective-setting archive

Completed phone experiment, retained as a baseline. This tests an on-device LLM in the role it retains under [DECISIONS #109](../../docs/DECISIONS.md): conversation, Q&A and setting a bounded objective outside the low-level navigation loop. It predates the revised robot cycle. Bucket C scores a **PROPOSED** dynamic-map prompt, PARSE_SYS-D, not deployed code. Its value is as a baseline and as rubric experience for the next benchmark; it does not select a model for deployment.

This experiment is separate from the [Laya/Von strategic selector](../strategic_selector/README.md). It did not run motors, camera, YOLO, LiDAR or a live mission. The current deployed `navigate_to` path is a hardcoded five-room enum. The PoC prompt instead used exactly `kitchen`, `living_room`, `schlafzimmer`, `bureau`, `bathroom`, `balkon`. Bucket C therefore measures synthetic command interpretation under the proposed prompt, not navigation success, person identity or physical safety. See [DECISIONS #104](../../docs/DECISIONS.md), [#106](../../docs/DECISIONS.md), [#108](../../docs/DECISIONS.md) and [#109](../../docs/DECISIONS.md) for the surrounding architecture and separate experiments.

## Archived experiment

[runs/](runs/) contains the final available `bench.py` and every `~/bench_results_*.json` found in the phone home directory. [ARTIFACTS.md](ARTIFACTS.md) records byte sizes and matching phone/archive SHA-256 values. No weights are in Git.

The harness ran in Termux on a rooted Pixel 7 against llama-server build 1609 (`e1a1abb7`), context 4096 and `n_predict=512`; `--swa-full` was used for Gemma only. Temperature was 0.1 for A/B and 0.05 for C. A has seven conversation prompts, B has seven Q&A prompts including two honesty traps, and C has 13 objective-setting prompts. Full configs have 81 rows: (7 + 7 + 13) × 3. The Qwen think-off smoke has 27 rows. A/B answers were collected but never blind-graded.

The harness used family-specific prompt wrappers and stop tokens, Qwen3.5 think-on/off modes, a random system-message nonce to defeat prompt-cache reuse, cache dropping and cooldown between configs, and shuffled M-ids with a separate key. `/tmp` was unwritable in Termux, so logs moved to `$HOME`. An echoed nonce broke a greedy JSON regex; the replacement extractor is string-aware but still accepts arrays nested inside a bare object. On build 1609, `stopped_limit` was always null, so truncation is identified by `stop_type == "limit"`. Decode tokens/s is token-weighted over turns with at least 10 generated tokens. Reported TTFT is the server's `prompt_ms` proxy, not a streamed first-token timestamp.

Thermal and `/proc/meminfo` values were sampled outside the timed inference call. Configs ran back-to-back on a hot phone, so absolute seconds are thermally contaminated worst-case observations. The large relative Gemma speed gap was measured in these runs, but there was no thermally controlled comparison.

The result rows do not contain a `source_sha256` field. Association between any run and exact grader bytes is therefore **inferred from file times and behavior, not recorded by the harness**. The September 19 Gemma results predate the final source file's September 21 modification time. The September 21 Qwen results follow it, and the think-off file's survival of the former non-dict crash is consistent with the guard being active, but neither observation proves an exact source hash. The next harness must put its source SHA-256 in every result file.

## Stored run-1 results

These are the original harness summaries. C percentages use run 1 only; speed uses all successful turns. The stored refusal rubric is defective: it treats anything other than `navigate_to` or `find_object` as a valid refusal, so these quality columns are historical evidence rather than the final regrade.

| Config | Stored parse_ok | Stored exact | Cross-lingual | Critical fails | Stored reject violations | Decode tok/s weighted | TTFT median | Turn latency median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-e2b | 100% | 84.6% | 83.3% | 0 | 1 | 8.34 | 13.4 s | 16.4 s |
| gemma-e4b | 100% | 84.6% | 83.3% | 0 | 1 | 4.48 | 30.6 s | 35.8 s |
| qwen3.5-2B-think-off | 30.8% | 7.7% | 0.0% | 0 | 0 | — | — | — |

## Post-hoc refusal and parse regrade

The corrected structural refusal rule accepts only `[]` or exactly one `say` action. Every other parsed action shape is not exact and is a reject violation. This fixes two shared-rubric defects: `find_person` escaped the old violation check on Gemma E4B C7 run 2, and `patrol` escaped it on Qwen C8.

| Config / parse regime | Run-1 corrected exact | Run-1 violations | All-run corrected exact | All-run violations |
|---|---:|---:|---:|---:|
| gemma-e2b, array | 11/13 (84.6%) | 1 | 32/39 (82.1%) | 3 |
| gemma-e4b, array | 11/13 (84.6%) | 1 | 30/39 (76.9%) | 3 |
| qwen3.5-2B-think-off, lenient object | 8/13 (61.5%) | 2 | 8/13 (61.5%) | 2 |

The Gemmas tie only on run 1. Across all three runs E2B is slightly ahead, while E4B decodes at about half E2B's weighted rate and has roughly 31 s median TTFT. E4B therefore bought no objective-setting gain at about twice the latency cost; the all-run regrade strengthens that finding.

C8 is not a structural refusal problem for either Gemma: E4B returned `[]` in all three runs; E2B returned one `say` in all three. An interpretation question remains deliberately open. E2B run 1 said `Don't go to the kitchen`, echoing the command rather than clearly refusing. Under the structural single-`say` rule, E2B remains 84.6% run 1 and 82.1% all-run. Under a semantic rule requiring an explicit refusal, it falls to 10/13 (76.9%) run 1 and 31/39 (79.5%) all-run. Whether that echo also counts as a reject violation is undefined. The next benchmark must define “refusal” before running instead of deciding it retroactively.

Both Gemmas fail C7 in all three runs: E2B emits `navigate_to schlafzimmer` 3/3; E4B does so 2/3 and emits `find_person` once. Both also answer C13 with only `say`, never `find_person`. These are distinct operational failures and should not be hidden by one aggregate percentage.

Qwen think-off has three different parse figures from the same raw 13 turns:

- harness: 4/13 (30.8%), because nested `rooms` arrays inside bare objects were accepted;
- strict post-hoc: 2/13 (15.4%), requiring a top-level array;
- lenient post-hoc: 13/13 (100%), accepting a bare object as one action.

Parseability is not competence. Under the lenient regrade Qwen is only 8/13 exact after fixing C8. Two turns contain values outside the six-label map: C1 emits `bedroom` rather than `schlafzimmer`, and C8 emits `" balkon"` with a leading space. C1 is a critical invalid-room response. C7's navigation and C8's patrol are reject violations.

Both Gemmas emitted arrays throughout run 1, so the headline is unaffected by strict versus lenient parsing. Across all runs, E4B C11 run 3 emitted a nonce prefix followed by a bare object: `[vgk76iAoTkNZ] {"type":"find_person",...}`. Neither post-hoc regime accepts that prefixed shape, giving E4B 38/39 parseable C rows. The earlier blanket claim that Gemma always emitted arrays was therefore wrong.

## Invalid think-on result

Qwen3.5-2B-think-on quality is **INVALID**. It hit the 512-token cap on 34/81 turns (42%), reached 204402 ms and timed out on C13 run 2. Truncation contaminates parse and exact figures, so the default aggregator excludes it from quality output. Qwen3.5-4B-think-on was not run; expecting it to be slower is inference, not measurement. `think_chars` is broken for Qwen3.5, reporting about 19 characters even on 512-token, 204-second turns. Retire it; latency and truncation are the retained measures of think cost.

## Grading and derived output

[aggregate.py](aggregate.py) reads `runs/bench_results_*.json` without modifying it and writes `combined_speed.tsv`, `combined_bucket_c.tsv`, `blind_ab.txt` and `blind_key.tsv` only beside itself. Strict parse requires a top-level array; lenient parse additionally accepts one bare object. Any non-dict array element marks the reply malformed and prevents exact credit. The corrected structural refusal rule is applied to both regimes. Invalid map labels on `navigate_to` count as critical failures. `--include-invalid` includes truncated configs for diagnosis only; it does not repair them.

The A/B transcript uses run 1, stable shuffled M-ids and a separate key. Keep that key outside any grading view and record any future grader and rubric. A5's stored row has only the final reply, so its intermediate generated turns cannot be fully graded.

## Incomplete and provenance

- A/B were never blind-graded; no conversation, honesty or Q&A quality claim is established.
- Qwen3.5-4B think-on/off were not run. Qwen2.5-3B was dropped after a crash and has no result file; before crashing it emitted `bedroom` rather than `schlafzimmer`.
- Server logs, original blind files, an exact phone-condition trace, model-weight hashes, per-run source hashes and invocation UUID manifests are absent or unverified. No missing evidence was reconstructed.
- Claude (Opus 4.6) coded the final `parse_and_grade_c` crash guard. Codex performed the read-only post-hoc regrades and implemented this archive and aggregator. The human explicitly **waived the independent review required by `docs/WORKFLOW.md` for this archive**, using the on-device crash smoke test and the reproducible post-hoc checks as gates. This waiver does not establish model correctness or authorize hardware use.

[RUN_INDEX.md](RUN_INDEX.md) lists each config and status. [ARTIFACTS.md](ARTIFACTS.md) records the copied evidence and hashes. No helper here starts a server, contacts a network, runs hardware, stages, commits or pushes.
