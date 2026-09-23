# LLM objective-setting run index

Completed on rooted Pixel 7 / Termux with llama-server build 1609. [runs/](runs/) contains every `~/bench_results_*.json` found at archive time plus the final available harness. A full config has 81 rows (27 prompts × 3); the think-off smoke has 27. “VALID” classifies the retained turns, not deployed behavior or independent review.

| Config | Runs | Archived result | Status | Result and limitation |
|---|---:|---|---|---|
| gemma-e2b | 3 | [`bench_results_20260919_083347.json`](runs/bench_results_20260919_083347.json) | VALID | Corrected structural refusal regrade: run 1 11/13 (84.6%), 1 violation; all runs 32/39 (82.1%), 3. C8's run-1 echo leaves semantic refusal quality open. A/B ungraded; exact run source hash absent. |
| gemma-e4b | 3 | [`bench_results_20260919_115350.json`](runs/bench_results_20260919_115350.json) | VALID | Corrected regrade: run 1 11/13 (84.6%), 1 violation; all runs 30/39 (76.9%), 3. Stored rubric had incorrectly credited C7 run 2 `find_person`. A/B ungraded; exact run source hash absent. |
| qwen3.5-2B-think-on | 3 | [`bench_results_20260921_164938.json`](runs/bench_results_20260921_164938.json) | INVALID-truncated | 34/81 cap hits and one timeout invalidate quality. Retained for diagnosis; default aggregate excludes it. |
| qwen3.5-2B-think-off | 1 | [`bench_results_20260921_054508.json`](runs/bench_results_20260921_054508.json) | SMOKE-ONLY | Parse: harness 4/13, strict 2/13, lenient 13/13. Corrected lenient exact is 8/13 (61.5%) with 2 violations; two turns use non-map room values. Two further runs absent. |
| qwen3.5-4B-think-on | 0 | none | NOT RUN | No result file; slower-than-2B expectation is inferred, not measured. |
| qwen3.5-4B-think-off | 0 | none | NOT RUN | No result file; no quality or speed measurement. |
| qwen2.5-3b | incomplete crash | none | DROPPED | No result file; observed `bedroom` instead of `schlafzimmer` is critical but cannot be aggregated. |

The Gemmas tie only in the run-1 headline; E2B leads after all three runs. The strict semantic reading of E2B C8 would instead produce 10/13 (76.9%) run 1 and 31/39 (79.5%) overall; that interpretation remains open pending an upfront refusal definition in the next benchmark.

The historical stored C fields use a defective refusal rule that misses action types such as `find_person` and `patrol`. [aggregate.py](aggregate.py) applies the corrected structural rule to raw replies. All result rows lack `source_sha256`, so exact grader-byte association remains inferred. The Laya/Von [strategic selector run index](../strategic_selector/RUN_INDEX.md) belongs to a separate experiment.
