# Finalists: speed and accuracy on the 24 judgment cases (provisional timings)

Pixel Robot, Coder session, 2026-09-25. Research only: no motors, no `main.py`, no `docs/`, no commit, no push. Held-out still sealed.

**Gate: laya_en passes and von11 passes, both provisionally. s1o is out: its best warm median is about 24.9 s, far over 3000 ms.**

The biggest caveat first. **Android moved Termux out of the full-CPU cpuset (0–7) into `/foreground` (0–5).** When that happens, `taskset -c 4-7` silently runs on cores 4–5 only, and cores 6–7 can't be used at all. That's why two von11 settings (6–7 at 3 and 4 threads) failed to launch. Earlier runs recorded no cpuset evidence, so the effective cores behind every timing below are uncertain. `measure.py` now records the allowed CPUs at the start and end of each run, and refuses to start if the requested cores aren't all available. For the clean run, keep Termux as the top app, screen on.

Phone CPUs: cpu0–3 Cortex-A55 1.8 GHz, cpu4–5 A78 2.35 GHz, cpu6–7 X1 2.85 GHz.

## laya_en

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
- **One adaptation was needed:** laya-micro's `runtime.py` reads `mask_token_id` from the encoder config, and the English config doesn't have it. I added it through an overlay checkpoint: symlinks to the real files, plus `mask_token_id` 50284, which is the tokenizer's own `[MASK]` and the id torch Laya uses. The first attempt without it failed with `KeyError: 'mask_token_id'`.
- **Order sensitivity:** laya_en is acceptable on **24/24 in canonical order but only 7/24 reversed**.

## s1o (out)

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

## von11

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
- **5c:** yes, a second backend exists. BertaBackend `von-1.0` is an NLI cross-encoder loaded from the cached `d8bb5e07` snapshot. It's order-invariant (0 flips), but slower and less often preferred. The other listed backends are unavailable offline: `deberta-v3` is a third-party model that isn't cached, and `laya` needs `local_backends`, which isn't shipped.
- **5d:** the best Von variant is **plain von11** (all three are 24/24 acceptable; plain has the most preferred and is fastest). It gets 2753 ms at 4 threads on cores 4–7.

## repeat_search: dump and judgment

The dump is at `/termux-home/v3-runs/repeat_search_dump.md`: 6 cases × 2 orders, with state, options, labels, and each finalist's choice and full distribution.

My judgment: **keep the label, but it grades a rule the instruction never states.**

- "Preferred = the nearest unsearched room" is a sensible efficiency tie-break, and the distances are in the state. But the instruction only asks for "the most useful next action" and says nothing about distance.
- The acceptable set (any unsearched room, with call and ask excluded) is what the instruction actually supports. All three finalists are 6/6 acceptable in canonical order.
- In canonical order the finalists mostly choose the **first travel option listed**, not the nearest one. So the preferred misses reflect position bias more than a labelling error.
- If preferred matters for the decision, the instruction should state the distance priority.

## Harness changes (untracked, not committed)

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
| cases.py | `892a946dab0ff0973224b1fceb01400661f7a0f40f22892ace306d2dd4678781` (unchanged) |
| s1o_probs_evidence.py | `1f176e9430d13f863d47c21002946fd38c2ffd56f254f029e851a2b820a58194` (unchanged) |
| heldout.jsonl | `96d41308eb0d2fa0c3bf30c1370eb93d1b4b7afbfd13cd433370cfd5cc3b2601` (still sealed, never loaded) |

Other artifacts:

- laya_en ONNX: `laya_en.onnx` `bb76cb05dfa3a36cf7d807437257b7b78b9cb6ea868ec7f36e9f4f879b16e968`, `.onnx.data` `487746363a8da57bcadb4345352997d22a0fb90d70aa22c6856668d023242aba`
- overlay encoder config: `40076416c0cb3edf99f5a1fd8c54100256f9347eb2fd028d154f438d255fdf86`
- run outputs: `/termux-home/v3-runs/fin/` (tables `table_*.md`, parity `laya_en_onnx_parity.txt`, cache check `s1o_cache_check.txt`, diagnostic `s1o_cache_diag/`, llama-bench `s1o_bench/llama_bench.txt`)

## Review (AGY, gemini-3.1-pro-high, Ponytail off, `--sandbox`)

**PASS WITH CAVEATS**, status SUCCESS in 50 s, no denied actions, stderr empty. The frozen workspace and the harness were verified unchanged afterwards. Verbatim:

