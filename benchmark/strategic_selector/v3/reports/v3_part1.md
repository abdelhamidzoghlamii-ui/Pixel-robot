# Harness v3: development run and review

Pixel Robot, Coder session, 2026-09-24/25. Research only: no motors, no `main.py`, no `docs/`, no commit, no push.

I've built harness v3 and run the full development set once for all five candidates. The reviewer's verdict is **FAIL / INCOMPLETE**. Its "critical" finding is contradicted by the result files, and it says it couldn't find the result JSONs even though they were in the workspace, so as it stands this is an incomplete review, not a real pass or fail.

## Development results, 66 cases, default frame `filtered_text`

Preferred / acceptable counts; reversed means the same cases with the option order reversed.

| Candidate | Preferred | Acceptable | Preferred (reversed) | Acceptable (reversed) | Invalid | Order flips | Load ms | First call ms | Warm median ms | Warm P95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **s1o** (zero-shot Gemma 4 E2B) | **46** | **64** | 38 | 55 | 0 | 16/66 | 7717 | 29139 | 27174 | 30903 | 3666 |
| laya_en (0.3.20) | 34 | 54 | 25 | 36 | 0 | 19/66 | 28488 | 5266 | 3498 | 4700 | 2811 |
| von11 | 33 | 49 | 28 | 46 | 0 | **55/66** | 23329 | 7974 | 2806 | 5213 | 3363 |
| laya_multi (stock fp32) | 26 | 37 | 21 | 34 | 0 | 17/66 | 33386 | 1105 | 1083 | 1327 | 2585 |
| laya_micro (int8, 15,188 vocab) | 22 | 32 | 31 | 43 | 0 | 29/66 | 2028 | 1966 | 1306 | 1536 | **540** |

Per family, canonical order. Each cell is preferred/acceptable out of 6 cases (not "x out of y"):

| Family (n) | laya_en | laya_micro | laya_multi | s1o | von11 |
|---|---:|---:|---:|---:|---:|
| new_room (6) | 6/6 | 6/6 | 5/5 | 6/6 | 0/0 |
| room_finished (6) | 3/6 | 0/0 | 1/1 | 3/6 | 3/6 |
| hall_hint (6) | 2/6 | 0/0 | 1/3 | 2/6 | 2/6 |
| target_confirmed (6) | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| possible_person (6) | 0/6 | 0/6 | 0/6 | 0/6 | 0/0 |
| all_rooms_first (6) | 6/6 | 6/6 | 6/6 | 6/6 | 2/2 |
| all_rooms_called (6) | 0/0 | 0/0 | 0/0 | 4/4 | 6/6 |
| localization_lost (6) | 0/0 | 0/0 | 0/0 | 6/6 | 5/5 |
| route_blocked (6) | 6/6 | 4/4 | 6/6 | 6/6 | 6/6 |
| repeat_search (6) | 1/6 | 0/4 | 1/4 | 1/6 | 1/6 |
| heard_from_room (6) | 4/6 | 0/0 | 0/0 | 6/6 | 2/6 |

- **No candidate got any `possible_person` case right**, in canonical order. Only s1o and von11 ever got `all_rooms_called` and `localization_lost` right.
- **Timings are provisional.** The s1o times are dominated by llama.cpp 1609 evaluating prompts at about 14 tokens/s with prompt caching off.
- **laya_micro's tokenizer matched the stock tokenizer on 0/132 calls.** The pruned vocabulary splits common words, for example `▁Mission` becomes `▁Mis` + `sion`. So its inputs are out of distribution, which explains why it scores below stock laya_multi.
- Other checks: every Laya and Von argmax equalled the model's own choice (132/132 each). s1o's answer letters held a median 0.9999 of the full-vocabulary probability (minimum 0.6302). There were no ties and no tracebacks.
- stderr: von11 printed the known Transformers `UNEXPECTED` head-key report (as in the archived 1.1 run); laya_en printed its invalid-temperature `RuntimeWarning`; laya_micro printed the harmless onnxruntime `/sys/class/drm` warning.

**Held-out set:** 66 cases, seed 7919, SHA-256 `96d41308eb0d2fa0c3bf30c1370eb93d1b4b7afbfd13cd433370cfd5cc3b2601`. It was never loaded, printed or scored. The runner refuses it without `--unseal-heldout`, and a test checks that.

## Item 7: laya-micro

- **Latency:** the quantizer times random inputs of 3 items × 259 tokens × 28 options (laya-micro's own 3-question example). The smoke test sent 1 × 171 × 4. On the same graphs, fp32 took 408 vs 1922 ms and int8 471 vs 2039 ms. int8 isn't faster than fp32 on this CPU at either size.
- **Vocabulary:** the README quickstart says `--seed-top 4096`, but laya-micro's own `docs/METHOD.md` gives `seed_top 4,096 -> 7,779` and `8,192 -> 15,188 (the shipped checkpoint)`. Rebuilding with 8192 gave exactly 15,188 with parity 2805/2805, and the harness uses that build.

## Item 8: s1o

- **Source:** build 1609's `populate_token_probs` uses `get_token_probabilities()`, which is a softmax over the raw `llama_get_logits_ith` logits, whenever `post_sampling_probs=false`. Only `true` reads the sampler chain.
- **Measured:** with `false`, the A–F probabilities are bit-identical (difference 0.000000) at T=0, at T=1.7 with top_k=2, and with a +5 logit bias on B. With `true` and a neutral sampler chain they match within 5e-6; with `true` and top_k=2 only two tokens come back.
- **Configuration:** the adapter sends `post_sampling_probs:false`, `n_probs:1000` and `cache_prompt:false`. It renormalises over the option letters, and raises an error if a letter falls outside the top 1000.

## Harness: `benchmark/strategic_selector/v3/` (untracked, not committed)

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
- Design, briefly:
  - Cases come from v2's unmodified `build_case` plus seeded numeric noise (the nearest room stays nearest, so labels don't change). Separate seeds: dev 3101, held-out 7919.
  - The runner reuses v2's `run_case`, logs every call (state, options, full distribution, argmax, ms) and makes no warm-up calls.
  - Each candidate runs in its own venv through `measure.py`: cores 4–7, offline, peak RSS measured externally with `wait4`.
- Run outputs: `/termux-home/v3-runs/dev-full/`. Report: `/termux-home/v3-runs/dev-full.report.md`.

## Review (AGY, gemini-3.1-pro-high, Ponytail off, `--sandbox`)

Status SUCCESS, 229 s, no denied actions. stdout and stderr were kept separately (stderr was empty), and the frozen workspace was unchanged afterwards. Verbatim:

