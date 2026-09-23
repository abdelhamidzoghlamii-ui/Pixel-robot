# Archived phone artifacts

This archive contains the final available harness and every `~/bench_results_*.json` found in the phone home directory on 2026-09-23. It deliberately departs from the strategic selector's hashes-only phone-history precedent: DECISIONS #110 must be independently checkable from raw rows, and a successor thread needs the replies to reproduce the strict, lenient and corrected-refusal regrades. Only `bench.py` and JSON results were copied. No GGUF weights, caches, logs or `bench.py.bak` are in Git.

Each archive file below matched the phone original byte-for-byte after copying. `cmp` succeeded, sizes agree, and phone/archive SHA-256 values match.

| Archived artifact | Bytes | Phone SHA-256 | Archive SHA-256 | Match |
|---|---:|---|---|---|
| [`runs/bench.py`](runs/bench.py) | 26982 | `1e1b506c3276222b470125fa551d9afa692f98b9e5701ca351ce975f64c06080` | `1e1b506c3276222b470125fa551d9afa692f98b9e5701ca351ce975f64c06080` | YES |
| [`runs/bench_results_20260919_083347.json`](runs/bench_results_20260919_083347.json) — gemma-e2b, 81 rows | 109679 | `8a46ce308a54dd2eeac2340a89a04bb14ba886f5399092fef3c3b2558d3fda11` | `8a46ce308a54dd2eeac2340a89a04bb14ba886f5399092fef3c3b2558d3fda11` | YES |
| [`runs/bench_results_20260919_115350.json`](runs/bench_results_20260919_115350.json) — gemma-e4b, 81 rows | 111778 | `e228bf0e2b222b481a8c18ffc655f1abef9a321dc512d82cd8566aa2c5098a47` | `e228bf0e2b222b481a8c18ffc655f1abef9a321dc512d82cd8566aa2c5098a47` | YES |
| [`runs/bench_results_20260921_054508.json`](runs/bench_results_20260921_054508.json) — qwen3.5-2B-think-off, 27 rows | 36800 | `4c85fc58ead899b41c84f3327c7a46f90b140f4b564b9a71c0a2e36100df5358` | `4c85fc58ead899b41c84f3327c7a46f90b140f4b564b9a71c0a2e36100df5358` | YES |
| [`runs/bench_results_20260921_164938.json`](runs/bench_results_20260921_164938.json) — qwen3.5-2B-think-on, 81 rows | 235645 | `804cade9a772b9f2cb415c5843621edc70d8ecff37487f3a7e966081ea7ae6d6` | `804cade9a772b9f2cb415c5843621edc70d8ecff37487f3a7e966081ea7ae6d6` | YES |

No other `~/bench_results_*.json` files were present. Qwen2.5-3B and both Qwen3.5-4B modes therefore have no raw result artifact.

## Provenance boundary

The pre-patch `~/bench.py.bak` hash was verified on the phone as `dc53c4d15bdb9be69178d831f045d74d3dc6cf98a53d7c0b0ca7c2675996f442`; it is not archived. The copied final harness compiles and has the reported post-patch hash. However, none of the result JSONs records `source_sha256`. The September 19 Gemma results predate the final harness mtime; the September 21 Qwen results follow it. File times and successful handling of the former crash case support, but do not prove, grader-version association. Exact per-run source identity is **unverified**.

The original blind transcript/key and server logs were not specified as durable source artifacts. Their exact run mapping and hashes remain unverified. If they are archived later, preserve original bytes and add hashes without altering these raw JSONs.

## Recheck

From `benchmark/llm_objective_setting`:

```bash
stat -c '%s %n' runs/bench.py runs/bench_results_*.json
sha256sum runs/bench.py runs/bench_results_*.json
python -m py_compile runs/bench.py aggregate.py
python aggregate.py
```

`aggregate.py` reads the JSONs and writes only derived, ignored files in this directory. It does not run inference, contact a network, alter raw evidence or touch hardware.
