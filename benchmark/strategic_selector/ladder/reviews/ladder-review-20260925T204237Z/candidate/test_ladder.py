"""Checks for the runner's non-trivial logic: python3 test_ladder.py"""
import json
import tempfile
from ladder import fit, hints, load_cases, p95

f = fit([10, 20, 30, 40], [15, 25, 35, 45])
assert abs(f["fixed_ms"] - 5) < 1e-9 and abs(f["per_token_ms"] - 1) < 1e-9 and abs(f["r2"] - 1) < 1e-9
assert fit([10, 10], [1, 2]) is None
assert p95([1.0, 2.0]) == 1.95 and p95([7.0]) == 7.0

with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as t:
    t.write(json.dumps({"id": "a", "level": 2, "phrasing": "p", "situation": "s", "options": ["x", "y"], "answer": "z"}))
try:
    load_cases(t.name, None)
    raise AssertionError("answer outside options accepted")
except SystemExit as e:
    assert "not among options" in str(e)
assert len(load_cases("toy_cases.jsonl", {1, 5})) == 2

base = {"fit": {"fixed_ms": 100, "per_token_ms": 2, "r2": 0.9, "n": 6, "tokens_min": 20, "tokens_max": 200},
        "tokens_by_level": {1: 20, 5: 200}, "split_share": {"tokenise_ms": 0.05, "forward_ms": 0.9, "post_ms": 0.05},
        "split_ms": {"tokenise_ms": 1, "forward_ms": 18, "post_ms": 1}, "scaling": {2: {"median_ms": 300, "n": 3},
        4: {"median_ms": 290, "n": 3}}, "thread_choice_diffs": 0, "thread_max_prob_diff": 0, "cold": None,
        "cached": None, "median_ms": 20, "threads_main": 4,
        "repeat_worst": {"case_id": "c", "main_ms": 100, "repeat_ms": 110, "ratio": 1.1}, "cases": 10, "flips": 3, "by_level": {1: {"correct": (5, 5)},
        5: {"correct": (1, 5)}}, "by_phrasing": {}, "truncated": 0, "limit": 512, "peak_mib": 100}
h = "\n".join(hints(base, 8000))
assert "per-token cost is 80% of predicted latency at level 5" in h          # 2*200 / (100 + 400)
assert "differ by 3%, within the 15% noise floor" in h and "3/10 cases (30%)" in h and "from 100% at level 1" in h
assert "model forward is 90%" in h
assert "per-token" not in "\n".join(hints({**base, "fit": {**base["fit"], "r2": 0.2}}, 8000))
noisy = "\n".join(hints({**base, "repeat_worst": {"case_id": "c", "main_ms": 100, "repeat_ms": 350, "ratio": 3.5}}, 8000))
assert "no thread-count hint" in noisy and "faster than" not in noisy
fast4 = "\n".join(hints({**base, "scaling": {2: {"median_ms": 300, "n": 3}, 4: {"median_ms": 200, "n": 3}}}, 8000))
assert "4 threads is 33% faster than 2" in fast4
slow4 = "\n".join(hints({**base, "scaling": {2: {"median_ms": 200, "n": 3}, 4: {"median_ms": 300, "n": 3}}}, 8000))
assert "2 threads is 33% faster than 4" in slow4
print("ok")
