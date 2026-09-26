"""Checks for the runner's non-trivial logic: python3 test_ladder.py"""
import json
import os
import tempfile
from ladder import analyse, fit, hints, load_cases, p95

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
        4: {"median_ms": 290, "n": 3}}, "thread_choice_diffs": 0, "thread_max_prob_diff": 0, "thread_n": 6,
        "nondet_choice_diffs": 0, "nondet_max_prob_diff": 0, "nondet_n": 3, "cold": None,
        "cached": None, "median_ms": 20, "threads_main": 4,
        "repeat_worst": {"case_id": "c", "main_ms": 100, "repeat_ms": 110, "ratio": 1.1},
        "cases": 10, "flips": 3, "by_level": {1: {"correct": (10, 10)}, 5: {"correct": (2, 10)}},
        "by_phrasing": {"a": {"correct": (10, 10)}, "b": {"correct": (5, 10)}}, "truncated": 0, "limit": 512,
        "peak_mib": 100}
H = lambda **kw: "\n".join(hints({**base, **kw}, 8000))
h = H()
assert "per-token cost is 80% of predicted latency at level 5" in h          # 2*200 / (100 + 400)
assert "only 3% faster than 2, within the 15% noise floor" in h
assert "3/10 cases (30%)" in h and "from 100% at level 1" in h and "'a' 10/10 vs 'b' 5/10" in h
assert "model forward is 90%" in h
assert "per-token" not in H(fit={**base["fit"], "r2": 0.2})
# reviewer's scenario: the fastest is an intermediate thread count
mid = H(scaling={2: {"median_ms": 500, "n": 3}, 3: {"median_ms": 200, "n": 3}, 4: {"median_ms": 480, "n": 3}})
assert "3 threads is 60% faster than 2" in mid and "run at 3 threads" in mid
assert "no thread count beats 2 threads" in H(scaling={2: {"median_ms": 200, "n": 3}, 4: {"median_ms": 300, "n": 3}})
noisy = H(repeat_worst={"case_id": "c", "main_ms": 100, "repeat_ms": 350, "ratio": 3.5})
assert "no thread-count hint" in noisy and "faster than" not in noisy
# same-thread repeats vs other thread counts
assert "results depend on thread count" in H(thread_choice_diffs=2)
both = H(thread_choice_diffs=2, nondet_choice_diffs=1, nondet_max_prob_diff=0.3)
assert "run-to-run non-determinism" in both and "cannot separate" in both and "depend on thread count" not in both
# minimum sample size
few = H(cases=3, flips=1, by_level={1: {"correct": (2, 2)}, 5: {"correct": (0, 2)}},
        by_phrasing={"a": {"correct": (2, 2)}, "b": {"correct": (0, 2)}})
assert "order flips 1/3 cases: too few to judge" in few and "L1 2 decisions" in few and "falls from" not in few
assert "phrasing: a 2 decisions, b 2 decisions -> too few to judge" in few and "sensitive" not in few

# analyse(): a 0.0 ms reading must not crash the repeat ratio; same-thread diff counts as non-determinism
def row(block, thr, order, ms, choice):
    return {"case_id": "c1", "level": 1, "phrasing": "p", "order": order, "block": block, "threads": thr,
            "choice": choice, "correct": choice == "x", "acceptable": choice == "x", "total_ms": ms, "tokens": 50,
            "longest": 50, "limit": 512, "tokenise_ms": 0, "forward_ms": ms, "post_ms": 0,
            "distribution": {"x": 0.6, "y": 0.4} if choice == "x" else {"x": 0.4, "y": 0.6}}
def block(name, thr, rows):
    return {"block": name, "threads": thr, "rows": rows, "phases": {"file_read": 1, "import": 1, "init": 1, "warmup": 1},
            "spawn_to_ready_ms": 5, "file_bytes": 1, "load_state": {"mode": "cached"}, "peak_mib": 10,
            "wait4_maxrss_mib": 10, "timing_note": "", "token_def": "", "torch_threads": None}
m = analyse("laya_en", [block("main", 4, [row("main", 4, "written", 0.0, "x"), row("main", 4, "reversed", 5.0, "x")]),
                        block("t2", 2, [row("t2", 2, "written", 6.0, "x")]),
                        block("t4", 4, [row("t4", 4, "written", 4.0, "y")])], None)
assert m["repeat_worst"]["ratio"] > 1e6 and m["nondet_choice_diffs"] == 1 and m["thread_choice_diffs"] == 0
assert m["tok_per_s"] > 0
assert "per-token cost is 100%" in H(fit={**base["fit"], "fixed_ms": -50})

# thermal gate: parse the root log's last line, refuse a stale log, wait until z9 is back near idle
from datetime import datetime, timezone
import ladder
with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as t:
    t.write(f"zone9=BIG\n{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} z9=31000 z10=32000 z11=33000\n")
r = ladder.read_thermal(t.name)
assert (r["z9"], r["z10"], r["z11"]) == (31000, 32000, 33000)
with open(t.name, "w") as f:
    f.write("2026-01-01T00:00:00Z z9=31000 z10=32000 z11=33000\n")
try:
    ladder.read_thermal(t.name)
    raise AssertionError("stale thermal log accepted")
except SystemExit as e:
    assert "not running" in str(e)
readings = iter([{"at": "x", "z9": 35100}, {"at": "x", "z9": 35000}])  # idle 31: 35.1 waits, 35.0 passes
real_read, real_sleep = ladder.read_thermal, ladder.time.sleep
ladder.read_thermal, ladder.time.sleep = (lambda _: next(readings)), (lambda _: None)
assert ladder.thermal_gate("p", {"z9": 31000}, "x")["z9"] == 35000
readings = iter([{"at": "x", "z9": 27000}])  # cooler than idle starts at once (no abs)
assert ladder.thermal_gate("p", {"z9": 31000}, "x")["z9"] == 27000
ladder.read_thermal, ladder.time.sleep = real_read, real_sleep
os.unlink(t.name)

# Worker.close: a worker that ignores EOF is killed after the timeout and reaped
import subprocess
w = ladder.Worker.__new__(ladder.Worker)
w.proc = subprocess.Popen(["sleep", "60"], stdin=subprocess.PIPE, start_new_session=True)
began = ladder.time.monotonic()
assert w.close(timeout=0.5) is None and w.proc.returncode is not None and ladder.time.monotonic() - began < 5

# cold handshake: the watcher's answer is read, both files are removed; no answer times out
import threading
from pathlib import Path
d = Path(tempfile.mkdtemp())
ladder.DROP_REQUEST, ladder.DROP_DONE, ladder.HANDSHAKE_S = d / ".drop_request", d / ".drop_done", 3
def watcher(answer):
    while not ladder.DROP_REQUEST.exists():
        ladder.time.sleep(0.05)
    (d / "tmp").write_text(answer + "\n")
    (d / "tmp").rename(ladder.DROP_DONE)  # atomic, as the native-Termux watcher must do
for answer in ("ok", "failed"):
    threading.Thread(target=watcher, args=(answer,)).start()
    assert ladder.drop_handshake("x") == answer and not ladder.DROP_REQUEST.exists() and not ladder.DROP_DONE.exists()
ladder.HANDSHAKE_S = 0.2
assert ladder.drop_handshake("x") is None and not ladder.DROP_REQUEST.exists()

# losing cores raises CoresLost (the main loop waits and redoes the block)
real_cpus = ladder.allowed_cpus
ladder.allowed_cpus = lambda: {0, 1, 2, 3, 4, 5}
try:
    ladder.check_cores("during t3")
    raise AssertionError("lost cores not detected")
except ladder.CoresLost:
    pass
ladder.allowed_cpus = real_cpus

# --resume: main-only plan; new-format blocks need blocks.jsonl; legacy runs need full rows + a later block
cs = [{"id": "a"}, {"id": "b"}]
plan = ladder.block_plan(["m1", "m2"], [3, 4], {"m1"})
assert [b[0] for b in plan["m1"]] == ["main"] and [b[0] for b in plan["m2"]] == ["main", "t3", "t4"]
row = lambda m, b, t, c, o: json.dumps({"model": m, "block": b, "threads": t, "case_id": c, "order": o})
full = lambda m, b, t, orders: [row(m, b, t, c["id"], o) for c in cs for o in orders]
legacy = Path(tempfile.mkdtemp())
(legacy / "decisions.jsonl").write_text("\n".join(
    full("m1", "main", 4, ("written", "reversed")) + full("m1", "t3", 3, ("written",)) +
    full("m2", "main", 4, ("written", "reversed"))[:3] + full("m2", "t3", 3, ("written",))) + "\n")
done = ladder.load_done(legacy, plan, cs)  # m1 t3 unplanned, m2 main incomplete, m2 t3 last (unproven)
assert set(done) == {("m1", "main")} and done[("m1", "main")]["legacy"] and len(done[("m1", "main")]["rows"]) == 4
(legacy / "blocks.jsonl").write_text(json.dumps({"model": "m2", "block": "t3", "threads": 3}) + "\n")
assert set(ladder.load_done(legacy, plan, cs)) == {("m2", "t3")}  # new format: only blocks.jsonl entries count

# report copes with a main-only model whose main block is legacy (no load metadata)
lrows = [{"model": "m1", "block": "main", "threads": 4, "case_id": c, "order": o, "level": 1, "phrasing": "p",
          "choice": "x", "correct": True, "acceptable": True, "distribution": {"x": 1.0}, "total_ms": 100.0 + i,
          "tokenise_ms": 1.0, "forward_ms": 98.0, "post_ms": 1.0 + i, "tokens": 50 + i, "longest": 50 + i,
          "limit": 2048} for i, (c, o) in enumerate([("a", "written"), ("a", "reversed")])]
ladder.MODELS["m1"] = ladder.MODELS["s1o"]
m = ladder.analyse("m1", [{**ladder.LEGACY_META, "block": "main", "threads": 4, "rows": lrows}], None)
m["hints"] = ladder.hints(m, 8000)
text = ladder.report({"started": "t", "cases_file": "f", "cases_sha256": "0" * 64, "n_cases": 1, "levels": [1],
                      "models": ["m1"], "threads": [3, 4], "mem_total_mib": 8000, "per_model": [m]})
assert "main block only" in text and "cold load not recorded" in text and "cold load        n/a" in text
print("ok")
