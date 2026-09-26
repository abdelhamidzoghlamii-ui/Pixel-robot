#!/usr/bin/env python3
"""Bucket C (objective-setting) scores per model from a bench.py results file.

Usage: score_c.py RESULTS.json
Three parse regimes, as DECISIONS #110 reports them: the harness grader stored in each row by the rubric-fixed
bench.py (its extractor also accepts an array nested inside a bare object), aggregate.py strict (top-level array)
and aggregate.py lenient (a bare object counts as one action). All regimes use the same refusal rule.
"""
import json
import sys
from collections import Counter

sys.path.insert(0, "/termux-home/robot/benchmark/llm_objective_setting")
import aggregate  # noqa: E402

CROSS = {"C1", "C2", "C3", "C4", "C5", "C13"}


def main(path):
    rows = [r for r in json.load(open(path)) if r.get("bucket") == "C"]
    shas = Counter(r.get("source_sha256") for r in rows)
    print(f"{path}\nbench.py source_sha256 in rows: {dict(shas)}\n")
    print(f"{'model':<16}{'regime':<10}{'run-1 exact':>12}{'all-run exact':>15}{'parse':>8}{'crit':>6}{'reject viol':>12}"
          f"{'cross-ling r1':>14}{'errors':>8}{'trunc':>7}")
    fails = {}
    for model in sorted({r["config"] for r in rows}):
        mr = [r for r in rows if r["config"] == model]
        ok = [r for r in mr if "error" not in r]
        errors, trunc = len(mr) - len(ok), sum(bool(r.get("truncated")) for r in ok)
        regimes = {"harness": [dict(parse_ok=r["C_parse_ok"], exact=r["C_exact"], critical=r["C_critical_failure"],
                                    reject_violation=r["C_reject_violation"]) for r in ok],
                   "strict": [aggregate.grade(r["reply_raw"], r["prompt_id"]) for r in ok],
                   "lenient": [aggregate.grade(r["reply_raw"], r["prompt_id"], lenient=True) for r in ok]}
        for name, g in regimes.items():
            r1 = [x for x, r in zip(g, ok) if r["run_index"] == 1]
            cross = [x for x, r in zip(g, ok) if r["run_index"] == 1 and r["prompt_id"] in CROSS]
            frac = lambda xs: f"{sum(x['exact'] for x in xs)}/{len(xs)}"
            print(f"{model:<16}{name:<10}{frac(r1):>12}{frac(g):>15}{sum(x['parse_ok'] for x in g):>8}"
                  f"{sum(x['critical'] for x in g):>6}{sum(x['reject_violation'] for x in g):>12}{frac(cross):>14}{errors:>8}{trunc:>7}")
            if name == "harness":
                fails[model] = [(r["prompt_id"], r["run_index"], r["reply_clean"].strip()[:90]) for x, r in zip(g, ok) if not x["exact"]]
    print("\nHarness-grader misses (prompt, run, reply):")
    for model, fs in fails.items():
        print(f"  {model}: {len(fs)}")
        for pid, run, reply in fs:
            print(f"    {pid} run {run}: {reply!r}")


if __name__ == "__main__":
    main(sys.argv[1])
