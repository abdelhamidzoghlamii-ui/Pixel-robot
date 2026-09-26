"""Partial report of an interrupted ladder run, read-only: python3 partial.py OUT_DIR MODELS MAIN_ONLY"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/termux-home/ladder")
import ladder
out, models, main_only = Path(sys.argv[1]), sys.argv[2].split(","), set(filter(None, sys.argv[3].split(",")))
cases = ladder.load_cases("/termux-home/ladder/cases/ladder_cases_v1.jsonl", None)
plan = ladder.block_plan(models, [3, 4], main_only)
done = ladder.load_done(out, plan, cases)
print("completed valid blocks:", ", ".join(f"{m} {b}" for m, b in done))
per = []
for name in models:
    blocks = [done[(name, b)] for b, *_ in plan[name] if (name, b) in done]
    if any(b["block"] == "main" for b in blocks):
        m = ladder.analyse(name, blocks, cases)
        m["hints"] = ladder.hints(m, 7468)
        per.append(m)
print(ladder.report({"started": "partial report of " + out.name, "cases_file": "cases/ladder_cases_v1.jsonl",
                     "cases_sha256": "41eeafd28e2ce499", "n_cases": len(cases), "levels": [1, 2, 3, 4, 5],
                     "models": [m["model"] for m in per], "threads": [3, 4], "mem_total_mib": 7468, "per_model": per}))
