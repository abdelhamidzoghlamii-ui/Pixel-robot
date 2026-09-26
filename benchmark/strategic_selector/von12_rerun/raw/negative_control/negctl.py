#!/usr/bin/env python3
"""Check that the unmodified v2 benchmark regenerates the archived Von v1.1 inputs.

Model-free. Imports robot_selector_benchmark.py by path (read-only) and rebuilds,
for every archived row, the state, offered options (in order), descriptions,
instructions and labels exactly as run_case() builds them. Exits 1 on any mismatch.
"""
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

SEL = Path("/termux-home/robot/benchmark/strategic_selector")
SRC = SEL / "robot_selector_benchmark.py"
ARCHIVED = Path('/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/perturbed.json')
FOLLOWUP = SEL / "results/2026-09-23T155449Z-von-filtered-followup/stdout.txt"
EXPECTED_SHA = "3aa9d399d6978710c143a6f1dfe69541b288cea329f27077b2500e322e8c14e2"

sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
print(f"BENCHMARK_SHA256={sha} MATCH={sha == EXPECTED_SHA}")
if sha != EXPECTED_SHA:
    sys.exit(1)
spec = importlib.util.spec_from_file_location("rsb", SRC)
rsb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rsb)

cases = {c["id"]: c for c in rsb.audit_fixtures("von")}
rows = json.loads(ARCHIVED.read_text(encoding="utf-8"))["rows"]
mismatches, checked = [], 0
for row in rows:
    case = cases[row["case_id"]]
    frame, order = row["frame"], row["order"]
    rng = random.Random(case["id"] + frame + order)
    stage = "mode" if frame.startswith("two_stage") else "flat"
    opts = rsb.render_options(rsb.options_for(case, frame, stage, rng), order, rng)
    got = {
        "family": case["family"], "preferred": case["preferred"],
        "acceptable": case["acceptable"],
        "state": rsb.to_state(case, frame.endswith("json")),
        "offered_first": list(opts),
        "offered_first_descriptions": list(opts.items()),
    }
    # Compare descriptions as ordered item lists so option order is checked too.
    want = dict(row, offered_first_descriptions=[list(x) for x in row["offered_first_descriptions"].items()])
    got["offered_first_descriptions"] = [list(x) for x in got["offered_first_descriptions"]]
    if row["offered_second_descriptions"] is not None:
        dest = rsb.render_options(rsb.options_for(case, frame, "destination", rng), order, rng)
        got["offered_second_descriptions"] = [list(x) for x in dest.items()]
        want["offered_second_descriptions"] = [list(x) for x in row["offered_second_descriptions"].items()]
    for key, value in got.items():
        checked += 1
        if want[key] != value:
            mismatches.append((row["case_id"], frame, order, key))

# The 1.1 filtered_text held-out follow-up saved stdout only: compare its labels.
followup = {}
for line in FOLLOWUP.read_text(encoding="utf-8").splitlines():
    if line.startswith("CASE="):
        fields = dict(p.split("=", 1) for p in line.split())
        followup[fields["CASE"]] = fields["EXPECTED"]
held = [c for c in cases.values() if c["id"].endswith("_2")]
followup_ok = followup == {c["id"]: c["preferred"] for c in held}

from collections import Counter
print("ARCHIVED_ROWS", len(rows), dict(Counter((r["split"], r["frame"]) for r in rows)))
print(f"FIELDS_CHECKED={checked} MISMATCHES={len(mismatches)}")
for m in mismatches[:20]:
    print("MISMATCH", m)
print(f"FOLLOWUP_STDOUT_EXPECTED_LABELS_MATCH={followup_ok} ({len(followup)} cases)")
print("NOTE: 1.1 filtered_text held-out has no archived JSON; its states/options cannot be compared, only labels.")
ok = not mismatches and followup_ok and len(rows) == 132
print(f"DETERMINISM={'IDENTICAL' if ok else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
