#!/usr/bin/env python3
"""Harness v3 runner: one candidate, one split, canonical and reversed option order.

Usually launched through measure.py (venv, offline env, cores 4-7, external RSS).
Scoring, eligible options, option order and two-stage logic are v2's unmodified
run_case(); every model call is logged with state, ordered options, instruction,
full distribution, argmax and ms. There are no warm-up calls: the first call is
reported as first_call_ms and excluded from the warm median/P95.
The held-out split is sealed: it is refused unless --unseal-heldout is given.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapters import CANDIDATES
from cases import load_v2

DEFAULT_FRAME = "filtered_text"
# Human decision: Python answers the rule families; the selector is judged on these only.
JUDGMENT_FAMILIES = ("room_finished", "hall_hint", "repeat_search", "heard_from_room")


class Recorder:
    """Wraps a candidate adapter in the interface v2's run_case() calls."""

    def __init__(self, adapter):
        self.adapter, self.name, self.log = adapter, adapter.name, []

    def decide(self, state, options, instruction):
        began = time.perf_counter()
        dist = self.adapter.decide(state, options, instruction)
        ms = (time.perf_counter() - began) * 1000
        if list(dist) != list(options) and set(dist) == set(options):
            dist = {k: dist[k] for k in options}
        if set(dist) != set(options):
            raise RuntimeError(f"distribution keys {sorted(dist)} != options {sorted(options)}")
        top = max(dist.values())
        choice = next(k for k in options if dist[k] == top)  # ties go to the first offered option
        self.log.append({"state": state, "options": list(options), "descriptions": dict(options),
                         "instruction": instruction, "distribution": dist, "argmax": choice,
                         "tie": sum(v == top for v in dist.values()) > 1, "ms": round(ms, 3),
                         "extra": dict(getattr(self.adapter, "extra", {}))})
        return {"choice": choice, "probabilities": dist, "confidence": None, "tokens": None,
                "elapsed_ms": ms}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(split, subset, families):
    manifest = json.loads((HERE / "CASES.json").read_text())
    info = manifest["splits"][split]
    path = HERE / info["file"]
    if sha(path) != info["sha256"]:
        raise SystemExit(f"{path.name} hash does not match CASES.json")
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    if families is not None:
        cases = [c for c in cases if c["family"] in families]
    return (cases[:subset] if subset else cases), info["sha256"]


def summarise(rows, calls, quantile):
    by_order = {}
    for order in ("normal", "reverse"):
        part = [r for r in rows if r["order"] == order]
        fam = {}
        for r in part:
            f = fam.setdefault(r["family"], {"n": 0, "preferred": 0, "acceptable": 0})
            f["n"] += 1
            f["preferred"] += r["preferred_match"]
            f["acceptable"] += r["acceptable_match"]
        by_order[order] = {"n": len(part), "preferred": sum(r["preferred_match"] for r in part),
                           "acceptable": sum(r["acceptable_match"] for r in part),
                           "invalid_or_ineligible": sum(r["invalid_or_ineligible"] for r in part),
                           "per_family": fam}
    normal = {r["case_id"]: r["choice"] for r in rows if r["order"] == "normal"}
    reverse = {r["case_id"]: r["choice"] for r in rows if r["order"] == "reverse"}
    warm = [c["ms"] for c in calls[1:]]
    return {**by_order, "order_flips": sum(normal[c] != reverse[c] for c in normal),
            "cases": len(normal), "calls": len(calls), "ties": sum(c["tie"] for c in calls),
            "first_call_ms": round(calls[0]["ms"], 1) if calls else None,
            "warm_p50_ms": round(statistics.median(warm), 1) if warm else None,
            "warm_p95_ms": round(quantile(warm, 0.95), 1) if warm else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    ap.add_argument("--split", choices=("dev", "heldout"), default="dev")
    ap.add_argument("--unseal-heldout", action="store_true", help="required to load the sealed held-out split")
    ap.add_argument("--frame", default=DEFAULT_FRAME, help="any v2 frame; default " + DEFAULT_FRAME)
    ap.add_argument("--dev-subset", type=int, default=0, metavar="N", help="first N dev cases (spans families)")
    ap.add_argument("--families", default="judgment",
                    help="'judgment' (default: %s), 'all', or a comma-separated list" % ",".join(JUDGMENT_FAMILIES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.split == "heldout" and not args.unseal_heldout:
        raise SystemExit("held-out split is sealed; pass --unseal-heldout only when authorised")
    if args.split == "heldout" and args.dev_subset:
        raise SystemExit("--dev-subset applies to the dev split only")
    v2 = load_v2()
    if args.frame not in v2.FRAMES:
        raise SystemExit(f"unknown frame {args.frame}; v2 frames: {v2.FRAMES}")
    if args.out.exists():
        raise FileExistsError(args.out)
    families = (JUDGMENT_FAMILIES if args.families == "judgment" else None if args.families == "all"
                else tuple(args.families.split(",")))
    if families is not None and set(families) - set(v2.FAMILIES):
        raise SystemExit(f"unknown families: {sorted(set(families) - set(v2.FAMILIES))}")
    cases, cases_sha = load_cases(args.split, args.dev_subset, families)

    began = time.perf_counter()
    adapter = CANDIDATES[args.candidate]()
    print(f"LOADED {args.candidate} load_ms={adapter.load_ms:.1f} "
          f"(construct {1000 * (time.perf_counter() - began):.1f})", flush=True)
    rec = Recorder(adapter)
    rows = []
    try:
        for case in cases:
            for order in ("normal", "reverse"):
                start = len(rec.log)
                row = v2.run_case(rec, case, args.frame, order)
                row["split"], row["calls"] = args.split, rec.log[start:]
                rows.append(row)
                print(f"{order[:3]} {case['id']} pref={row['preferred']} chose={row['choice']} "
                      f"ok={row['acceptable_match']} ms={row['total_ms']:.0f}", flush=True)
    finally:
        closed = adapter.close() if hasattr(adapter, "close") else None
    summary = summarise(rows, rec.log, v2.quantile)
    report = {"candidate": args.candidate, "doc": (adapter.__doc__ or "").strip(), "split": args.split,
              "frame": args.frame, "dev_subset": args.dev_subset, "families": families, "cases_sha256": cases_sha,
              "load_ms": round(adapter.load_ms, 1), "summary": summary, "close": closed,
              "harness_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))},
              "v2_sha256": sha(HERE.parent / "robot_selector_benchmark.py"),
              "python": sys.version, "rows": rows}
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in ("normal", "reverse")}))
    for order in ("normal", "reverse"):
        s = summary[order]
        print(f"{order.upper()} preferred={s['preferred']}/{s['n']} acceptable={s['acceptable']}/{s['n']}")


if __name__ == "__main__":
    main()
