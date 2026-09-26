#!/usr/bin/env python3
"""Markdown tables from one or more measure.py output directories, recomputed from rows.

Usage: report.py [--families judgment|all|a,b] DIR [DIR ...]
Default families are the four judgment families; counts, flips and warm stats are
recomputed from the logged rows, so a run over all families reports correctly too.
Warm stats exclude the run's first call. Each row of the table is one DIR/candidate.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import load_v2
from run import JUDGMENT_FAMILIES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="judgment")
    ap.add_argument("dirs", nargs="+", type=Path)
    args = ap.parse_args()
    v2 = load_v2()
    fams = (JUDGMENT_FAMILIES if args.families == "judgment" else v2.FAMILIES if args.families == "all"
            else tuple(args.families.split(",")))
    results = []
    for outdir in args.dirs:
        for path in sorted(outdir.glob("*.block.json")):
            block = json.loads(path.read_text())
            data = json.loads((outdir / f"{block['candidate']}.json").read_text())
            results.append((outdir, block, data))

    print(f"Families: {', '.join(fams)}\n")
    print("| Run | Candidate | Cores/threads | Cases | Acceptable | Preferred | Acc. rev | Pref. rev | Flips | "
          "Load ms | First-call ms | Warm p50 ms | Warm p95 ms | Peak RSS MiB |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for outdir, block, d in results:
        rows = [r for r in d["rows"] if r["family"] in fams]
        first = d["rows"][0]["calls"][0] if d["rows"] else None
        warm = [c["ms"] for r in rows for c in r["calls"] if c is not first]
        by = {o: [r for r in rows if r["order"] == o] for o in ("normal", "reverse")}
        normal = {r["case_id"]: r["choice"] for r in by["normal"]}
        reverse = {r["case_id"]: r["choice"] for r in by["reverse"]}
        flips = sum(normal[c] != reverse.get(c) for c in normal)
        n = len(normal)
        p95 = round(v2.quantile(warm, 0.95), 1) if warm else None
        print(f"| {outdir.name} | {d['candidate']} | {block.get('cores', '4-7')}/{block.get('threads', '4')} | {n} | "
              f"{sum(r['acceptable_match'] for r in by['normal'])} | {sum(r['preferred_match'] for r in by['normal'])} | "
              f"{sum(r['acceptable_match'] for r in by['reverse'])} | {sum(r['preferred_match'] for r in by['reverse'])} | "
              f"{flips}/{n} | {d['load_ms']} | {d['summary']['first_call_ms']} | "
              f"{round(statistics.median(warm), 1) if warm else None} | {p95} | {block['peak_rss_mib_wait4']} |")

    print("\nPer family, canonical order. Each cell: acceptable y/n · preferred x/n.\n")
    print("| Family | " + " | ".join(f"{o.name}/{d['candidate']}" for o, _, d in results) + " |")
    print("|---|" + "---:|" * len(results))
    for fam in fams:
        cells = []
        for _, _, d in results:
            rows = [r for r in d["rows"] if r["family"] == fam and r["order"] == "normal"]
            cells.append(f"A {sum(r['acceptable_match'] for r in rows)}/{len(rows)} · "
                         f"P {sum(r['preferred_match'] for r in rows)}/{len(rows)}")
        print(f"| {fam} | " + " | ".join(cells) + " |")

    for outdir, block, d in results:
        calls = [c for r in d["rows"] if r["family"] in fams for c in r["calls"]]
        notes = []
        for key, label in (("tokenizer_parity_vs_stock", "tokenizer parity vs stock"),):
            vals = [c["extra"][key] for c in calls if key in c["extra"]]
            if vals:
                notes.append(f"{label} {sum(vals)}/{len(vals)} calls")
        native = [c["argmax"] == c["extra"]["native_choice"] for c in calls if "native_choice" in c["extra"]]
        if native:
            notes.append(f"argmax == native choice {sum(native)}/{len(native)} calls")
        mass = [c["extra"]["letter_mass_full_vocab"] for c in calls if "letter_mass_full_vocab" in c["extra"]]
        if mass:
            notes.append(f"letter mass in full vocab min {min(mass):.4f} median {sorted(mass)[len(mass) // 2]:.4f}")
        toks = [c["extra"]["prompt_tokens"] for c in calls if "prompt_tokens" in c["extra"]]
        if toks:
            notes.append(f"prompt tokens median {sorted(toks)[len(toks) // 2]} (min {min(toks)}, max {max(toks)})")
        rots = [c["extra"]["rotations"] for c in calls if "rotations" in c["extra"]]
        if rots:
            chk = [c["extra"]["batched_vs_sequential_max_abs_diff"] for c in calls
                   if "batched_vs_sequential_max_abs_diff" in c["extra"]]
            notes.append(f"rotations per decision {min(rots)}-{max(rots)}; batched==sequential check on "
                         f"{len(chk)} decisions, max |diff| {max(chk) if chk else 'n/a'}")
        if d.get("close"):
            notes.append(f"server {d['close']}")
        print(f"\n- {outdir.name}/{d['candidate']}: exit {block['exit_code']}; " + "; ".join(notes))


if __name__ == "__main__":
    main()
