#!/usr/bin/env python3
"""Summarise memcheck runs: RssAnon/VmRSS growth per turn (least-squares slope over turns 1-N and
last-minus-first/(N-1)), plus median speed. Reads <run>/turns.jsonl and <run>/meta.json."""
import json
import statistics as st
import sys
from pathlib import Path


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


print("| run | shape | --cache-ram 0 | cpus | turns | RssAnon t0→t1→tN MiB | RssAnon growth/turn KiB (slope; (tN−t1)/(N−1)) | "
      "RssAnon max MiB | VmRSS t1→tN MiB | VmRSS slope KiB/turn | gen tok/s median | prompt ms median |")
print("|---|---|---|---|---:|---|---|---:|---|---:|---:|---:|")
for run in sys.argv[1:]:
    m = json.loads(Path(run, "meta.json").read_text())
    rows = [json.loads(l) for l in Path(run, "turns.jsonl").read_text().splitlines()]
    t = [r for r in rows if r["turn"] > 0]
    n = len(t)
    xs = [r["turn"] for r in t]
    ra = [r["RssAnon_kB"] for r in t]
    vm = [r["VmRSS_kB"] for r in t]
    mib = lambda k: f"{k / 1024:.1f}"
    print(f"| {run} | {m.get('shape', 'robot')} | {'yes' if m['cache_ram_0'] else 'no'} | {m['cpus_allowed']} | {n} | "
          f"{mib(rows[0]['RssAnon_kB'])}→{mib(ra[0])}→{mib(ra[-1])} | {slope(xs, ra):.1f}; {(ra[-1] - ra[0]) / (n - 1):.1f} | "
          f"{mib(max(ra))} | {mib(vm[0])}→{mib(vm[-1])} | {slope(xs, vm):.0f} | "
          f"{st.median(r['predicted_per_second'] for r in t):.2f} | {st.median(r['prompt_ms'] for r in t):.0f} |")
