#!/usr/bin/env python3
"""Markdown tables from a measure.py output directory: overall, per family, parity."""
import json
import sys
from pathlib import Path


def main():
    outdir = Path(sys.argv[1])
    results = []
    for path in sorted(outdir.glob("*.block.json")):
        block = json.loads(path.read_text())
        data = json.loads((outdir / f"{block['candidate']}.json").read_text())
        results.append((block, data))
    print("| Candidate | Cases | Preferred | Acceptable | Pref. rev | Acc. rev | Invalid | Order flips | Ties | "
          "Load ms | First-call ms | Warm p50 ms | Warm p95 ms | Peak RSS MiB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for block, d in results:
        s, n, r = d["summary"], d["summary"]["normal"], d["summary"]["reverse"]
        print(f"| {d['candidate']} | {s['cases']} | {n['preferred']} | {n['acceptable']} | {r['preferred']} | "
              f"{r['acceptable']} | {n['invalid_or_ineligible'] + r['invalid_or_ineligible']} | "
              f"{s['order_flips']}/{s['cases']} | {s['ties']} | {d['load_ms']} | {s['first_call_ms']} | "
              f"{s['warm_p50_ms']} | {s['warm_p95_ms']} | {block['peak_rss_mib_wait4']} |")
    families = list(results[0][1]["summary"]["normal"]["per_family"]) if results else []
    print("\nPer family, canonical order. Each cell: preferred x/n · acceptable y/n.\n")
    head = [d["candidate"] for _, d in results]
    print("| Family | " + " | ".join(head) + " |")
    print("|---|" + "---:|" * len(head))
    for fam in families:
        cells = []
        for _, d in results:
            f = d["summary"]["normal"]["per_family"][fam]
            cells.append(f"P {f['preferred']}/{f['n']} · A {f['acceptable']}/{f['n']}")
        print(f"| {fam} | " + " | ".join(cells) + " |")
    for block, d in results:
        calls = [c for row in d["rows"] for c in row["calls"]]
        parity = [c["extra"]["tokenizer_parity_vs_stock"] for c in calls if "tokenizer_parity_vs_stock" in c["extra"]]
        native = [c["argmax"] == c["extra"]["native_choice"] for c in calls if "native_choice" in c["extra"]]
        mass = [c["extra"]["letter_mass_full_vocab"] for c in calls if "letter_mass_full_vocab" in c["extra"]]
        notes = []
        if parity:
            notes.append(f"tokenizer parity vs stock {sum(parity)}/{len(parity)} calls")
        if native:
            notes.append(f"argmax == native choice {sum(native)}/{len(native)} calls")
        if mass:
            notes.append(f"A-letter mass in full vocab min {min(mass):.4f} median {sorted(mass)[len(mass) // 2]:.4f}")
        if d.get("close"):
            notes.append(f"server {d['close']}")
        print(f"\n- {d['candidate']}: exit {block['exit_code']}; " + "; ".join(notes))


if __name__ == "__main__":
    main()
