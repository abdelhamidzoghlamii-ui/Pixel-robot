#!/usr/bin/env python3
"""Run filtered_text and two_stage_text on Von held-out cases, normal + reversed.

Imports the unmodified v2 benchmark by path and reuses its Adapter, run_case and
aggregate, mirroring main(): same Adapter (4 intra-op / 1 inter-op threads), same
two warm-up calls, then held-out normal followed by reversed for each frame.
`--probe` instead loads Von and makes one warm-up decision (load-warning check).
"""
import hashlib
import importlib.util
import json
import random
import resource
import sys
from pathlib import Path

SRC = Path("/termux-home/robot/benchmark/strategic_selector/robot_selector_benchmark.py")
EXPECTED_SHA = "3aa9d399d6978710c143a6f1dfe69541b288cea329f27077b2500e322e8c14e2"
FRAMES = ("filtered_text", "two_stage_text")


def load_benchmark():
    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    print(f"BENCHMARK_SHA256={sha}", flush=True)
    if sha != EXPECTED_SHA:
        raise SystemExit("benchmark source differs from executed v2")
    spec = importlib.util.spec_from_file_location("rsb", SRC)
    rsb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rsb)
    return rsb


def main():
    probe = sys.argv[1:] == ["--probe"]
    out = Path("von12-probe.json" if probe else "von12-heldout-extra-results.json")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    rsb = load_benchmark()
    cases = rsb.audit_fixtures("von")
    dev = [c for c in cases if int(c["id"].rsplit("_", 1)[1]) < 2]
    held = [c for c in cases if int(c["id"].rsplit("_", 1)[1]) == 2]
    import von
    print(f"VON_SDK={von.__version__}", flush=True)
    adapter = rsb.Adapter("von")
    warm_state = rsb.to_state(dev[0], False)
    warm_options = rsb.options_for(dev[0], "filtered_text", "flat", random.Random(0))
    for n in range(1 if probe else 2):
        warm = adapter.decide(warm_state, warm_options,
                              "Which high-level script should the robot run next to find Chiara?")
        print(f"WARMUP_{n+1}_MS={warm['elapsed_ms']:.1f} CHOICE={warm['choice']}", flush=True)
    report = {"model": "von", "sdk": von.__version__, "first_call_ms": adapter.first_call_ms}
    if not probe:
        for frame in FRAMES:
            normal = [dict(rsb.run_case(adapter, c, frame), split="heldout") for c in held]
            reverse = [dict(rsb.run_case(adapter, c, frame, "reverse"), split="heldout_order_reversed")
                       for c in held]
            for a, b in zip(normal, reverse):
                print(f"{frame} CASE={a['case_id']} EXPECTED={a['preferred']} NORMAL={a['choice']} "
                      f"REVERSED={b['choice']} ACCEPTABLE={a['acceptable_match']} MS={a['total_ms']:.0f}",
                      flush=True)
            flips = sum(a["choice"] != b["choice"] for a, b in zip(normal, reverse))
            report[frame] = {"heldout": rsb.aggregate(normal), "reversed": rsb.aggregate(reverse),
                             "order_reversal_flips": flips, "n": len(held), "rows": normal + reverse}
            print(f"{frame} NORMAL_SUMMARY: {json.dumps(report[frame]['heldout'])}", flush=True)
            print(f"{frame} REVERSED_SUMMARY: {json.dumps(report[frame]['reversed'])}", flush=True)
            print(f"{frame} ORDER_REVERSAL_FLIPS: {flips} / {len(held)}", flush=True)
    report["inference_calls"] = adapter.calls
    report["max_rss_mib_self"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"INFERENCE_CALLS={adapter.calls} RESULT_FILE={out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
