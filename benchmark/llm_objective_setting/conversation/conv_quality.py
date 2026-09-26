#!/usr/bin/env python3
"""Quality run of the DECISIONS #110 benchmark (the rubric-fixed bench.py, unchanged otherwise) on five models,
with the robot's llama.cpp binary and server_manager.py's flags. Research only: offline, no motors, no main.py.

Usage: conv_quality.py OUT_DIR [--runs 3]
bench.do_run does the prompting and grading; this driver only swaps what bench.py hardcodes for its own phone
setup: the model table, the binary and server flags (conv_speed.server_cmd, pinned to cores 4-7), the output
directory, and stopping only the server it started (bench's pkill would also hit other llama-servers). Thermal
readings come from the root logger's thermal.log when it is fresh (bench's own `su` reading fails in proot).
Timings from this run do not count; run_conversation.sh measures speed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/termux-home/ladder")
import conv_speed as cs  # noqa: E402  model table, server command, start/stop
import ladder  # noqa: E402

bench = cs.bench
THERMAL_LOG = "/termux-home/ladder/thermal.log"
_proc = []


def start_server(config_name, log_file):
    try:
        with open(log_file, "a") as f:
            proc, _ = cs.start(bench.MODELS[config_name]["path"], f)
    except RuntimeError as e:
        print(f"Failed to start server for {config_name}: {e}")
        return None
    _proc.append(proc)
    return proc


def stop_server():
    while _proc:
        cs.stop(_proc.pop())


def zone9():
    try:
        return ladder.read_thermal(THERMAL_LOG)["z9"]
    except (SystemExit, OSError, IndexError, ValueError):  # no or stale logger: bench's "unknown" value
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    bench.MODELS = {name: {"path": gguf, "family": family, "think": think} for name, (gguf, family, think) in cs.MODELS.items()}
    bench.BIN_PATH = cs.sm.LLAMA_SERVER
    bench.HOME = str(a.out)
    bench.start_server, bench.stop_server, bench.get_zone9_temp = start_server, stop_server, zone9
    print(f"bench.py source_sha256 {bench.SOURCE_SHA256}; binary {bench.BIN_PATH}", flush=True)
    bench.do_run(argparse.Namespace(configs=list(bench.MODELS), runs=a.runs, dry_run=False))


if __name__ == "__main__":
    main()
