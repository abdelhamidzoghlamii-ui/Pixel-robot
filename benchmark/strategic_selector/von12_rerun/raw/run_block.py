#!/usr/bin/env python3
"""External block runner: offline env, battery + thermal-log readings, peak RSS.

Usage: run_block.py NAME -- COMMAND...
Writes NAME.stdout.txt, NAME.stderr.txt and NAME.block.json (exclusive create).
Peak RSS is ru_maxrss from os.wait4() on the child, i.e. the kernel's high-water
mark for the child and any descendant it reaped, measured outside the process.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF_HOME = HERE / "hf-cache"
# Root-only CPU zones are logged from native Termux (su) to this file; see README.
THERMAL_LOG = HERE / "thermal.log"
BATTERY = "/data/data/com.termux/files/usr/bin/termux-battery-status"


def utc():
    return datetime.now(timezone.utc).isoformat()


def reading():
    out = {"utc": utc()}
    try:
        out["battery"] = json.loads(subprocess.run([BATTERY], capture_output=True, text=True,
                                                   timeout=30, check=True).stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        out["battery"] = {"unavailable": type(exc).__name__}
    try:
        out["thermal_log_last_line"] = THERMAL_LOG.read_text().splitlines()[-1]
    except (OSError, IndexError) as exc:
        out["thermal_log_last_line"] = f"unavailable: {type(exc).__name__}"
    return out


def main():
    name, sep, *command = sys.argv[1:]
    assert sep == "--" and command, __doc__
    env = os.environ.copy()
    env.update(HF_HOME=str(HF_HOME), HF_HUB_CACHE=str(HF_HOME / "hub"),
               HUGGINGFACE_HUB_CACHE=str(HF_HOME / "hub"), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0",
               OMP_NUM_THREADS="4")
    record = {"name": name, "command": command, "cwd": str(HERE),
              "environment": {k: env[k] for k in ("HF_HOME", "HF_HUB_OFFLINE",
                              "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS")},
              "start": reading()}
    with open(HERE / f"{name}.stdout.txt", "xb") as out, open(HERE / f"{name}.stderr.txt", "xb") as err:
        proc = subprocess.Popen(command, cwd=HERE, env=env, stdout=out, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)
    record["end"] = reading()
    record["exit_code"] = os.waitstatus_to_exitcode(status)
    record["peak_rss_mib_wait4"] = round(usage.ru_maxrss / 1024, 1)
    with open(HERE / f"{name}.block.json", "x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({k: record[k] for k in ("name", "exit_code", "peak_rss_mib_wait4")}))
    return record["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
