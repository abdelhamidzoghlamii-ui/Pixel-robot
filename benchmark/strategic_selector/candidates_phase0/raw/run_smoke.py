#!/usr/bin/env python3
"""External smoke runner: offline env, peak RSS via os.wait4 on the child.

Usage: run_smoke.py DIR NAME -- COMMAND...
Runs COMMAND in DIR with HF_HOME=DIR/hf-cache and offline flags; writes
DIR/NAME.stdout.txt, NAME.stderr.txt and NAME.block.json (exclusive create).
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    folder, name, sep, *command = sys.argv[1:]
    assert sep == "--" and command, __doc__
    folder = Path(folder).resolve()
    env = os.environ.copy()
    hf = folder / "hf-cache"
    env.update(HF_HOME=str(hf), HF_HUB_CACHE=str(hf / "hub"), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0",
               OMP_NUM_THREADS="4")
    record = {"name": name, "command": command, "cwd": str(folder),
              "start_utc": datetime.now(timezone.utc).isoformat()}
    with open(folder / f"{name}.stdout.txt", "xb") as out, open(folder / f"{name}.stderr.txt", "xb") as err:
        proc = subprocess.Popen(command, cwd=folder, env=env, stdout=out, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)
    record["end_utc"] = datetime.now(timezone.utc).isoformat()
    record["exit_code"] = os.waitstatus_to_exitcode(status)
    record["peak_rss_mib_wait4"] = round(usage.ru_maxrss / 1024, 1)
    with open(folder / f"{name}.block.json", "x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({k: record[k] for k in ("name", "exit_code", "peak_rss_mib_wait4")}))
    return record["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
