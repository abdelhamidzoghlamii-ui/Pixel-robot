#!/usr/bin/env python3
"""Task C rerun (human decision 4): cold server start, then main.warm_up() (timed), then five voice commands through main.py's
own parse_command() (its 40 s request timeout). Run with native Termux Python, where main.py's
imports resolve. Starts the server via server_manager.start_setup('setup_q4') and stops it after."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROBOT = "/data/data/com.termux/files/home/robot"
sys.path.insert(0, ROBOT)
import main            # noqa: E402
import server_manager  # noqa: E402

COMMANDS = ["Find Chiara and tell her the pizza is here", "Go to the bathroom",
            "Find my phone in the living room", "Tell Abdel the meeting starts in ten minutes",
            "Go to the kitchen and say lunch is ready"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main_():
    out = {"main_py_sha256": sha(ROBOT + "/main.py"), "server_manager_sha256": sha(ROBOT + "/server_manager.py"),
           "script_sha256": sha(__file__), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.perf_counter()
    ok = server_manager.start_setup("setup_q4")
    out["server_ready"], out["start_s"] = ok, round(time.perf_counter() - t0, 1)
    t = time.perf_counter()
    main.warm_up()
    out["warm_up_s"] = round(time.perf_counter() - t, 2)
    print(json.dumps({"warm_up_s": out["warm_up_s"]}), flush=True)
    out["parses"] = []
    try:
        for i, cmd in enumerate(COMMANDS, 1):
            t = time.perf_counter()
            actions = main.parse_command(cmd)
            row = {"n": i, "command": cmd, "seconds": round(time.perf_counter() - t, 2), "actions": actions}
            out["parses"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        server_manager.start_setup("stop")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main_()
