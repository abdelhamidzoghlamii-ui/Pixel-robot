#!/usr/bin/env python3
"""Launch run.py for one candidate in its own venv, offline; external peak RSS.

Cores (default 4-7) and threads (default 4) come from V3_CORES / V3_THREADS.

Usage: measure.py OUTDIR CANDIDATE [run.py args...]
Writes OUTDIR/CANDIDATE.{json,stdout.txt,stderr.txt,block.json}. Peak RSS is
ru_maxrss from os.wait4 on the child: the largest single process among it and
the descendants it reaped (s1o's llama-server included).
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEL = "/termux-home/sel-candidates"
ENVS = {  # candidate -> (python, HF_HOME)
    "mock": (sys.executable, None),
    "von11": ("/termux-home/laya-test/venv/bin/python", "/termux-home/von-test/hf-cache"),
    "laya_en": (f"{SEL}/laya0320/venv/bin/python", f"{SEL}/laya0320/hf-cache"),
    "laya_multi": (f"{SEL}/laya-micro/venv/bin/python", f"{SEL}/laya-micro/hf-cache"),
    "laya_micro": (f"{SEL}/laya-micro/venv/bin/python", f"{SEL}/laya-micro/hf-cache"),
    "s1o": (f"{SEL}/s1o/venv/bin/python", None),
    "s1o_if": (f"{SEL}/s1o/venv/bin/python", None),
    "laya_en_onnx": (f"{SEL}/laya-micro/venv/bin/python", f"{SEL}/laya0320/hf-cache"),
    "von11_perm": ("/termux-home/laya-test/venv/bin/python", "/termux-home/von-test/hf-cache"),
    "von10_nli": ("/termux-home/laya-test/venv/bin/python", "/termux-home/von-test/hf-cache"),
}
CORES = os.environ.get("V3_CORES", "4-7")      # thread/core sweeps set these
THREADS = os.environ.get("V3_THREADS", "4")


def allowed_cpus():
    """This process's CPU list. Android moves Termux between cpusets (top-app 0-7,
    foreground 0-5), so a requested core set can silently shrink; record it and refuse."""
    line = next(l for l in open("/proc/self/status") if l.startswith("Cpus_allowed_list"))
    cpus = set()
    for part in line.split(":", 1)[1].strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def main():
    outdir, candidate, *rest = sys.argv[1:]
    lo, _, hi = CORES.partition("-")
    wanted = set(range(int(lo), int(hi or lo) + 1))
    if not wanted <= allowed_cpus():
        raise SystemExit(f"cores {CORES} not all allowed now (allowed {sorted(allowed_cpus())}); "
                         "bring Termux to the foreground (top-app cpuset) and retry")
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    python, hf_home = ENVS[candidate]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               USE_TF="0", OMP_NUM_THREADS=THREADS, V3_THREADS=THREADS, S1O_SERVER_LOG=str(outdir / f"{candidate}.server.log"))
    if hf_home:
        env.update(HF_HOME=hf_home, HF_HUB_CACHE=f"{hf_home}/hub")
    command = ["taskset", "-c", CORES, python, str(HERE / "run.py"), "--candidate", candidate,
               "--out", str(outdir / f"{candidate}.json"), *rest]
    record = {"candidate": candidate, "command": command, "hf_home": hf_home, "cores": CORES, "threads": THREADS,
              "env": {k: v for k, v in env.items() if k.startswith(("S1O_", "V3_"))},
              "cpus_allowed_start": sorted(allowed_cpus()), "start_utc": datetime.now(timezone.utc).isoformat()}
    with open(outdir / f"{candidate}.stdout.txt", "xb") as out, open(outdir / f"{candidate}.stderr.txt", "xb") as err:
        proc = subprocess.Popen(command, cwd=outdir, env=env, stdout=out, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)
    record.update(end_utc=datetime.now(timezone.utc).isoformat(), cpus_allowed_end=sorted(allowed_cpus()),
                  exit_code=os.waitstatus_to_exitcode(status),
                  peak_rss_mib_wait4=round(usage.ru_maxrss / 1024, 1))
    with open(outdir / f"{candidate}.block.json", "x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({k: record[k] for k in ("candidate", "exit_code", "peak_rss_mib_wait4")}))
    return record["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
