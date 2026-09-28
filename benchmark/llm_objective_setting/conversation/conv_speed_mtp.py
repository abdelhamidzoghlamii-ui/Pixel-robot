#!/usr/bin/env python3
"""MTP speculative-decoding speed runner. Research only: offline, no motors, no main.py.

Usage: conv_speed_mtp.py --out DIR [--models a,b] [--ids A6,C1] [--resume] [--thermal-log PATH]

conv_speed.py's cold block (thermal gate, page-cache drop through the handshake, cores 4-7, cpuset guard, redo on
lost cores, --resume) with these changes: temperature 0 for every prompt, only the 13 prompts in IDS, no cached
block, and per config extra server flags (MTP drafting). Qwen think is OFF as before.
Draft acceptance = sum(draft_n_accepted) / sum(draft_n) from the server's final timings. For each MTP-on config the
report counts replies byte-identical to its MTP-off twin (PAIRS). C-bucket replies are also written in bench.py's
result format to DIR/c_results.json for score_c.py.
Writes DIR/blocks.jsonl, DIR/turns.jsonl, DIR/report.txt, DIR/c_results.json.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import conv_speed as cs  # the archived runner: its prompts, server flags, streaming turn, stop, VmHWM, summary
from conv_speed import bench, ladder, sm
import aggregate  # noqa: E402  on conv_speed's sys.path: score_c.py's strict and lenient graders

M = cs.M
GEMMA_DRAFT = M + "mtp/mtp-gemma-4-E2B-it-Q8_0.gguf"
MTP = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
CONFIGS = {  # config -> (gguf, prompt family, think, extra flags)
    "gemma_e2b_q40": (M + "gemma-4-E2B-it-Q4_0.gguf", "gemma", None, []),
    "gemma_e2b_q40_mtp": (M + "gemma-4-E2B-it-Q4_0.gguf", "gemma", None, ["--model-draft", GEMMA_DRAFT, *MTP]),
    "qwen35_4b_q4km": (M + "qwen35/Qwen3.5-4B-Q4_K_M.gguf", "chatml", "OFF", []),
    "qwen35_4b_q40mtp": (M + "qwen35/mtp/Qwen3.5-4B-Q4_0-MTP.gguf", "chatml", "OFF", []),
    "qwen35_4b_q40mtp_on": (M + "qwen35/mtp/Qwen3.5-4B-Q4_0-MTP.gguf", "chatml", "OFF", MTP),
}
PAIRS = {"gemma_e2b_q40_mtp": "gemma_e2b_q40", "qwen35_4b_q40mtp_on": "qwen35_4b_q40mtp"}  # MTP on -> MTP off
IDS = "A5 A6 B1 B2 B6 B7 C1 C6 C7 C8 C9 C10 C11".split()
BY_ID = {p["id"]: p for p in bench.PROMPTS_A + bench.PROMPTS_B + bench.PROMPTS_C}


def server_cmd(name):
    """conv_speed.server_cmd (server_manager.py's binary and flags plus --cache-ram 0) with this config's flags."""
    gguf, _, _, extra = CONFIGS[name]
    return cs.server_cmd(gguf) + extra


def start(name, log, timeout=180):
    """conv_speed.start with server_cmd(name): cores 4-7; returns (process, seconds from spawn to /health ok)."""
    if cs.healthy(cs.PORT):
        raise SystemExit(f"port {cs.PORT} is already serving; stop that server first")
    began = time.perf_counter()
    proc = subprocess.Popen(["taskset", "-c", ladder.CORES, *server_cmd(name)], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    while not cs.healthy(cs.PORT):
        if proc.poll() is not None or time.perf_counter() - began > timeout:
            cs.stop(proc)
            raise RuntimeError(f"llama-server did not become healthy (exit {proc.poll()}); see the server log")
        time.sleep(0.1)
    return proc, time.perf_counter() - began


def prompts(family, think, ids):
    """conv_speed's prompts (bench.py wrappers, stop tokens, nonce), restricted to ids, at temperature 0. The nonce
    generator is seeded so an MTP-on config and its MTP-off twin get byte-identical prompts (the nonces still differ
    between prompts; every block starts a fresh server with --cache-ram 0 and cache_prompt false)."""
    random.seed(0)
    return [(pid, dict(body, temperature=0)) for pid, body in cs.prompts(family, think) if pid in ids]


def run_block(name, ids, out, thermal):
    """conv_speed.run_block's cold block for one config."""
    gguf, family, think, extra = CONFIGS[name]
    files = [gguf] + [GEMMA_DRAFT] * ("--model-draft" in extra)
    ladder.check_cores("before cold")
    therm = {}
    if thermal:
        therm["start"] = ladder.thermal_gate(thermal[0], thermal[1], f"{name} cold")
        print(f"  [{name}] thermal start: {ladder.fmt_thermal(therm['start'])} (waited {therm['start']['waited_s']} s)",
              flush=True)
    load_state = ladder.make_cold(name, files)
    ladder.check_cores("before starting cold")
    with open(out / "logs" / f"{name}.cold.server.log", "w") as log:
        proc, load_s = start(name, log)
        rows = []
        try:
            for pid_, body in prompts(family, think, ids):
                ttft, tm, stop_type, text = cs.turn(body)
                rows.append({"model": name, "block": "cold", "prompt_id": pid_, "ttft_s": ttft, "stop_type": stop_type,
                             "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                             "prompt_tok_s": tm.get("prompt_per_second"), "predicted_n": tm.get("predicted_n"),
                             "predicted_ms": tm.get("predicted_ms"), "gen_tok_s": tm.get("predicted_per_second"),
                             "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
                             "timings": tm, "reply": text})
                ladder.check_cores("during cold")
            peak = cs.vm_hwm_mib(proc.pid)
        finally:
            cs.stop(proc)
    if thermal:
        therm["end"] = ladder.read_thermal(thermal[0])
        print(f"  [{name}] thermal end:   {ladder.fmt_thermal(therm['end'])}", flush=True)
    ladder.check_cores("after cold")
    return {"model": name, "block": "cold", "load_s": load_s, "load_state": load_state, "peak_rss_mib": peak,
            "thermal": therm or None}, rows


def acceptance(rows):
    drafted = sum(r["draft_n"] or 0 for r in rows)
    return sum(r["draft_n_accepted"] or 0 for r in rows) / drafted if drafted else None


def report(done, models):
    f = lambda v, spec: format(v, spec) if v is not None else "n/a"
    t = lambda b, k: ladder.fmt_thermal(b["thermal"][k]).split(" degC")[0] if b.get("thermal") else "n/a"
    L = ["MTP SPEED REPORT  (server_manager flags + --cache-ram 0, cores 4-7, cold block, temperature 0)",
         "ttft = client time to first streamed token, median; prompt tok/s median; gen tok/s token-weighted (>= 10 tokens);",
         "accept = accepted / drafted tokens; same = replies byte-identical to the MTP-off twin", ""]
    L.append(f"{'model':<21}{'load s':>7}{'ttft s':>8}{'prompt t/s':>11}{'gen t/s':>9}{'peak MiB':>9}{'trunc':>6}{'accept':>8}"
             f"{'same':>8}   thermal start -> end (z9 z10 z11 degC); load state")
    for name in models:
        if (name, "cold") not in done:
            L.append(f"{name:<21}  not run")
            continue
        meta, rows = done[(name, "cold")]
        s = cs.summary(meta, rows)
        same = "-"
        if name in PAIRS and (PAIRS[name], "cold") in done:
            off = {r["prompt_id"]: r["reply"] for r in done[(PAIRS[name], "cold")][1]}
            same = f"{sum(off.get(r['prompt_id']) == r['reply'] for r in rows)}/{len(rows)}"
        L.append(f"{name:<21}{f(s['load_s'], '.1f'):>7}{f(s['ttft_s'], '.2f'):>8}{f(s['prompt_tok_s'], '.1f'):>11}"
                 f"{f(s['gen_tok_s'], '.2f'):>9}{s['peak_rss_mib']:>9}{s['truncated']:>6}{f(acceptance(rows), '.1%'):>8}"
                 f"{same:>8}   {t(meta, 'start')} -> {t(meta, 'end')}; {meta['load_state']['mode'].split(' (')[0]}")
    return "\n".join(L)


def c_results(done):
    """C-bucket rows in bench.py's result format (run_index 1), graded by bench.parse_and_grade_c, for score_c.py.
    The graders raise on some reply shapes (e.g. "object": null, a list as room); such a row gets score_c.py's "error"
    field, so score_c.py counts it under errors and scores the rest instead of crashing."""
    out = []
    for (name, _), (_, rows) in done.items():
        for r in rows:
            p = BY_ID[r["prompt_id"]]
            if r["prompt_id"][0] != "C":
                continue
            clean = bench.strip_think(r["reply"])
            row = {"config": name, "bucket": "C", "prompt_id": p["id"], "lang": p["lang"], "run_index": 1,
                   "source_sha256": bench.SOURCE_SHA256, "reply_raw": r["reply"], "reply_clean": clean,
                   "truncated": r["stop_type"] == "limit"}
            try:
                parse_ok, exact, crit, viol = bench.parse_and_grade_c(clean, p["expected"])
                for lenient in (False, True):  # the regimes score_c.py adds
                    aggregate.grade(r["reply"], p["id"], lenient=lenient)
                row.update(C_parse_ok=parse_ok, C_exact=exact, C_critical_failure=crit, C_reject_violation=viol)
            except Exception as e:  # any grader crash on model text
                row["error"] = f"grader raised {e!r}"
            out.append(row)
    return out


def read_jsonl(p):
    """Rows of a JSONL file; a torn last line (a write cut off by a kill) is dropped."""
    lines = p.read_text().splitlines() if p.exists() else []
    rows = [json.loads(l) for l in lines[:-1]]
    try:
        rows += [json.loads(l) for l in lines[-1:]]
    except json.JSONDecodeError:
        print(f"{p}: dropping torn last line", flush=True)
    return rows


def load_done(out):
    """conv_speed.load_done, tolerant of interrupted writes: blocks listed in blocks.jsonl (written only after the
    block passed) with their turns; turns of a block without a record (interrupted) are ignored."""
    turns = read_jsonl(out / "turns.jsonl")
    return {(m["model"], m["block"]): (m, [r for r in turns if (r["model"], r["block"]) == (m["model"], m["block"])])
            for m in read_jsonl(out / "blocks.jsonl")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default=",".join(CONFIGS))
    ap.add_argument("--ids", default=",".join(IDS), help="prompt ids (a subset of IDS, for toy runs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--thermal-log", default=None)
    a = ap.parse_args()
    models, ids = a.models.split(","), a.ids.split(",")
    if set(models) - set(CONFIGS):
        raise SystemExit(f"unknown models {sorted(set(models) - set(CONFIGS))}")
    if set(ids) - set(IDS):
        raise SystemExit(f"ids outside the 13 {sorted(set(ids) - set(IDS))}")
    if a.resume and not a.out.is_dir():
        raise SystemExit(f"--resume: {a.out} does not exist")
    a.out.mkdir(parents=True, exist_ok=a.resume)
    (a.out / "logs").mkdir(exist_ok=True)
    done = load_done(a.out) if a.resume else {}
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if a.resume:
        for (m, _), (_, rows) in done.items():
            if sorted(r["prompt_id"] for r in rows) != sorted(ids):
                raise SystemExit(f"--resume: {m}'s completed block has prompts {[r['prompt_id'] for r in rows]}, not {ids}")
        # rewrite both logs with only the completed blocks (no orphan turns, no torn line to append onto);
        # the old files are kept as *.before_resume_<stamp>, the new ones replace them atomically
        for fname, rows in (("blocks.jsonl", [m for m, _ in done.values()]),
                            ("turns.jsonl", [r for _, rs in done.values() for r in rs])):
            if (a.out / fname).exists():
                shutil.copy2(a.out / fname, a.out / f"{fname}.before_resume_{stamp}")
            (a.out / f"{fname}.tmp").write_text("".join(json.dumps(r) + "\n" for r in rows))
            os.replace(a.out / f"{fname}.tmp", a.out / fname)
        print("resume: keeping " + (", ".join(m for m, _ in done) or "nothing"), flush=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (a.out / f"run_{stamp}.json").write_text(json.dumps({
        "conv_speed_mtp_sha256": sha(__file__), "conv_speed_sha256": sha(cs.__file__), "bench_sha256": bench.SOURCE_SHA256,
        "ladder_sha256": sha(ladder.__file__), "ids": ids,
        "llama_server": sm.LLAMA_SERVER, "libggml_cpu_sha256": sha(Path(sm.LLAMA_SERVER).parent / "libggml-cpu.so"),
        "server_cmds": {m: server_cmd(m) for m in models}}, indent=1) + "\n")
    ladder.wait_cores("start")
    thermal = None
    if a.thermal_log:
        idle = ladder.read_thermal(a.thermal_log)
        print(f"thermal idle reading (gate z9 <= idle + {ladder.GATE_MC / 1000:.0f} degC): {ladder.fmt_thermal(idle)}", flush=True)
        thermal = (a.thermal_log, idle)
    with open(a.out / "turns.jsonl", "a") as tlog, open(a.out / "blocks.jsonl", "a") as blog:
        for name in models:
            if (name, "cold") in done:
                print(f"[{name}] cold: already completed, skipped (--resume)", flush=True)
                continue
            print(f"[{name}] cold: {len(ids)} turns", flush=True)
            while True:
                ladder.wait_cores(f"{name} cold")
                try:
                    meta, rows = run_block(name, ids, a.out, thermal)
                    break
                except ladder.CoresLost as e:
                    print(f"  [{name} cold] {e}: discarding this block's data and redoing it", flush=True)
            tlog.writelines(json.dumps(r) + "\n" for r in rows)
            tlog.flush()
            blog.write(json.dumps(meta) + "\n")
            blog.flush()
            done[(name, "cold")] = (meta, rows)
    text = report(done, models)
    (a.out / "report.txt").write_text(text + "\n")
    print(text)
    (a.out / "c_results.json").write_text(json.dumps(c_results(done), indent=1) + "\n")


if __name__ == "__main__":
    main()
