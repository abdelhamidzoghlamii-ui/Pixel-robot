#!/usr/bin/env python3
"""Conversation speed runner. Research only: offline, no motors, no main.py.

Usage: conv_speed.py --out DIR [--models a,b] [--resume] [--thermal-log PATH]

Per model, two blocks, one server each, llama-server pinned to cores 4-7 with server_manager.py's binary and flags:
  cold    thermal gate, page-cache drop (ladder.make_cold: LADDER_COLD_HANDSHAKE=1 or a terminal prompt, else the
          weights are evicted), start the server, time spawn -> /health ok, then one pass over the #110 prompts.
  cached  thermal gate, start again with the weights in page cache, time the load, same pass.
Each turn streams /completion: time to first token is the first streamed content (client clock), prompt and
generation tok/s come from the server's timings. Peak RSS is the server's VmHWM. Blocks are gated, guarded and
resumed like ladder.py: losing cores 4-7 discards and redoes the block; --resume keeps completed blocks.
Writes DIR/blocks.jsonl, DIR/turns.jsonl, DIR/report.txt.
"""
import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/termux-home/ladder")
sys.path.insert(0, "/termux-home/robot")
sys.path.insert(0, "/termux-home/robot/benchmark/llm_objective_setting")
import ladder  # noqa: E402  thermal gate, core guard, cold handshake
import server_manager as sm  # noqa: E402  the robot's binary and flags
import bench  # noqa: E402  the #110 prompts, wrappers and sampling

M = "/data/data/com.termux/files/home/models/"
MODELS = {  # config -> (gguf, prompt family, think)
    "gemma_e2b_q4km": (M + "gemma-4-e2b-it-q4_k_m.gguf", "gemma", None),
    "gemma_e2b_q40": (M + "gemma-4-E2B-it-Q4_0.gguf", "gemma", None),
    "gemma_e4b_q4km": (M + "gemma-4-e4b-it-q4_k_m.gguf", "gemma", None),
    "qwen35_4b": (M + "qwen35/Qwen3.5-4B-Q4_K_M.gguf", "chatml", "OFF"),
    "qwen35_2b": (M + "qwen35/Qwen3.5-2B-Q4_K_M.gguf", "chatml", "OFF"),
}
PORT = 8080


def server_cmd(gguf, port=PORT):
    """server_manager.start_server's command line, as an argv list (no shell, no pkill), plus --cache-ram 0.
    DEVIATION (human decision 2026-09-26): with server_manager's flags alone the server's host prompt cache grew
    ~104 MiB anonymous memory per request on Qwen3.5-4B until Android killed it (~25 turns, Termux too); with
    --cache-ram 0 it stayed flat over 20 turns. server_manager.py's comment claims this flag but never passes it."""
    return [sm.LLAMA_SERVER, "-m", gguf, "--port", str(port), "--ctx-size", "2048", "--threads", sm.THREADS,
            "--threads-batch", sm.BATCH_THREADS, "--parallel", "1", "--swa-full", "--host", "127.0.0.1",
            "--cache-ram", "0"]


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def start(gguf, log, port=PORT, timeout=180):
    """Starts the server on cores 4-7; returns (process, load seconds from spawn to /health ok)."""
    if healthy(port):
        raise SystemExit(f"port {port} is already serving; stop that server first")
    began = time.perf_counter()
    proc = subprocess.Popen(["taskset", "-c", ladder.CORES, *server_cmd(gguf, port)], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    while not healthy(port):
        if proc.poll() is not None or time.perf_counter() - began > timeout:
            stop(proc)
            raise RuntimeError(f"llama-server did not become healthy (exit {proc.poll()})")
        time.sleep(0.1)
    return proc, time.perf_counter() - began


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def vm_hwm_mib(pid):
    return next(int(l.split()[1]) // 1024 for l in open(f"/proc/{pid}/status") if l.startswith("VmHWM:"))


def prompts(family, think):
    """The #110 A, B and C prompts, one pass, with bench.py's wrappers, stop tokens and temperatures."""
    out = []
    for bucket, ps, sys_msg, temp in (("A", bench.PROMPTS_A, bench.SYS_CHAT, 0.1), ("B", bench.PROMPTS_B, bench.SYS_CHAT, 0.1),
                                      ("C", bench.PROMPTS_C, bench.SYS_PARSE, 0.05)):
        for p in ps:
            text = [{"user": t["user"]} for t in p["text"]] if p.get("multi") else p["text"]  # A5: final turn, no history
            prompt, stop_tokens = bench.format_prompt(sys_msg, text, family, think)  # random nonce: no prefix reuse
            out.append((p["id"], {"prompt": prompt, "stop": stop_tokens, "temperature": temp, "n_predict": 512,
                                  "cache_prompt": False, "stream": True}))
    return out


def turn(body, port=PORT):
    """One streamed completion: (seconds to first streamed content, final timings, stop_type, text)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    began, first, text, final = time.perf_counter(), None, [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("content"):
                first = first or time.perf_counter() - began
                text.append(ev["content"])
            if ev.get("stop"):
                final = ev
    if final is None:
        raise RuntimeError("stream ended without a final event")
    return first, final.get("timings", {}), final.get("stop_type"), "".join(text)


def run_block(name, block, out, thermal):
    gguf, family, think = MODELS[name]
    ladder.check_cores("before " + block)
    therm = {}
    if thermal:
        therm["start"] = ladder.thermal_gate(thermal[0], thermal[1], f"{name} {block}")
        print(f"  [{name} {block}] thermal start: {ladder.fmt_thermal(therm['start'])} (waited {therm['start']['waited_s']} s)",
              flush=True)
    load_state = ladder.make_cold(name, [gguf]) if block == "cold" else {"mode": "cached"}
    ladder.check_cores("before starting " + block)
    with open(out / "logs" / f"{name}.{block}.server.log", "w") as log:
        proc, load_s = start(gguf, log)
        rows = []
        try:
            for pid_, body in prompts(family, think):
                ttft, tm, stop_type, text = turn(body)
                rows.append({"model": name, "block": block, "prompt_id": pid_, "ttft_s": ttft, "stop_type": stop_type,
                             "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                             "prompt_tok_s": tm.get("prompt_per_second"), "predicted_n": tm.get("predicted_n"),
                             "predicted_ms": tm.get("predicted_ms"), "gen_tok_s": tm.get("predicted_per_second"),
                             "reply": text})
                ladder.check_cores(f"during {block}")
            peak = vm_hwm_mib(proc.pid)
        finally:
            stop(proc)
    if thermal:
        therm["end"] = ladder.read_thermal(thermal[0])
        print(f"  [{name} {block}] thermal end:   {ladder.fmt_thermal(therm['end'])}", flush=True)
    ladder.check_cores("after " + block)
    return {"model": name, "block": block, "load_s": load_s, "load_state": load_state, "peak_rss_mib": peak,
            "thermal": therm or None}, rows


def summary(meta, rows):
    """Per block: medians over turns; generation tok/s is token-weighted over turns with >= 10 generated tokens."""
    gen = [r for r in rows if (r["predicted_n"] or 0) >= 10]
    ms = sum(r["predicted_ms"] for r in gen)
    med = lambda k: statistics.median([r[k] for r in rows if r[k] is not None]) if rows else None
    return {"load_s": meta["load_s"], "ttft_s": med("ttft_s"), "prompt_tok_s": med("prompt_tok_s"),
            "gen_tok_s": sum(r["predicted_n"] for r in gen) / ms * 1000 if ms else None,
            "peak_rss_mib": meta["peak_rss_mib"], "turns": len(rows), "truncated": sum(r["stop_type"] == "limit" for r in rows)}


def report(done, models):
    f = lambda v, spec: format(v, spec) if v is not None else "n/a"
    t = lambda b, k: ladder.fmt_thermal(b["thermal"][k]).split(" degC")[0] if b.get("thermal") else "n/a"
    L = ["CONVERSATION SPEED REPORT  (server_manager flags, cores 4-7, one pass over the 27 #110 prompts per block)",
         "ttft = client time to first streamed token; prompt tok/s median; gen tok/s token-weighted (>= 10 tokens)", ""]
    L.append(f"{'model':<16}{'block':<8}{'load s':>8}{'ttft s':>8}{'prompt t/s':>11}{'gen t/s':>9}{'peak MiB':>9}{'trunc':>6}"
             "   thermal start -> end (z9 z10 z11 degC); load state")
    for name in models:
        for block in ("cold", "cached"):
            if (name, block) not in done:
                L.append(f"{name:<16}{block:<8}  not run")
                continue
            meta, rows = done[(name, block)]
            s = summary(meta, rows)
            L.append(f"{name:<16}{block:<8}{f(s['load_s'], '.1f'):>8}{f(s['ttft_s'], '.2f'):>8}{f(s['prompt_tok_s'], '.1f'):>11}"
                     f"{f(s['gen_tok_s'], '.2f'):>9}{s['peak_rss_mib']:>9}{s['truncated']:>6}   {t(meta, 'start')} -> "
                     f"{t(meta, 'end')}; {meta['load_state']['mode'].split(' (')[0]}")
    return "\n".join(L)


def load_done(out):
    """Blocks that completed with valid data: listed in blocks.jsonl (written only after the block passed)."""
    turns = {}
    for line in (out / "turns.jsonl").read_text().splitlines() if (out / "turns.jsonl").exists() else []:
        r = json.loads(line)
        turns.setdefault((r["model"], r["block"]), []).append(r)
    metas = [json.loads(l) for l in (out / "blocks.jsonl").read_text().splitlines()] if (out / "blocks.jsonl").exists() else []
    return {(m["model"], m["block"]): (m, turns.get((m["model"], m["block"]), [])) for m in metas}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--thermal-log", default=None)
    a = ap.parse_args()
    models = a.models.split(",")
    if set(models) - set(MODELS):
        raise SystemExit(f"unknown models {sorted(set(models) - set(MODELS))}")
    if a.resume and not a.out.is_dir():
        raise SystemExit(f"--resume: {a.out} does not exist")
    a.out.mkdir(parents=True, exist_ok=a.resume)
    (a.out / "logs").mkdir(exist_ok=True)
    done = load_done(a.out) if a.resume else {}
    if a.resume:
        print("resume: keeping " + (", ".join(f"{m} {b}" for m, b in done) or "nothing"), flush=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    bin_dir = Path(sm.LLAMA_SERVER).parent
    (a.out / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json").write_text(json.dumps({
        "conv_speed_sha256": sha(__file__), "bench_sha256": bench.SOURCE_SHA256, "ladder_sha256": sha(ladder.__file__),
        "llama_server": sm.LLAMA_SERVER, "libggml_cpu_sha256": sha(bin_dir / "libggml-cpu.so"),
        "server_cmd": server_cmd("<gguf>"), "models": {m: MODELS[m] for m in models}}, indent=1) + "\n")
    ladder.wait_cores("start")
    thermal = None
    if a.thermal_log:
        idle = ladder.read_thermal(a.thermal_log)
        print(f"thermal idle reading (gate z9 <= idle + {ladder.GATE_MC / 1000:.0f} degC): {ladder.fmt_thermal(idle)}", flush=True)
        thermal = (a.thermal_log, idle)
    with open(a.out / "turns.jsonl", "a") as tlog, open(a.out / "blocks.jsonl", "a") as blog:
        for name in models:
            for block in ("cold", "cached"):
                if (name, block) in done:
                    print(f"[{name}] {block}: already completed, skipped (--resume)", flush=True)
                    continue
                print(f"[{name}] {block}: {len(prompts(*MODELS[name][1:]))} turns", flush=True)
                while True:
                    ladder.wait_cores(f"{name} {block}")
                    try:
                        meta, rows = run_block(name, block, a.out, thermal)
                        break
                    except ladder.CoresLost as e:
                        print(f"  [{name} {block}] {e}: discarding this block's data and redoing it", flush=True)
                tlog.writelines(json.dumps(r) + "\n" for r in rows)
                tlog.flush()
                blog.write(json.dumps(meta) + "\n")
                blog.flush()
                done[(name, block)] = (meta, rows)
    text = report(done, models)
    (a.out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
