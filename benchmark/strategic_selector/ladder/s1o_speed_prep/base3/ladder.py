#!/usr/bin/env python3
"""Ladder benchmark runner. Research only: offline, no motors, no main.py.

Usage: ladder.py CASES.jsonl --out DIR [--models a,b] [--levels 1,2] [--threads 2,3,4]

CASES.jsonl, one case per line:
  {"id": str, "level": 1-5, "phrasing": str, "situation": str, "options": [str, ...] or {key: desc},
   "answer": str, "acceptable": [str, ...] (optional), "instruction": str (optional)}

Per model (models run one after another, one worker at a time, pinned to cores 4-7):
  block "main":  cold load (the runner pauses for the human to drop the page cache from native Termux
                 and checks /proc/meminfo; else the model's weight files are evicted with posix_fadvise), then every case in written order and reversed, at the
                 highest --threads value. Accuracy, flips, latency fit and time split come from here.
  block "t<N>":  cached load at N threads, then every case in written order once. Thread scaling
                 compares these blocks only (same work, same load state); t<max> is the cached load.
Writes DIR/decisions.jsonl (every decision), DIR/results.json, DIR/report.txt; prints the report.
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
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
JEVLIKE = Path("/termux-home/robot/benchmark/strategic_selector/manual/jevlike")
sys.path.insert(0, str(JEVLIKE))
from jevlike import LABEL, MODELS, allowed_cpus  # noqa: E402  (menu name -> adapter, python, HF_HOME, status)

CORES = "4-7"
MIN_N = 10  # level/phrasing hints need this many decisions per group; order flips this many cases
DEFAULT_INSTRUCTION = "Choose the single best next action for the robot."
SEL = Path("/termux-home/sel-candidates")
LAYA_SNAP = "hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
VON11_SNAP = Path("/termux-home/von-test/hf-cache/hub/models--wfzyx--von/snapshots/d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd")
VON12_SNAP = Path("/termux-home/von12-test/hf-cache/hub/models--wfzyx--von/snapshots/5df8185a4f2327ad0a7cd117cc4f701ac557b9ae")
# Weight/tokenizer files each model loads (read in the file_read phase, evicted for the cold load).
WEIGHTS = {
    "laya_en": [SEL / "laya_en_onnx/laya_en.onnx", SEL / "laya_en_onnx/laya_en.onnx.data", SEL / "laya_en_onnx/ckpt"],
    "von11": [VON11_SNAP],  # option_marker.pt on top of the base model.safetensors: both are loaded
    "s1o": [Path("/termux-home/models/gemma-4-e2b-it-q4_k_m.gguf")],
    "laya_multi": [SEL / "laya-micro/hf-cache" / LAYA_SNAP / "multilingual"],
    "laya_micro": [SEL / "laya-micro/laya-micro-8192.int8.onnx", SEL / "laya-micro/laya-micro-8192.int8.onnx.data",
                   SEL / "laya-micro/laya-multilingual-en-8192"],
    "von12": [VON12_SNAP],
    "von10_nli": [VON11_SNAP / f for f in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")],
}
# s1o speed variants: same S1O adapter and letter scoring, other llama.cpp build / GGUF. name -> (env, gguf)
B1609, B2351 = "/termux-home/llama.cpp/build/bin", "/termux-home/llama.cpp-upstream/build/bin"
# b2351 keeps a host-RAM prompt cache and SWA checkpoints by default; off, so no decision reuses another's prompt.
# -b/-ub 512 are the defaults, pinned: the longest ladder prompt (173 tokens) is evaluated as one batch.
# Flash attention off: b2351's CPU flash attention segfaults on Gemma 4 prompts of >= 64 tokens (b1609 runs it on).
B2351_ARGS = "--flash-attn off --cache-ram 0 --ctx-checkpoints 0 --batch-size 512 --ubatch-size 512"
GGUFS = {"q4km": "/termux-home/models/gemma-4-e2b-it-q4_k_m.gguf", "q40": "/termux-home/models/gemma-4-E2B-it-Q4_0.gguf",
         "qwen2b": "/termux-home/models/qwen35/Qwen3.5-2B-Q4_K_M.gguf",
         "qwen08b": "/termux-home/models/qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"}
VARIANTS = {"s1o_b1609": (B1609, "q4km", ""), "s1o_b2351": (B2351, "q4km", B2351_ARGS),
            "s1o_b2351_q40": (B2351, "q40", B2351_ARGS), "s1o_b2351_qwen2b": (B2351, "qwen2b", B2351_ARGS),
            "s1o_b2351_qwen08b": (B2351, "qwen08b", B2351_ARGS)}
VARIANT_ENV = {n: {"S1O_LLAMA_BIN": b, "S1O_GGUF": GGUFS[g], "S1O_SERVER_ARGS": a} for n, (b, g, a) in VARIANTS.items()}
MODELS = {**MODELS, **{n: MODELS["s1o"] for n in VARIANTS}}
WEIGHTS.update({n: [Path(GGUFS[g])] for n, (_, g, _) in VARIANTS.items()})


def files_of(paths):
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.is_file())
        elif p.exists():
            out.append(p)
    return sorted({str(q.resolve()) for q in out})  # snapshots are symlinks into blobs/


# ---------------------------------------------------------------- cases

def load_cases(path, levels):
    cases, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        c = json.loads(line)
        where = f"{path}:{n}"
        for key in ("id", "level", "phrasing", "situation", "options", "answer"):
            if key not in c:
                raise SystemExit(f"{where}: missing {key!r}")
        opts = c["options"] if isinstance(c["options"], dict) else {o: "" for o in c["options"]}
        if len(opts) < 2 or c["level"] not in range(1, 6) or c["id"] in seen:
            raise SystemExit(f"{where}: need >=2 options, level 1-5 and a unique id")
        acceptable = set(c.get("acceptable", [])) | {c["answer"]}
        if not acceptable <= set(opts):
            raise SystemExit(f"{where}: answer/acceptable not among options: {sorted(acceptable - set(opts))}")
        seen.add(c["id"])
        cases.append({"id": c["id"], "level": c["level"], "phrasing": c["phrasing"], "situation": c["situation"],
                      "options": opts, "answer": c["answer"], "acceptable": sorted(acceptable),
                      "instruction": c.get("instruction", DEFAULT_INSTRUCTION)})
    return [c for c in cases if levels is None or c["level"] in levels]


# ---------------------------------------------------------------- workers

def check_cores(when):
    if not set(range(4, 8)) <= allowed_cpus():
        raise SystemExit(f"cores {CORES} not all allowed {when} (allowed {sorted(allowed_cpus())}); "
                         "bring Termux to the foreground (top-app cpuset) and retry")


DROP_CMD = "su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"


def cached_mib():
    return next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("Cached:"))


def evict(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # read-only files: every page is clean
        finally:
            os.close(fd)


def make_cold(name, files):
    """Ask the human to drop the page cache from native Termux (Magisk su does not work in proot) and check
    /proc/meminfo that it happened; otherwise evict only this model's weight files."""
    before = cached_mib()
    if not sys.stdin.isatty():
        why = "no terminal to prompt on"
    else:
        print(f"\n[{name}] COLD LOAD NEXT. In native Termux (not proot) run:\n    {DROP_CMD}\n"
              "then press Enter here. Type s + Enter to skip (weights-cold fallback).", flush=True)
        answer = input("> ").strip().lower()
        after = cached_mib()
        if answer != "s" and after < 0.5 * before:
            return {"mode": "full-cold (page cache dropped by the human with drop_caches)",
                    "cached_mib_before": before, "cached_mib_after": after}
        why = "skipped by the human" if answer == "s" else f"Enter pressed but page cache did not drop ({before} -> {after} MiB)"
    evict(files)
    return {"mode": "weights-cold (posix_fadvise DONTNEED on the weight files; libraries stay cached)",
            "fallback_reason": why, "cached_mib_before": before}


class Worker:
    def __init__(self, name, threads, files, logdir, tag):
        adapter, python, hf, _ = MODELS[name]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   USE_TF="0", OMP_NUM_THREADS=str(threads), V3_THREADS=str(threads),
                   LADDER_FILES=json.dumps(files), S1O_SERVER_LOG=str(logdir / f"{name}.{tag}.server.log"),
                   **VARIANT_ENV.get(name, {}))
        if hf:
            env.update(HF_HOME=hf, HF_HUB_CACHE=f"{hf}/hub")
        (logdir / f"{name}.{tag}.server.log").unlink(missing_ok=True)  # s1o opens it exclusive-create
        began = time.perf_counter()
        self.proc = subprocess.Popen(["taskset", "-c", CORES, python, str(HERE / "ladder_worker.py"), adapter],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
                                     stderr=open(logdir / f"{name}.{tag}.stderr.txt", "w"), start_new_session=True)
        try:
            self.ready = self.ask(None)
        except BaseException:
            self.close()
            raise
        self.spawn_to_ready_ms = (time.perf_counter() - began) * 1000

    def ask(self, req):
        if req is not None:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited; see its stderr log")
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def close(self, timeout=30):
        """EOF -> adapter.close(); returns ru_maxrss (MiB) of the largest process in the worker's tree, or None
        if the worker did not exit within timeout seconds (then its whole process group is killed)."""
        maxrss = None
        try:
            self.proc.stdin.close()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                pid, _, usage = os.wait4(self.proc.pid, os.WNOHANG)
                if pid:
                    maxrss = round(usage.ru_maxrss / 1024, 1)
                    self.proc.returncode = 0
                    break
                time.sleep(0.2)
            else:
                print(f"worker {self.proc.pid} still running {timeout} s after EOF; killing its process group",
                      file=sys.stderr, flush=True)
        finally:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)  # backstop: llama-server or anything left
            except ProcessLookupError:
                pass
            if self.proc.returncode is None:
                self.proc.wait()  # reap the killed worker
        return maxrss


THERMAL_MAX_AGE_S = 30  # the root logger writes every ~5 s; an older last line means it stopped
GATE_MC = 2000         # a block starts only when z9 <= idle reading + 2 degC


def read_thermal(path):
    """Last line of the root thermal log: '2026-09-25T21:51:27Z z9=36000 z10=36000 z11=37000' (millidegrees)."""
    last = Path(path).read_text().strip().splitlines()[-1].split()
    stamp = datetime.strptime(last[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if age > THERMAL_MAX_AGE_S:
        raise SystemExit(f"thermal log {path} last line is {age:.0f} s old: the logger is not running")
    z = dict(kv.split("=") for kv in last[1:])
    return {"at": last[0], **{k: int(z[k]) for k in ("z9", "z10", "z11")}}


def thermal_gate(path, idle, label):
    """Waits until z9 <= idle + GATE_MC (cooler than idle is fine); returns the reading the block starts at."""
    began = time.monotonic()
    while (t := read_thermal(path))["z9"] > idle["z9"] + GATE_MC:
        print(f"  [{label}] waiting: z9 {t['z9'] / 1000:.1f} degC, need <= {(idle['z9'] + GATE_MC) / 1000:.1f} "
              f"(idle {idle['z9'] / 1000:.1f} + {GATE_MC / 1000:.0f}), {time.monotonic() - began:.0f} s", flush=True)
        time.sleep(10)
    t["waited_s"] = round(time.monotonic() - began)
    return t


def fmt_thermal(t):
    return " ".join(f"{k} {t[k] / 1000:.1f}" for k in ("z9", "z10", "z11")) + f" degC at {t['at']}"


def run_block(name, block, threads, cases, orders, files, out, log, cold, thermal=None):
    check_cores("before " + block)
    therm = None
    if thermal:  # gate before the cache drop, so the block starts right after the drop
        path, idle = thermal
        therm = {"start": thermal_gate(path, idle, f"{name} {block}")}
        print(f"  [{name} {block}] thermal start: {fmt_thermal(therm['start'])} (waited {therm['start']['waited_s']} s)",
              flush=True)
    load_state = make_cold(name, files) if cold else {"mode": "cached"}
    w = Worker(name, threads, files, out / "logs", block)
    rows = []
    try:
        for case in cases:
            for order in orders:
                keys = list(case["options"]) if order == "written" else list(case["options"])[::-1]
                r = w.ask({"state": case["situation"], "instruction": case["instruction"],
                           "options": {k: case["options"][k] for k in keys}})
                top = max(r["dist"].values())
                choice = next(k for k in keys if r["dist"][k] == top)  # ties: first offered, as in v3 run.py
                row = {"model": name, "case_id": case["id"], "level": case["level"], "phrasing": case["phrasing"],
                       "order": order, "block": block, "threads": threads, "options": keys,
                       "choice": choice, "tie": list(r["dist"].values()).count(top) > 1,
                       "correct": choice == case["answer"], "acceptable": choice in case["acceptable"],
                       "distribution": r.pop("dist"), **r}
                log.write(json.dumps(row) + "\n")
                rows.append(row)
        mem = w.ask({"cmd": "mem"})
    finally:
        maxrss = w.close()
    if thermal:
        therm["end"] = read_thermal(thermal[0])
        print(f"  [{name} {block}] thermal end:   {fmt_thermal(therm['end'])}", flush=True)
    check_cores("after " + block)
    return {"block": block, "threads": threads, "load_state": load_state, "thermal": therm, "phases": w.ready["phases"],
            "file_bytes": w.ready["file_bytes"], "spawn_to_ready_ms": w.spawn_to_ready_ms,
            "timing_note": w.ready["timing_note"], "token_def": w.ready["token_def"], "torch_threads": w.ready["torch_threads"],
            "rss_mib": mem["rss_mib"], "peak_mib": mem["peak_mib"], "wait4_maxrss_mib": maxrss, "rows": rows}


# ---------------------------------------------------------------- statistics

def median(xs):
    return statistics.median(xs) if xs else None


def p95(xs):
    return statistics.quantiles(xs, n=100, method="inclusive")[94] if len(xs) >= 2 else (xs[0] if xs else None)


def fit(xs, ys):
    """Least squares ms = fixed + per_token * tokens. None when the tokens have no spread."""
    if len(set(xs)) < 2:
        return None
    slope, intercept = statistics.linear_regression(xs, ys)
    r2 = statistics.correlation(xs, ys) ** 2 if len(set(ys)) > 1 else 0.0
    return {"fixed_ms": intercept, "per_token_ms": slope, "r2": r2, "n": len(xs), "tokens_min": min(xs),
            "tokens_max": max(xs)}


def rate(rows, key):
    return (sum(r[key] for r in rows), len(rows))


def analyse(name, blocks, cases):
    main = next(b for b in blocks if b["block"] == "main")
    rows = main["rows"]
    by = lambda field: {v: {"correct": rate([r for r in rows if r[field] == v], "correct"),
                            "acceptable": rate([r for r in rows if r[field] == v], "acceptable")}
                        for v in sorted({r[field] for r in rows})}
    written = {r["case_id"]: r["choice"] for r in rows if r["order"] == "written"}
    reverse = {r["case_id"]: r["choice"] for r in rows if r["order"] == "reversed"}
    ms = [r["total_ms"] for r in rows]
    levels = sorted({r["level"] for r in rows})
    scaling = {b["threads"]: {"median_ms": median([r["total_ms"] for r in b["rows"]]), "n": len(b["rows"])}
               for b in blocks if b["block"] != "main"}
    main_written = {r["case_id"]: r for r in rows if r["order"] == "written"}
    def diffs(bs):  # choice changes and max |probability change| vs the main block's written-order decisions
        rs = [r for b in bs for r in b["rows"]]
        return (sum(r["choice"] != main_written[r["case_id"]]["choice"] for r in rs),
                max([abs(r["distribution"][k] - main_written[r["case_id"]]["distribution"][k])
                     for r in rs for k in r["distribution"]] or [0]))
    other = [b for b in blocks if b["block"] != "main" and b["threads"] != main["threads"]]
    same = [b for b in blocks if b["block"] != "main" and b["threads"] == main["threads"]]
    thread_choice_diffs, thread_prob_diff = diffs(other)
    nondet_choice_diffs, nondet_prob_diff = diffs(same)
    cached = next((b for b in blocks if b["block"] == f"t{main['threads']}"), None)
    # Same decision, same thread count, two blocks: their ratio shows how disturbed the timings were.
    repeat = [(r["case_id"], main_written[r["case_id"]]["total_ms"], r["total_ms"]) for r in (cached or {"rows": []})["rows"]]
    ratio = lambda a, b: max(a, b) / max(1e-6, min(a, b))  # epsilon: a 0.0 ms reading must not divide by zero
    worst = max(repeat, key=lambda x: ratio(x[1], x[2]), default=None)
    load = lambda b: None if b is None else {**b["phases"], "total": sum(b["phases"].values()),
                                              "spawn_to_ready": b["spawn_to_ready_ms"], "file_bytes": b["file_bytes"],
                                              "state": b["load_state"]}
    return {
        "model": name, "threads_main": main["threads"], "cases": len(written),
        "correct": rate(rows, "correct"), "acceptable": rate(rows, "acceptable"),
        "by_level": by("level"), "by_phrasing": by("phrasing"),
        "flips": sum(written[c] != reverse[c] for c in written), "flip_cases": [c for c in written if written[c] != reverse[c]],
        "median_ms": median(ms), "p95_ms": p95(ms),
        "tok_per_s": median([r["tokens"] / (max(1e-6, r["total_ms"]) / 1000) for r in rows]),
        "split_ms": {k: median([r[k] for r in rows]) for k in ("tokenise_ms", "forward_ms", "post_ms")},
        "split_share": {k: sum(r[k] for r in rows) / max(1e-6, sum(ms)) for k in ("tokenise_ms", "forward_ms", "post_ms")},
        "fit": fit([r["tokens"] for r in rows], ms),
        "tokens_by_level": {L: median([r["tokens"] for r in rows if r["level"] == L]) for L in levels},
        "truncated": sum(r["longest"] >= r["limit"] for r in rows), "limit": rows[0]["limit"],
        "scaling": scaling, "repeat_worst": worst and {"case_id": worst[0], "main_ms": worst[1], "repeat_ms": worst[2],
                                                        "ratio": ratio(worst[1], worst[2])},
        "thread_choice_diffs": thread_choice_diffs, "thread_max_prob_diff": thread_prob_diff,
        "nondet_choice_diffs": nondet_choice_diffs, "nondet_max_prob_diff": nondet_prob_diff,
        "nondet_n": sum(len(b["rows"]) for b in same), "thread_n": sum(len(b["rows"]) for b in other),
        "cold": load(main), "cached": load(cached),
        "peak_mib": max(b["peak_mib"] for b in blocks), "wait4_maxrss_mib": max(b["wait4_maxrss_mib"] or 0 for b in blocks),
        "timing_note": main["timing_note"], "token_def": main["token_def"], "torch_threads": {b["block"]: b["torch_threads"] for b in blocks},
    }


# ---------------------------------------------------------------- hints (every hint carries its numbers)

def hints(m, mem_total_mib):
    out = []
    pct = lambda x: f"{x * 100:.0f}%"
    f = m["fit"]
    if f is None:
        out.append(f"no latency-vs-length hint: all decisions had the same token count "
                   f"({sorted(set(m['tokens_by_level'].values()))}), nothing to fit")
    elif f["r2"] < 0.5:
        out.append(f"latency is poorly explained by input length (R2 {f['r2']:.2f}, n {f['n']}, tokens "
                   f"{f['tokens_min']}-{f['tokens_max']}) -> shorter input is not a reliable speed-up")
    elif f["per_token_ms"] <= 0:
        out.append(f"no measurable per-token cost (slope {f['per_token_ms']:.3f} ms/token, R2 {f['r2']:.2f}) "
                   "-> shorter input will not help")
    else:
        top = max(m["tokens_by_level"])
        t = m["tokens_by_level"][top]
        # a negative fitted fixed term would push this past 100%; cap it, the evidence shows the term
        share = min(1.0, f["per_token_ms"] * t / max(1e-6, f["fixed_ms"] + f["per_token_ms"] * t))
        ev = (f"fit {f['fixed_ms']:.0f} ms + {f['per_token_ms']:.2f} ms/token x {t:.0f} tokens at level {top}, "
              f"R2 {f['r2']:.2f}")
        if share >= 0.5:
            out.append(f"per-token cost is {pct(share)} of predicted latency at level {top} ({ev}) -> shorter input will help")
        else:
            out.append(f"fixed cost is {pct(1 - share)} of predicted latency at level {top} ({ev}) -> shortening input "
                       "saves little; fewer model calls will help more")
    s, ms = m["split_share"], m["split_ms"]
    split_ev = (f"median tokenise {ms['tokenise_ms']:.1f} / forward {ms['forward_ms']:.1f} / post {ms['post_ms']:.1f} ms; "
                f"shares {pct(s['tokenise_ms'])}/{pct(s['forward_ms'])}/{pct(s['post_ms'])}")
    if s["tokenise_ms"] >= 0.2:
        out.append(f"tokenisation is {pct(s['tokenise_ms'])} of decision time ({split_ev}) -> pre-tokenise constant "
                   "parts or use a faster tokenizer")
    if s["post_ms"] >= 0.2:
        out.append(f"post-processing is {pct(s['post_ms'])} of decision time ({split_ev}) -> the glue code around the "
                   "model is worth optimising")
    if s["forward_ms"] >= 0.8:
        out.append(f"model forward is {pct(s['forward_ms'])} of decision time ({split_ev}) -> only fewer tokens or a "
                   "faster model will help; glue code is not the bottleneck")
    sc = {t: v["median_ms"] for t, v in m["scaling"].items() if v["median_ms"] is not None}
    rw = m["repeat_worst"]
    if len(sc) >= 2 and rw and rw["ratio"] > 1.3:
        out.append(f"no thread-count hint: the same {m['threads_main']}-thread decision took {rw['main_ms']:.0f} ms in the "
                   f"main block and {rw['repeat_ms']:.0f} ms in the t{m['threads_main']} block ({rw['case_id']}, "
                   f"{rw['ratio']:.1f}x; limit 1.3x) -> timings were disturbed (thermal or scheduler); rerun the scaling blocks")
    elif len(sc) >= 2:
        lo = min(sc)
        best = min(sc, key=lambda t: (sc[t], t))  # fastest; a tie goes to fewer threads
        ranked = " < ".join(f"{t} thr {sc[t]:.0f} ms" for t in sorted(sc, key=lambda t: (sc[t], t)))
        ev = f"ranked {ranked} (median of {m['scaling'][lo]['n']} each)"
        noise = max(0.15, rw["ratio"] - 1 if rw else 0)  # a gap below the repeat noise is not a finding
        gain = (sc[lo] - sc[best]) / sc[lo]
        if best == lo:
            out.append(f"no thread count beats {lo} threads ({ev}) -> run at {lo} threads")
        elif gain <= noise:
            out.append(f"fastest ({best} threads) is only {pct(gain)} faster than {lo}, within the {pct(noise)} noise floor "
                       f"({ev}) -> {lo} threads frees cores at no measurable cost")
        else:
            out.append(f"{best} threads is {pct(gain)} faster than {lo}, above the {pct(noise)} noise floor ({ev}) "
                       f"-> run at {best} threads")
    if m["nondet_choice_diffs"]:
        out.append(f"run-to-run non-determinism: the same decision at the same {m['threads_main']} threads changed choice "
                   f"in {m['nondet_choice_diffs']}/{m['nondet_n']} repeats (max probability difference "
                   f"{m['nondet_max_prob_diff']:.4f}) -> results are not reproducible run to run")
    if m["thread_choice_diffs"]:
        if m["nondet_choice_diffs"]:
            out.append(f"choice changed at other thread counts in {m['thread_choice_diffs']}/{m['thread_n']} decisions, "
                       "but same-thread repeats also changed -> cannot separate a thread effect from run-to-run "
                       "non-determinism")
        else:
            out.append(f"choice changed with thread count in {m['thread_choice_diffs']}/{m['thread_n']} decisions (max "
                       f"probability difference {m['thread_max_prob_diff']:.4f}) while same-thread repeats matched "
                       f"({m['nondet_n']}) -> results depend on thread count")
    cold, cached = m["cold"], m["cached"]
    if cold and cached:
        d = cold["total"] - cached["total"]
        if d >= 1000 and d >= 0.25 * cached["total"]:
            out.append(f"{cold['state']['mode'].split(' ')[0]} load is {d / 1000:.1f} s slower than cached "
                       f"({cold['total'] / 1000:.1f} vs {cached['total'] / 1000:.1f} s; file read "
                       f"{cold['file_read'] / 1000:.1f} vs {cached['file_read'] / 1000:.1f} s) -> keep the worker resident "
                       "or the weights in page cache")
        for phase, advice in (("init", "a faster-loading format or a resident worker will help"),
                              ("import", "library import dominates; a resident worker avoids it")):
            share = cached[phase] / cached["total"]
            if share >= 0.5:
                out.append(f"{phase} is {pct(share)} of cached load ({cached[phase] / 1000:.1f} of "
                           f"{cached['total'] / 1000:.1f} s) -> {advice}")
        if m["median_ms"] and cached["warmup"] >= 2 * m["median_ms"] and cached["warmup"] - m["median_ms"] >= 200:
            out.append(f"first call takes {cached['warmup']:.0f} ms vs median {m['median_ms']:.0f} ms, although the warm-up "
                       "input is a short 2-option decision -> make a warm-up call at startup")
    n = m["cases"]
    if n < MIN_N:
        out.append(f"order flips {m['flips']}/{n} cases: too few to judge (need >= {MIN_N} cases)")
    elif m["flips"] / n >= 0.1:
        out.append(f"choice changed under reversed order in {m['flips']}/{n} cases ({pct(m['flips'] / n)}) -> order bias; "
                   "average over orders or fix the order")
    lv = {L: v["correct"] for L, v in m["by_level"].items()}
    few = {L: c[1] for L, c in lv.items() if c[1] < MIN_N}
    if len(lv) >= 2 and (min(lv) in few or max(lv) in few):
        out.append("level: " + ", ".join(f"L{L} {c[1]} decisions" for L, c in lv.items()) +
                   f" -> too few to judge (need >= {MIN_N} decisions at the lowest and highest level)")
    elif len(lv) >= 2:
        a, b = min(lv), max(lv)
        ra, rb = lv[a][0] / lv[a][1], lv[b][0] / lv[b][1]
        if ra - rb >= 0.3:
            out.append(f"accuracy falls from {pct(ra)} at level {a} ({lv[a][0]}/{lv[a][1]}) to {pct(rb)} at level {b} "
                       f"({lv[b][0]}/{lv[b][1]}) -> the model breaks down at the harder levels")
    ph = {p: v["correct"] for p, v in m["by_phrasing"].items() if v["correct"][1] >= MIN_N}
    if len(m["by_phrasing"]) >= 2 and len(ph) < 2:
        out.append("phrasing: " + ", ".join(f"{p} {v['correct'][1]} decisions" for p, v in m["by_phrasing"].items()) +
                   f" -> too few to judge (need >= {MIN_N} decisions in at least two phrasings)")
    elif len(ph) >= 2:
        best, worst = max(ph, key=lambda p: ph[p][0] / ph[p][1]), min(ph, key=lambda p: ph[p][0] / ph[p][1])
        gap = ph[best][0] / ph[best][1] - ph[worst][0] / ph[worst][1]
        if gap >= 0.25:
            out.append(f"sensitive to phrasing: '{best}' {ph[best][0]}/{ph[best][1]} vs '{worst}' {ph[worst][0]}/"
                       f"{ph[worst][1]} correct ({pct(gap)} gap) -> normalise phrasing before the model")
    if m["truncated"]:
        out.append(f"{m['truncated']} decisions hit the {m['limit']}-token limit -> input was cut; shorten it")
    if m["peak_mib"] >= 0.4 * mem_total_mib:
        out.append(f"peak RAM {m['peak_mib']} MiB is {pct(m['peak_mib'] / mem_total_mib)} of device RAM "
                   f"({mem_total_mib} MiB) -> only one such model can stay resident")
    return out


# ---------------------------------------------------------------- report

def frac(t):
    return f"{t[0]}/{t[1]} ({t[0] / t[1] * 100:.0f}%)" if t[1] else "-"


def report(res):
    L = []
    p = L.append
    p(f"LADDER REPORT  {res['started']}")
    p(f"cases {res['cases_file']} sha256 {res['cases_sha256'][:16]}  n={res['n_cases']}  levels {res['levels']}")
    p(f"models {', '.join(res['models'])}  threads {res['threads']}  cores {CORES}  device RAM {res['mem_total_mib']} MiB")
    p("timing: perf_counter in the worker; decision = wall time of adapter.decide; post = total - tokenise - forward")
    for m in res["per_model"]:
        p("")
        p(f"== {LABEL.get(m['model'], m['model'])} [{MODELS[m['model']][3]}] ==")
        p(f"accuracy        correct {frac(m['correct'])}  acceptable {frac(m['acceptable'])}  (both orders)")
        p("  by level      " + "  ".join(f"L{k} {frac(v['correct'])}" for k, v in m["by_level"].items()))
        p("  by phrasing   " + "  ".join(f"{k} {frac(v['correct'])}" for k, v in m["by_phrasing"].items()))
        p(f"order flips     {m['flips']}/{m['cases']}" + (f"  {m['flip_cases']}" if m["flips"] else ""))
        p(f"decision ms     median {m['median_ms']:.0f}  P95 {m['p95_ms']:.0f}  (main block, {m['threads_main']} threads, "
          f"n={m['correct'][1]})")
        sm = m["split_ms"]
        p(f"  split median  tokenise {sm['tokenise_ms']:.1f}  forward {sm['forward_ms']:.1f}  post {sm['post_ms']:.1f}   "
          f"[{m['timing_note']}]")
        p(f"tokens/s        {m['tok_per_s']:.0f}   tokens by level " +
          "  ".join(f"L{k} {v:.0f}" for k, v in m["tokens_by_level"].items()) +
          f"   limit {m['limit']}, hit {m['truncated']}x")
        p(f"token count     {m['token_def']}")
        f = m["fit"]
        p("latency fit     " + (f"ms = {f['fixed_ms']:.1f} + {f['per_token_ms']:.3f} x tokens   R2 {f['r2']:.2f}  "
                                f"(n={f['n']}, tokens {f['tokens_min']}-{f['tokens_max']})" if f else "not possible (no token spread)"))
        p("threads         " + "  ".join(f"{t}: {v['median_ms']:.0f} ms" for t, v in sorted(m["scaling"].items())) +
          f"  (written order, fresh cached worker, n={next(iter(m['scaling'].values()))['n']} each)"
          f"   torch threads seen {m['torch_threads']}")
        rw = m["repeat_worst"]
        if rw:
            p(f"                repeat check: worst {rw['case_id']} {rw['main_ms']:.0f} ms (main) vs {rw['repeat_ms']:.0f} ms "
              f"(t{m['threads_main']}) = {rw['ratio']:.2f}x" + ("  DISTURBED (>1.3x)" if rw["ratio"] > 1.3 else ""))
        p(f"                run-to-run non-determinism (main vs t{m['threads_main']}, same threads): choice differs in "
          f"{m['nondet_choice_diffs']}/{m['nondet_n']}, max prob diff {m['nondet_max_prob_diff']:.4f}")
        p(f"                other thread counts vs main: choice differs in {m['thread_choice_diffs']}/{m['thread_n']}, "
          f"max prob diff {m['thread_max_prob_diff']:.4f}")
        p(f"load (s)        {'phase':<10}{'cold':>8}{'cached':>8}")
        for ph in ("file_read", "import", "init", "warmup", "total", "spawn_to_ready"):
            c, w = m["cold"], m["cached"]
            p(f"                {ph:<10}{c[ph] / 1000:>8.2f}{(w[ph] / 1000 if w else float('nan')):>8.2f}")
        p(f"                weight files {m['cold']['file_bytes'] / 2**20:.0f} MiB; cold = {m['cold']['state']['mode']}")
        st = m["cold"]["state"]
        if "fallback_reason" in st:
            p(f"                full cold not used: {st['fallback_reason']}")
        elif "cached_mib_after" in st:
            p(f"                page cache {st['cached_mib_before']} -> {st['cached_mib_after']} MiB after the human drop")
        p(f"RAM             peak {m['peak_mib']} MiB (VmHWM, worker{' + llama-server' if m['model'] == 's1o' else ''})   "
          f"largest single process {m['wait4_maxrss_mib']} MiB (wait4)")
        p("HINTS")
        for h in m["hints"]:
            p(f"  - {h}")
    p("")
    p("== CATEGORY WINNERS ==")
    ms = res["per_model"]
    def rank(label, key, fmt, reverse=False):
        vals = [(key(m), m["model"]) for m in ms if key(m) is not None]
        vals.sort(key=lambda v: v[0], reverse=reverse)
        p(f"{label:<17}{vals[0][1]:<12} " + "  ".join(f"{n} {fmt(v)}" for v, n in vals))
    rank("accuracy", lambda m: (m["correct"][0] / m["correct"][1], m["acceptable"][0] / m["acceptable"][1]),
         lambda v: f"{v[0] * 100:.0f}%/{v[1] * 100:.0f}%acc", reverse=True)
    rank("order stability", lambda m: m["flips"] / m["cases"], lambda v: f"{v * 100:.0f}% flips")
    rank("decision speed", lambda m: m["median_ms"], lambda v: f"{v:.0f} ms")
    rank("cached load", lambda m: m["cached"]["total"] if m["cached"] else None, lambda v: f"{v / 1000:.1f} s")
    rank("cold load", lambda m: m["cold"]["total"], lambda v: f"{v / 1000:.1f} s")
    rank("RAM", lambda m: m["peak_mib"], lambda v: f"{v} MiB")
    p("(ties keep the listed model order; accuracy ties break on acceptable)")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="all", help="comma list of " + ",".join(MODELS))
    ap.add_argument("--levels", default=None, help="comma list, e.g. 1,2")
    ap.add_argument("--threads", default="2,3,4", help="comma list; the highest also runs the main block")
    ap.add_argument("--thermal-log", default=None, help="root thermal log (z9/z10/z11 millidegrees); gates every block "
                    "on z9 <= (reading at run start) + 2 degC and prints start/end readings")
    a = ap.parse_args()
    models = list(MODELS) if a.models == "all" else a.models.split(",")
    if set(models) - set(MODELS):
        raise SystemExit(f"unknown models {sorted(set(models) - set(MODELS))}")
    threads = sorted({int(t) for t in a.threads.split(",")})
    levels = {int(x) for x in a.levels.split(",")} if a.levels else None
    cases = load_cases(a.cases, levels)
    if not cases:
        raise SystemExit("no cases left after --levels")
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "logs").mkdir()
    check_cores("at start")
    thermal = None
    if a.thermal_log:
        idle = read_thermal(a.thermal_log)
        print(f"thermal idle reading (gate z9 <= idle + {GATE_MC / 1000:.0f} degC): {fmt_thermal(idle)}", flush=True)
        thermal = (a.thermal_log, idle)
    mem_total = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemTotal"))
    res = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cases_file": str(Path(a.cases).resolve()),
           "cases_sha256": hashlib.sha256(Path(a.cases).read_bytes()).hexdigest(), "n_cases": len(cases),
           "levels": sorted({c["level"] for c in cases}), "models": models, "threads": threads,
           "mem_total_mib": mem_total, "thermal_idle": thermal and thermal[1], "per_model": []}
    with open(a.out / "decisions.jsonl", "x") as log:
        for name in models:
            files = files_of(WEIGHTS[name])
            print(f"[{name}] main block: cold load, {len(cases)} cases x 2 orders at {threads[-1]} threads", flush=True)
            blocks = [run_block(name, "main", threads[-1], cases, ("written", "reversed"), files, a.out, log, cold=True,
                                thermal=thermal)]
            for t in threads:
                print(f"[{name}] t{t}: cached load, {len(cases)} cases written order", flush=True)
                blocks.append(run_block(name, f"t{t}", t, cases, ("written",), files, a.out, log, cold=False,
                                        thermal=thermal))
            m = analyse(name, blocks, cases)
            m["hints"] = hints(m, mem_total)
            m["blocks"] = [{k: v for k, v in b.items() if k != "rows"} for b in blocks]
            res["per_model"].append(m)
    order = {"CANDIDATE": 0, "OUT (speed)": 1, "DROPPED": 2}
    res["per_model"].sort(key=lambda m: order[MODELS[m["model"]][3]])
    (a.out / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    text = report(res)
    (a.out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
