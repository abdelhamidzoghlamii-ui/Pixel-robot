#!/usr/bin/env python3
"""One selector model per process, JSON lines on stdin/stdout.

Usage (inside the model's venv): worker.py ADAPTER
Reuses the v3 harness adapters unmodified. Protocol:
  -> ready line: {"ready": true, "load_ms": ..., "warmup_ms": ...} or {"error": "..."}
  <- {"state": str, "options": {key: desc}, "instruction": str, "text": bool}
  -> {"dist": {key: p}, "ms": float, "text": str|null} or {"error": "..."}
EOF on stdin (or SIGTERM) closes the adapter, which stops s1o's llama-server.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

V3 = str(Path(__file__).resolve().parents[2] / "v3")

# Libraries print to stdout; keep the real stdout private for the protocol.
_out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(obj):
    _out.write(json.dumps(obj) + "\n")


def encoded(ad, state, options, instruction):
    """(token-id sequences the model reads, decode fn, model token limit), rebuilt with its own tokenizer."""
    qdef = {"type": "choice", "instructions": instruction, "criteria": options}
    kind = type(ad).__name__
    if kind in ("LayaEnOnnx", "LayaMicro"):
        from runtime import to_internal
        ids, _ = ad.rt.build_sequence(state, to_internal(qdef))
        return [ids], lambda i: ad.rt.tok.decode(i, skip_special_tokens=False), ad.rt.max_len
    if kind in ("LayaEn", "LayaMulti"):
        from laya.common import build_sequence
        ag = ad.agent
        limit = ag.cfg.get("max_len", 512)
        ids, _ = build_sequence(ag.tok, state, ag._to_internal(qdef), limit, ag.cfg.get("head_max_len", 192))
        return [ids], ag.tok.decode, limit
    if kind == "Von11":
        from von.engine import VonEngine
        from von.backends.option_marker_backend import _format_state
        model = VonEngine.get_instance().backend._get_model()
        packed = model.pack_sequence(_format_state(state), instruction,
                                     [(d or k).strip() for k, d in options.items()])
        return [model.tokenizer(packed)["input_ids"]], model.tokenizer.decode, model.tokenizer.model_max_length
    if kind == "Von10Nli":
        _, tok = ad.backend._get_model_and_tok()
        st = ad.format_state(state)
        return ([tok(st, f"{instruction} {d or k}", truncation=True, max_length=512)["input_ids"]
                 for k, d in options.items()], tok.decode, 512)  # 512: evaluate_choice's truncation
    if kind == "S1O":
        return ([ad.prompt_ids(state, options, instruction)],
                lambda i: ad._post("/detokenize", {"tokens": i})["content"], 2048)  # --ctx-size 2048
    raise ValueError(kind)


def mem_mib(ad):
    """Current and peak RSS of this worker plus s1o's llama-server, from /proc."""
    pids = ["self"] + ([ad.server.pid] if hasattr(ad, "server") else [])
    cur = peak = 0
    for pid in pids:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                cur += int(line.split()[1]) // 1024
            elif line.startswith("VmHWM"):
                peak += int(line.split()[1]) // 1024
    return cur, peak


def main():
    sys.path.insert(0, V3)
    from adapters import CANDIDATES
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # run the finally: close()
    ad = None
    try:
        try:
            ad = CANDIDATES[sys.argv[1]]()
            began = time.perf_counter()  # warm-up: von loads its weights on the first decide
            ad.decide("The robot is idle in the hallway.", {"wait": "", "explore": ""}, "Pick one.")
            send({"ready": True, "load_ms": ad.load_ms, "warmup_ms": (time.perf_counter() - began) * 1000})
        except Exception as e:
            send({"error": f"{type(e).__name__}: {e}"})
            return
        for line in sys.stdin:
            req = json.loads(line)
            try:
                began = time.perf_counter()
                dist = ad.decide(req["state"], req["options"], req["instruction"])
                ms = (time.perf_counter() - began) * 1000
                seqs, decode, limit = encoded(ad, req["state"], req["options"], req["instruction"])
                text = None
                if req.get("text"):
                    text = decode(seqs[0]) if len(seqs) == 1 else "\n".join(
                        f"[pair {i + 1}] {decode(q)}" for i, q in enumerate(seqs))
                cur, peak = mem_mib(ad)
                send({"dist": {k: float(dist[k]) for k in req["options"]}, "ms": ms, "text": text,
                      "tokens": sum(map(len, seqs)), "longest": max(map(len, seqs)), "limit": limit,
                      "rss_mib": cur, "peak_mib": peak})
            except Exception as e:
                send({"error": f"{type(e).__name__}: {e}"})
    finally:
        if ad is not None and hasattr(ad, "close"):
            ad.close()


if __name__ == "__main__":
    main()
