#!/usr/bin/env python3
"""Ladder worker: one model per process, JSON lines on stdin/stdout (the jevlike worker model).

Usage (inside the model's venv): ladder_worker.py ADAPTER
Env: LADDER_FILES = JSON list of the model's weight files; V3_THREADS/OMP_NUM_THREADS set by ladder.py.

Startup phases, reported in the ready line (ms, time.perf_counter):
  file_read  read every weight file once, so it sits in the page cache (a cold read if
             ladder.py evicted it); init then parses from cache
  import     the heavy libraries the adapter needs (torch / onnxruntime ...)
  init       the v3 adapter constructor, plus von's lazy weight load (_get_model), which the
             Von11 adapter otherwise defers to the first decide
  warmup     the first decide call
Per decision (ms): total = wall time around adapter.decide; tokenise = time inside the model's
tokenizer calls; forward = time inside the model forward (onnxruntime run / torch module forward /
s1o's /completion HTTP call); post = total - tokenise - forward (tensor packing, softmax, glue).
"""
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

JEVLIKE = Path("/termux-home/robot/benchmark/strategic_selector/manual/jevlike")
sys.path.insert(0, str(JEVLIKE))
import worker as jw  # noqa: E402  importing it moves library prints off the protocol stdout

IMPORTS = {"laya_en_onnx": ["onnxruntime", "tokenizers"], "laya_micro": ["onnxruntime", "tokenizers"],
           "laya_multi": ["torch", "laya"], "von11": ["torch", "von"], "von10_nli": ["torch", "von", "transformers"],
           "s1o": []}
TOKENIZER_METHODS = {"__call__", "encode", "encode_plus", "batch_encode_plus", "encode_batch", "tokenize"}
WARMUP = ("The robot is idle in the hallway.", {"wait": "", "explore": ""}, "Pick one.")


class Clock:
    def __init__(self):
        self.tok = self.fwd = 0.0
        self._fwd_start = None

    def timed(self, fn, field):
        def run(*a, **k):
            began = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                setattr(self, field, getattr(self, field) + time.perf_counter() - began)
        return run

    def hook(self, module):
        """Torch forward hooks on the top module only, so nested modules are not double counted."""
        def pre(*_):
            self._fwd_start = time.perf_counter()
        def post(*_):
            self.fwd += time.perf_counter() - self._fwd_start
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)


class Timed:
    """Stands in for a tokenizer or ONNX session: times the named methods, delegates the rest."""
    def __init__(self, obj, clock, field, methods):
        self._obj, self._clock, self._field, self._methods = obj, clock, field, methods

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        return self._clock.timed(attr, self._field) if name in self._methods else attr

    def __call__(self, *a, **k):
        return self._clock.timed(self._obj, self._field)(*a, **k)


def instrument(ad, clock):
    """Returns a note describing what was timed; also triggers von's deferred weight load."""
    kind = type(ad).__name__
    if kind in ("LayaEnOnnx", "LayaMicro"):
        ad.rt.build_sequence = clock.timed(ad.rt.build_sequence, "tok")
        ad.rt.sess = Timed(ad.rt.sess, clock, "fwd", {"run"})
        if kind == "LayaMicro":  # same runtime.py call, minus the adapter's per-call tokenizer-parity check
            from adapters import LayaEnOnnx
            ad.decide = types.MethodType(LayaEnOnnx.decide, ad)
        return "tokenise=runtime.build_sequence, forward=onnxruntime run"
    if kind in ("LayaEn", "LayaMulti"):
        ad.agent.tok = Timed(ad.agent.tok, clock, "tok", TOKENIZER_METHODS)
        clock.hook(ad.agent.model)
        return "tokenise=agent tokenizer calls, forward=agent.model forward hooks"
    if kind == "Von11":
        from von.engine import VonEngine
        model = VonEngine.get_instance().backend._get_model()
        model.tokenizer = Timed(model.tokenizer, clock, "tok", TOKENIZER_METHODS)
        clock.hook(model)
        return "tokenise=OptionMarkerModel.tokenizer calls, forward=OptionMarkerModel forward hooks"
    if kind == "Von10Nli":
        ad.backend._tokenizer = Timed(ad.backend._tokenizer, clock, "tok", TOKENIZER_METHODS)
        clock.hook(ad.backend._model)
        return "tokenise=NLI tokenizer calls, forward=NLI model forward hooks"
    if kind == "S1O":
        ad.encode = clock.timed(ad.encode, "tok")  # schema.build tokenizes through adapter.encode (/tokenize)
        ad.completion_probs = clock.timed(ad.completion_probs, "fwd")
        post = ad._post
        def logged(path, body):
            res = post(path, body)
            if path == "/completion":
                ad.server_timings = res.get("timings", {})
            return res
        ad._post = logged
        return "tokenise=/tokenize HTTP calls, forward=/completion HTTP call (server prompt eval inside)"
    raise ValueError(kind)


def read_files(paths):
    total = 0
    for p in paths:
        with open(p, "rb", buffering=0) as f:
            while (chunk := f.read(8 << 20)):
                total += len(chunk)
    return total


def main():
    name = sys.argv[1]
    phases, clock, ad = {}, Clock(), None
    ms = lambda t: (time.perf_counter() - t) * 1000
    try:
        try:
            t = time.perf_counter()
            file_bytes = read_files(json.loads(os.environ.get("LADDER_FILES", "[]")))
            phases["file_read"] = ms(t)
            t = time.perf_counter()
            for mod in IMPORTS[name]:
                importlib.import_module(mod)
            phases["import"] = ms(t)
            sys.path.insert(0, jw.V3)
            from adapters import CANDIDATES
            t = time.perf_counter()
            ad = CANDIDATES[name]()
            note = instrument(ad, clock)
            phases["init"] = ms(t)
            t = time.perf_counter()
            ad.decide(*WARMUP)
            phases["warmup"] = ms(t)
            if clock.tok == 0 or clock.fwd == 0:
                raise RuntimeError(f"instrumentation missed a phase: tokenise={clock.tok} forward={clock.fwd}")
            torch = sys.modules.get("torch")
            jw.send({"ready": True, "phases": phases, "file_bytes": file_bytes, "timing_note": note,
                     "threads_env": os.environ.get("V3_THREADS"),
                     "torch_threads": torch.get_num_threads() if torch else None})
        except Exception as e:
            jw.send({"error": f"{type(e).__name__}: {e}"})
            return
        for line in sys.stdin:
            req = json.loads(line)
            try:
                if req.get("cmd") == "mem":
                    cur, peak = jw.mem_mib(ad)
                    jw.send({"rss_mib": cur, "peak_mib": peak})
                    continue
                clock.tok = clock.fwd = 0.0
                began = time.perf_counter()
                dist = ad.decide(req["state"], req["options"], req["instruction"])
                total = ms(began)
                tok, fwd = clock.tok * 1000, clock.fwd * 1000
                seqs, _, limit = jw.encoded(ad, req["state"], req["options"], req["instruction"])  # untimed
                out = {"dist": {k: float(dist[k]) for k in req["options"]}, "total_ms": total,
                       "tokenise_ms": tok, "forward_ms": fwd, "post_ms": total - tok - fwd,
                       "tokens": sum(map(len, seqs)), "longest": max(map(len, seqs)), "limit": limit}
                if hasattr(ad, "server_timings"):
                    out["server_timings"] = ad.server_timings
                jw.send(out)
            except Exception as e:
                jw.send({"error": f"{type(e).__name__}: {e}"})
    finally:
        if ad is not None and hasattr(ad, "close"):
            ad.close()


if __name__ == "__main__":
    main()
