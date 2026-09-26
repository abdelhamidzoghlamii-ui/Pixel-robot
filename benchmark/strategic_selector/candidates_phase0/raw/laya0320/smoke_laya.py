#!/usr/bin/env python3
"""Laya 0.3.20 smoke: load pinned local snapshot on CPU, one choice decision on toy.json."""
import json
import time
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parent / "hf-cache/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
toy = json.loads((Path(__file__).resolve().parent.parent / "toy.json").read_text())

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
began = time.perf_counter()
import laya
agent = laya.load(str(SNAPSHOT), device="cpu")
load_ms = (time.perf_counter() - began) * 1000
began = time.perf_counter()
with torch.inference_mode():
    raw = agent.predict(toy["state"], {"next": {"type": "choice", "instructions": toy["instruction"],
                                                "criteria": toy["options"]}})
first_ms = (time.perf_counter() - began) * 1000
answer = raw["answers"]["next"]
print(json.dumps({"laya": laya.__version__, "torch": torch.__version__, "load_ms": round(load_ms, 1),
                  "first_call_ms": round(first_ms, 1), "choice": answer["choice"],
                  "probabilities": answer.get("probabilities"), "confidence": answer.get("confidence")}))
