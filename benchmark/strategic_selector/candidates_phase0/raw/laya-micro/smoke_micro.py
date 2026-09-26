#!/usr/bin/env python3
"""laya-micro smoke: runtime.py (onnxruntime, no torch), int8 graph, one choice on toy.json."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
toy = json.loads((HERE.parent / "toy.json").read_text())

began = time.perf_counter()
from runtime import LayaRuntime
rt = LayaRuntime(str(HERE / "laya-multilingual-en"), str(HERE / "laya-micro.int8.onnx"), 4)
load_ms = (time.perf_counter() - began) * 1000
began = time.perf_counter()
out = rt.decide(toy["state"], {"next": {"type": "choice", "instructions": toy["instruction"],
                                        "criteria": toy["options"]}})
first_ms = (time.perf_counter() - began) * 1000
answer = out["answers"]["next"]

# The pruned vocabulary only guarantees parity on its build corpus; check the toy text too.
from tokenizers import Tokenizer
stock = Tokenizer.from_file(str(next((HERE / "hf-cache/hub/models--convaiinnovations--laya/snapshots").glob("*/multilingual/tokenizer/tokenizer.json"))))
text = " ".join([toy["state"], toy["instruction"], *toy["options"], *toy["options"].values()])
parity = stock.encode(text).tokens == rt.tok.encode(text).tokens

print(json.dumps({"load_ms": round(load_ms, 1), "first_call_ms": round(first_ms, 1),
                  "choice": answer["choice"], "probabilities": answer["probabilities"],
                  "confidence": answer["confidence"], "torch_imported": "torch" in sys.modules,
                  "toy_tokenizer_parity_vs_stock": parity}))
