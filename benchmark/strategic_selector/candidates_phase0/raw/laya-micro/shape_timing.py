"""Item 7 evidence: same graphs, smoke shape vs quantize_onnx.py's default bench shape."""
import json, sys, time
import numpy as np, onnxruntime as ort
sys.path.insert(0, "src")
from runtime import LayaRuntime, to_internal, INPUT_NAMES, OUTPUT_NAMES
toy = json.load(open("../toy.json"))
rt = LayaRuntime("laya-multilingual-en", "laya-micro.int8.onnx", 4)
q = to_internal({"type": "choice", "instructions": toy["instruction"], "criteria": toy["options"]})
ids, markers = rt.build_sequence(toy["state"], q)
print("SMOKE_SHAPE items=1 seq=%d nopt=%d" % (len(ids), len(markers)))
def feed(n_item, n_seq, n_opt, rng):
    return {"input_ids": rng.integers(5, 1000, (n_item, n_seq), dtype=np.int64),
            "attention_mask": np.ones((n_item, n_seq), dtype=np.int64),
            "marker_pos": np.stack([np.sort(rng.choice(np.arange(1, n_seq), n_opt, replace=False)) for _ in range(n_item)]),
            "marker_mask": np.ones((n_item, n_opt), dtype=bool), "qtype": np.zeros((n_item,), dtype=np.int64)}
opts = ort.SessionOptions(); opts.intra_op_num_threads = 4; opts.inter_op_num_threads = 1
for graph in ("laya-micro.onnx", "laya-micro.int8.onnx"):
    s = ort.InferenceSession(graph, opts, providers=["CPUExecutionProvider"])
    for shape in ((1, len(ids), len(markers)), (3, 259, 28)):
        f = feed(*shape, np.random.default_rng(0)); s.run(OUTPUT_NAMES, f)
        t = []
        for _ in range(10):
            t0 = time.perf_counter(); s.run(OUTPUT_NAMES, f); t.append((time.perf_counter() - t0) * 1000)
        print(f"{graph} shape={shape} median_ms={np.median(t):.0f}")
