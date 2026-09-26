#!/usr/bin/env python3
"""system-one-open zero-shot smoke on llama.cpp 1609 + installed Gemma 4 E2B Q4_K_M GGUF.

The trained system-one-open weights are unpublished, so this is the repo's prompt
builder (s1/schema.build, eval mode, no shuffle) on the *base* GGUF: one forward
pass, read the next-token distribution at the "Answer: (" slot, keep the option
letters, renormalise (== engine.slot_logits masked softmax at T=1.0).
Starts its own llama-server child and reaps it with os.wait4 for peak RSS.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from s1.schema import Example, Q, LETTERS, build

LLAMA = Path("/termux-home/llama.cpp/build/bin")
GGUF = "/termux-home/models/gemma-4-e2b-it-q4_k_m.gguf"
PORT = 8091
URL = f"http://127.0.0.1:{PORT}"
toy = json.loads((HERE.parent / "toy.json").read_text())


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


class ServerTokenizer:
    """Just what schema.build needs, backed by the GGUF's own tokenizer."""
    def __init__(self):
        with_bos = post("/tokenize", {"content": "", "add_special": True})["tokens"]
        self.bos_token_id = with_bos[0] if with_bos else None

    def encode(self, text, add_special_tokens=False):
        return post("/tokenize", {"content": text, "add_special": add_special_tokens})["tokens"]


env = dict(os.environ, LD_LIBRARY_PATH=f"{LLAMA}:/data/data/com.termux/files/usr/lib")
server = subprocess.Popen(
    ["taskset", "-c", "4-7", str(LLAMA / "llama-server"), "-m", GGUF, "--threads", "4",
     "--threads-batch", "4", "--parallel", "1", "--swa-full", "--ctx-size", "2048",
     "--host", "127.0.0.1", "--port", str(PORT)],
    stdout=open(HERE / "server.log", "xb"), stderr=subprocess.STDOUT, env=env)
began = time.perf_counter()
try:
    while True:
        try:
            if json.loads(urllib.request.urlopen(URL + "/health", timeout=2).read()).get("status") == "ok":
                break
        except OSError:
            pass
        if server.poll() is not None or time.perf_counter() - began > 180:
            raise SystemExit(f"llama-server did not become healthy (exit={server.poll()})")
        time.sleep(0.2)
    load_ms = (time.perf_counter() - began) * 1000

    tok = ServerTokenizer()
    keys = list(toy["options"])
    q = Q(toy["instruction"], keys, -1, kind="choice", descs=[toy["options"][k] for k in keys])
    built = build(Example(toy["state"], [q]), tok, shuffle_choice=False, train=False)
    letters = {LETTERS[j]: tok.encode(LETTERS[j]) for j in range(len(keys))}
    assert all(len(v) == 1 for v in letters.values()), letters

    began = time.perf_counter()
    res = post("/completion", {"prompt": built["ids"], "n_predict": 1, "n_probs": 200,
                               "temperature": 0, "cache_prompt": False})
    first_ms = (time.perf_counter() - began) * 1000
    top = res["completion_probabilities"][0]
    entries = top.get("top_logprobs") or top.get("probs")
    by_id = {e["id"]: e for e in entries}
    raw = {}
    for j, k in enumerate(keys):
        e = by_id.get(letters[LETTERS[j]][0])
        if e is None:
            raise SystemExit(f"letter {LETTERS[j]} not in top-{len(entries)}: cannot form distribution")
        raw[k] = e["prob"] if "prob" in e else 2.718281828459045 ** e["logprob"]
    mass = sum(raw.values())
    probs = {k: round(v / mass, 4) for k, v in raw.items()}
    print(json.dumps({"llama_cpp": "1609 (e1a1abb7)", "gguf": GGUF, "prompt_tokens": len(built["ids"]),
                      "load_ms": round(load_ms, 1), "first_call_ms": round(first_ms, 1),
                      "choice": max(probs, key=probs.get), "probabilities": probs,
                      "letter_mass_in_full_vocab": round(mass, 4),
                      "top1_token": entries[0].get("token"), "generated": res.get("content")}))
finally:
    server.terminate()
    _, status, usage = os.wait4(server.pid, 0)
    print(json.dumps({"server_exit": os.waitstatus_to_exitcode(status),
                      "server_peak_rss_mib_wait4": round(usage.ru_maxrss / 1024, 1)}))
