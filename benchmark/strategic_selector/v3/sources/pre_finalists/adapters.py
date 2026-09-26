"""One adapter per candidate: decide(state, options, instruction) -> {option: probability}.

`options` is an ordered {key: description} dict; the returned dict has one
probability per offered key (the full distribution, not just the argmax).
Constructors load the model and set `load_ms`. Heavy imports happen inside the
adapter, so each candidate runs in its own virtualenv (see measure.py).
`extra` holds per-call diagnostics that the runner logs with the call.
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SEL = Path("/termux-home/sel-candidates")
LAYA_SNAPSHOT = "hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def torch_threads():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    return torch


class Mock:
    """Test double: order-invariant scores from option keys; no model."""
    name = "mock"

    def __init__(self):
        self.load_ms, self.extra = 0.0, {}

    def decide(self, state, options, instruction):
        raw = {k: 1 + sum(map(ord, k)) % 7 for k in options}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}


class Von11:
    """von-sdk 1.1.1, weights wfzyx/von@d8bb5e07 (first decide loads the weights)."""
    name = "von11"

    def __init__(self):
        began = time.perf_counter()
        self.torch = torch_threads()
        import von
        self.von, self.extra = von, {}
        self.load_ms = (time.perf_counter() - began) * 1000

    def decide(self, state, options, instruction):
        with self.torch.inference_mode():
            answer = self.von.decide(state=state, choices=options, instructions=instruction)
        self.extra = {"native_choice": answer.choice}
        return dict(answer.probabilities)


class _Laya:
    def __init__(self, snapshot, subfolder=None):
        began = time.perf_counter()
        self.torch = torch_threads()
        import laya
        self.agent = laya.load(str(snapshot), device="cpu", subfolder=subfolder)
        self.version, self.extra = laya.__version__, {}
        self.load_ms = (time.perf_counter() - began) * 1000

    def decide(self, state, options, instruction):
        with self.torch.inference_mode():
            raw = self.agent.predict(state, {"next": {"type": "choice", "instructions": instruction,
                                                      "criteria": options}})
        answer = raw["answers"]["next"]
        self.extra = {"native_choice": answer["choice"]}
        return dict(answer["probabilities"])


class LayaEn(_Laya):
    """laya 0.3.20, English root checkpoint, convaiinnovations/laya@55cf4c4e."""
    name = "laya_en"

    def __init__(self):
        super().__init__(SEL / "laya0320/hf-cache" / LAYA_SNAPSHOT)


class LayaMulti(_Laya):
    """laya 0.3.20, stock fp32 multilingual subfolder, same revision."""
    name = "laya_multi"

    def __init__(self):
        super().__init__(SEL / "laya-micro/hf-cache" / LAYA_SNAPSHOT, subfolder="multilingual")


class LayaMicro:
    """laya-micro@686fc466 runtime.py: vocab-pruned (seed-top 8192, 15,188 tokens), int8 block-64 ONNX."""
    name = "laya_micro"
    ROOT = SEL / "laya-micro"

    def __init__(self):
        began = time.perf_counter()
        sys.path.insert(0, str(self.ROOT / "src"))
        from runtime import LayaRuntime
        from tokenizers import Tokenizer
        self.rt = LayaRuntime(str(self.ROOT / "laya-multilingual-en-8192"),
                              str(self.ROOT / "laya-micro-8192.int8.onnx"), 4)
        self.load_ms = (time.perf_counter() - began) * 1000
        # Pruning only guarantees tokenizer parity on laya-micro's own corpus; log ours per call.
        self.stock = Tokenizer.from_file(str(self.ROOT / "hf-cache" / LAYA_SNAPSHOT / "multilingual/tokenizer/tokenizer.json"))
        self.extra = {}

    def decide(self, state, options, instruction):
        out = self.rt.decide(state, {"next": {"type": "choice", "instructions": instruction,
                                              "criteria": options}})
        answer = out["answers"]["next"]
        text = "\n".join([state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
                          instruction, *options, *[v for v in options.values() if v]])
        self.extra = {"native_choice": answer["choice"],
                      "tokenizer_parity_vs_stock": self.stock.encode(text).tokens == self.rt.tok.encode(text).tokens}
        return dict(answer["probabilities"])


class S1O:
    """system-one-open@77f1f7cc prompt builder, zero-shot on base Gemma 4 E2B Q4_K_M, llama.cpp 1609.

    One forward pass; the distribution is the raw softmax over the full vocabulary
    (post_sampling_probs=false, the pre-sampling path of build 1609) read at the
    "Answer: (" slot, restricted to the option letters and renormalised, which is
    engine.slot_logits' masked softmax at temperature 1.0.
    """
    name = "s1o"
    LLAMA = Path("/termux-home/llama.cpp/build/bin")
    GGUF = "/termux-home/models/gemma-4-e2b-it-q4_k_m.gguf"
    PORT = 8091
    N_PROBS = 1000

    def __init__(self):
        sys.path.insert(0, str(SEL / "s1o/src"))
        from s1.schema import Example, Q, LETTERS, build
        self.Example, self.Q, self.LETTERS, self.build = Example, Q, LETTERS, build
        self.url = f"http://127.0.0.1:{self.PORT}"
        env = dict(os.environ, LD_LIBRARY_PATH=f"{self.LLAMA}:/data/data/com.termux/files/usr/lib")
        self.log = open(os.environ.get("S1O_SERVER_LOG", "s1o-server.log"), "xb")
        began = time.perf_counter()
        self.server = subprocess.Popen(
            [str(self.LLAMA / "llama-server"), "-m", self.GGUF, "--threads", "4", "--threads-batch", "4",
             "--parallel", "1", "--swa-full", "--ctx-size", "2048", "--host", "127.0.0.1",
             "--port", str(self.PORT)], stdout=self.log, stderr=subprocess.STDOUT, env=env)
        while True:
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except OSError:
                pass
            if self.server.poll() is not None or time.perf_counter() - began > 180:
                raise RuntimeError(f"llama-server not healthy (exit={self.server.poll()})")
            time.sleep(0.2)
        self.load_ms = (time.perf_counter() - began) * 1000
        with_bos = self._post("/tokenize", {"content": "", "add_special": True})["tokens"]
        self.bos_token_id = with_bos[0] if with_bos else None
        self.letter_ids = [self.encode(L)[0] for L in LETTERS]
        assert all(len(self.encode(L)) == 1 for L in LETTERS[:12])
        self.extra = {}

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def encode(self, text, add_special_tokens=False):  # the tokenizer interface schema.build needs
        return self._post("/tokenize", {"content": text, "add_special": add_special_tokens})["tokens"]

    def completion_probs(self, ids, **overrides):
        body = {"prompt": ids, "n_predict": 1, "n_probs": self.N_PROBS, "post_sampling_probs": False,
                "temperature": 0, "cache_prompt": False}
        body.update(overrides)
        res = self._post("/completion", body)
        top = res["completion_probabilities"][0]
        key = "top_probs" if body["post_sampling_probs"] else "top_logprobs"
        return {e["id"]: (e["prob"] if "prob" in e else math.exp(e["logprob"])) for e in top[key]}

    def prompt_ids(self, state, options, instruction):
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        keys = list(options)
        q = self.Q(instruction, keys, -1, kind="choice", descs=[options[k] for k in keys])
        return self.build(self.Example(state, [q]), self, shuffle_choice=False, train=False)["ids"]

    def decide(self, state, options, instruction):
        ids = self.prompt_ids(state, options, instruction)
        by_id = self.completion_probs(ids)
        raw = {}
        for j, key in enumerate(options):
            p = by_id.get(self.letter_ids[j])
            if p is None:
                raise RuntimeError(f"letter {self.LETTERS[j]} outside top-{self.N_PROBS}")
            raw[key] = p
        mass = sum(raw.values())
        self.extra = {"prompt_tokens": len(ids), "letter_mass_full_vocab": mass}
        return {k: v / mass for k, v in raw.items()}

    def close(self):
        self.server.terminate()
        _, status, usage = os.wait4(self.server.pid, 0)
        self.log.close()
        return {"server_exit": os.waitstatus_to_exitcode(status),
                "server_peak_rss_mib_wait4": round(usage.ru_maxrss / 1024, 1)}


CANDIDATES = {c.name: c for c in (Mock, Von11, LayaEn, LayaMulti, LayaMicro, S1O)}
