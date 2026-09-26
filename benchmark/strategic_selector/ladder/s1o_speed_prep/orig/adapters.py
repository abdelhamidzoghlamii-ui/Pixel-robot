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
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SEL = Path("/termux-home/sel-candidates")
LAYA_SNAPSHOT = "hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


THREADS = int(os.environ.get("V3_THREADS", "4"))  # set by measure.py for thread sweeps


def torch_threads():
    import torch
    torch.set_num_threads(THREADS)
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
                              str(self.ROOT / "laya-micro-8192.int8.onnx"), THREADS)
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
            [str(self.LLAMA / "llama-server"), "-m", self.GGUF, "--threads", str(THREADS),
             "--threads-batch", str(THREADS), "--parallel", "1", "--swa-full", "--ctx-size", "2048",
             "--host", "127.0.0.1", "--port", str(self.PORT),
             *shlex.split(os.environ.get("S1O_SERVER_ARGS", ""))],
            stdout=self.log, stderr=subprocess.STDOUT, env=env)
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


class LayaEnOnnx:
    """laya_en (convaiinnovations/laya@55cf4c4e root) exported to ONNX fp32 with laya-micro@686fc466's
    export_onnx.py (no pruning, no quantisation), run through laya-micro's runtime.py (onnxruntime).

    runtime.py reads mask_token_id from encoder/config.json, which only the multilingual config
    carries. laya_en_onnx/ckpt is an overlay: symlinks to the snapshot's tokenizer/ and
    rl_agent_config.json, plus a copy of encoder/config.json with mask_token_id = the tokenizer's
    own [MASK] id (50284), the id torch Laya uses via tok.mask_token_id."""
    name = "laya_en_onnx"
    ONNX = SEL / "laya_en_onnx/laya_en.onnx"
    CKPT = SEL / "laya_en_onnx/ckpt"

    def __init__(self):
        began = time.perf_counter()
        sys.path.insert(0, str(SEL / "laya-micro/src"))
        from runtime import LayaRuntime
        self.rt = LayaRuntime(str(self.CKPT), str(self.ONNX), THREADS)
        self.load_ms = (time.perf_counter() - began) * 1000
        self.extra = {}

    def decide(self, state, options, instruction):
        out = self.rt.decide(state, {"next": {"type": "choice", "instructions": instruction, "criteria": options}})
        answer = out["answers"]["next"]
        self.extra = {"native_choice": answer["choice"]}
        return dict(answer["probabilities"])


class Von11Perm:
    """von-sdk 1.1.1 option-marker, permutation-averaged: every cyclic rotation of the offered
    options is scored, mapped back to option keys and averaged; choice = argmax of the average.

    von-sdk 1.1.1 has no public batch API (decide/evaluate_choice take one question), but
    OptionMarkerModel.forward takes a padded batch with per-row mask positions, so all rotations
    go through ONE forward pass here. Each row then gets the backend's own _effective_temperature,
    exactly as evaluate_choice does. For the first V3_VON_CHECK_N decisions every rotation is also
    scored through the public evaluate_choice and the max |difference| is logged.
    """
    name = "von11_perm"

    def __init__(self):
        began = time.perf_counter()
        self.torch = torch_threads()
        from von.engine import VonEngine
        from von.types import Choice
        from von.backends import option_marker_backend as omb
        self.Choice, self.format_state = Choice, omb._format_state
        self.backend = VonEngine.get_instance().backend
        assert type(self.backend).__name__ == "OptionMarkerBackend"
        self.model = self.backend._get_model()  # load weights now, so load_ms includes them
        self.tok = self.model.tokenizer
        self.load_ms = (time.perf_counter() - began) * 1000
        self.check_left = int(os.environ.get("V3_VON_CHECK_N", "0"))
        self.extra = {}

    def rotations(self, state_text, instruction, options):
        keys = list(options)
        rots = [keys[i:] + keys[:i] for i in range(len(keys))]
        texts = [self.model.pack_sequence(state_text, instruction,
                                          [(options[k] or k).strip() for k in rot]) for rot in rots]
        enc = self.tok(texts, return_tensors="pt", padding=True)
        pos = [(row == self.model.mask_token_id).nonzero(as_tuple=True)[0].tolist() for row in enc["input_ids"]]
        assert all(len(p) == len(keys) for p in pos)
        with self.torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], mask_positions=pos)
        out = []
        for rot, lg in zip(rots, logits):
            t = self.backend._effective_temperature(lg, state_text, len(rot), self.tok, None)
            out.append(dict(zip(rot, self.torch.softmax(lg / max(t, 1e-4), dim=-1).tolist())))
        return rots, out

    def decide(self, state, options, instruction):
        state_text = self.format_state(state)
        rots, per_rot = self.rotations(state_text, instruction, options)
        avg = {k: sum(p[k] for p in per_rot) / len(per_rot) for k in options}
        self.extra = {"rotations": len(rots), "batched_forward": True,
                      "rotation_argmax": [max(p, key=p.get) for p in per_rot]}
        if self.check_left > 0:
            self.check_left -= 1
            diff = 0.0
            for rot, p in zip(rots, per_rot):
                ref = self.backend.evaluate_choice("next", state_text, self.Choice(
                    instructions=instruction, criteria={k: options[k] for k in rot})).probabilities
                diff = max(diff, max(abs(ref[k] - p[k]) for k in rot))
            self.extra["batched_vs_sequential_max_abs_diff"] = diff
        return avg


class Von10Nli:
    """von-sdk 1.1.1's other shipped backend: BertaBackend variant "von-1.0", an NLI cross-encoder
    (ModernBertForSequenceClassification, entailment/neutral/contradiction) loaded from the cached
    wfzyx/von@d8bb5e07 snapshot; each option is scored by its entailment logit."""
    name = "von10_nli"

    def __init__(self):
        began = time.perf_counter()
        torch_threads()
        from von.engine import VonEngine
        from von.types import Choice
        from von.backends import berta_backend as bb
        self.Choice, self.format_state = Choice, bb._format_state
        self.backend = VonEngine("von-1.0").backend
        assert type(self.backend).__name__ == "BertaBackend"
        self.backend._get_model_and_tok()
        self.load_ms = (time.perf_counter() - began) * 1000
        self.extra = {}

    def decide(self, state, options, instruction):
        ans = self.backend.evaluate_choice("next", self.format_state(state),
                                           self.Choice(instructions=instruction, criteria=options))
        self.extra = {"native_choice": ans.choice}
        return dict(ans.probabilities)


class S1OInstrFirst(S1O):
    """s1o with the constant instruction moved to the front of the prompt and only that prefix cached.

    Prompt: <bos>"Question (choice): {instruction}\\n" + "<state>\\n{state}\\n</state>\\n" +
    "\\nOptions:" + options rendered by s1.schema.render_option + "\\nAnswer: (". With S1O_CACHE=prefix,
    each decision first re-primes the slot with the prefix alone (n_predict 0), so the scored request
    reuses exactly the instruction prefix and nothing from the previous case or order.
    """
    name = "s1o_if"

    def __init__(self):
        self.cache = os.environ.get("S1O_CACHE", "off")
        assert self.cache in ("off", "prefix")
        super().__init__()
        from s1.schema import render_option
        self.render_option = render_option

    def split_prompt(self, state, options, instruction):
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        prefix = [self.bos_token_id] + self.encode(f"Question (choice): {instruction.strip()}\n")
        body = self.encode("<state>\n") + self.encode(state) + self.encode("\n</state>\n")
        body += self.encode("\nOptions:" + "".join("\n" + self.render_option(self.LETTERS[j], k, options[k])
                                                   for j, k in enumerate(options)) + "\nAnswer: (")
        return prefix, body

    def decide(self, state, options, instruction):
        prefix, body = self.split_prompt(state, options, instruction)
        if self.cache == "prefix":
            self._post("/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True})
        by_id = self.completion_probs(prefix + body, cache_prompt=self.cache == "prefix")
        raw = {k: by_id[self.letter_ids[j]] for j, k in enumerate(options)}
        mass = sum(raw.values())
        self.extra = {"prompt_tokens": len(prefix) + len(body), "prefix_tokens": len(prefix),
                      "letter_mass_full_vocab": mass, "cache": self.cache}
        return {k: v / mass for k, v in raw.items()}


CANDIDATES = {c.name: c for c in (Mock, Von11, LayaEn, LayaMulti, LayaMicro, S1O,
                                  LayaEnOnnx, Von11Perm, Von10Nli, S1OInstrFirst)}
