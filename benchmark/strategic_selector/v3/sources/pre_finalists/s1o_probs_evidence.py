#!/usr/bin/env python3
"""Item 8 evidence: are build 1609's n_probs pre- or post-sampling?

Same prompt (first dev case, filtered_text, canonical order) through the S1O
adapter's server, varying only sampler settings. If post_sampling_probs=false
reports the raw softmax, temperature/top_k/logit_bias must not change it, while
post_sampling_probs=true with top_k=2 must; post_sampling_probs=true with a
neutral chain (T=1, no truncation) must reproduce the raw softmax.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapters import S1O
from cases import load_v2

v2 = load_v2()
case = json.loads((HERE / "dev.jsonl").read_text().splitlines()[0])
options = v2.options_for(case, "filtered_text", "flat", random.Random(0))
instruction = ("Given the mission, evidence, room history and previous script result, "
               "choose the single most useful next high-level mission action. "
               "A travel script will handle movement and obstacles itself.")
s = S1O()
try:
    ids = s.prompt_ids(v2.to_state(case, False), options, instruction)
    letters = {s.LETTERS[j]: s.letter_ids[j] for j in range(len(options))}
    neutral = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
               "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
    runs = {
        "A_raw_default(T=0)": {},
        "B_raw_T1.7_topk2": {"temperature": 1.7, "top_k": 2},
        "C_raw_logit_bias_B+5": {"logit_bias": [[letters["B"], 5.0]]},
        "D_post_T1.7_topk2": {"post_sampling_probs": True, "temperature": 1.7, "top_k": 2},
        "E_post_neutral_chain": {"post_sampling_probs": True, **neutral},
    }
    got = {}
    for label, overrides in runs.items():
        probs = s.completion_probs(ids, **overrides)
        got[label] = {L: probs.get(t) for L, t in letters.items()}
        print(label, json.dumps({L: None if p is None else round(p, 6) for L, p in got[label].items()}))

    def maxdiff(a, b):
        return max(abs((got[a][L] or 0) - (got[b][L] or 0)) for L in letters)

    for other in ("B_raw_T1.7_topk2", "C_raw_logit_bias_B+5", "E_post_neutral_chain", "D_post_T1.7_topk2"):
        print(f"max|A - {other}| = {maxdiff('A_raw_default(T=0)', other):.6f}")
finally:
    print("server", s.close())
