"""Scratch diagnostic: same prompt, uncached x2 and prefix-cached x2, one server."""
import json, random, sys
sys.path.insert(0, "/termux-home/robot/benchmark/strategic_selector/v3")
from adapters import S1OInstrFirst
from cases import load_v2
v2 = load_v2()
case = next(json.loads(l) for l in open("/termux-home/robot/benchmark/strategic_selector/v3/dev.jsonl") if '"dev_repeat_search_5"' in l)
opts = v2.render_options(v2.options_for(case, "filtered_text", "flat", None), "reverse", None)
ins = ("Given the mission, evidence, room history and previous script result, choose the single most useful next "
       "high-level mission action. A travel script will handle movement and obstacles itself.")
s = S1OInstrFirst()
try:
    prefix, body = s.split_prompt(v2.to_state(case, False), opts, ins)
    L = [s.letter_ids[j] for j in range(len(opts))]
    def dist(cache):
        if cache:
            s._post("/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True})
        p = s.completion_probs(prefix + body, cache_prompt=cache)
        return [round(p[t], 5) for t in L]
    for label, cache in (("uncached_1", False), ("uncached_2", False), ("prefix_1", True), ("prefix_2", True)):
        print(label, dist(cache), flush=True)
finally:
    print("server", s.close())
