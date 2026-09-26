import json, os, sys, time, urllib.request
sys.path.insert(0, "/termux-home/robot/benchmark/strategic_selector/v3")
from adapters import S1O
c = json.loads(open(sys.argv[1]).read())
opts = c["options"] if isinstance(c["options"], dict) else {o: "" for o in c["options"]}
ad = S1O()
ids = ad.prompt_ids(c["situation"], opts, c.get("instruction", "Choose the single best next action for the robot."))
print("tokens", len(ids), "max id", max(ids))
for n in [int(x) for x in sys.argv[2].split(",")]:
    for psp in (False, True):
        try:
            p = ad.completion_probs(ids, n_probs=n, post_sampling_probs=psp)
            print("n_probs", n, "post", psp, "ok", len(p), flush=True)
        except Exception as e:
            time.sleep(1); print("n_probs", n, "post", psp, "FAIL", repr(e), ad.server.poll(), flush=True); raise SystemExit
ad.close()
