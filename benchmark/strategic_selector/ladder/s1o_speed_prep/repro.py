import json, os, sys, time
sys.path.insert(0, "/termux-home/robot/benchmark/strategic_selector/v3")
from adapters import S1O
cases = [json.loads(l) for l in open(sys.argv[1])]
ad = S1O()
try:
    for c in cases:
        opts = c["options"] if isinstance(c["options"], dict) else {o: "" for o in c["options"]}
        for keys in (list(opts), list(opts)[::-1]):
            t = time.perf_counter()
            try:
                d = ad.decide(c["situation"], {k: opts[k] for k in keys}, c.get("instruction", "Choose the single best next action for the robot."))
            except Exception as e:
                time.sleep(1)
                print("FAIL", c["id"], repr(e), "server exit", ad.server.poll(), flush=True); raise SystemExit(1)
            print(c["id"], max(d, key=d.get), f"{(time.perf_counter()-t)*1000:.0f} ms", ad.extra.get("prompt_tokens"), flush=True)
finally:
    if ad.server.poll() is None: ad.close()
