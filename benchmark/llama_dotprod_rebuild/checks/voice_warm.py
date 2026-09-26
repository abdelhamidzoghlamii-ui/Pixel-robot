"""3b follow-up: commands 1-2 with the server's prompt cache warmed by one discarded parse (the old binary's
cold first parse exceeds parse_command's 20 s timeout). Everything else as voice_gen.py."""
import json, sys, time
sys.argv = [sys.argv[0], "/dev/null"]
src = open(__file__.replace("voice_warm.py", "voice_gen.py")).read()
src = src[:src.index("res = {")]  # reuse its setup: server start, parse_command from main.py, capture
exec(src)
out = {}
for label, binary in (("old", OLD), ("new", NEW)):
    sm.LLAMA_SERVER = binary
    assert sm.start_setup("setup_q4")
    t = time.perf_counter(); ns["parse_command"]("Come back"); warm = round(time.perf_counter() - t, 1)
    out[label] = {"warmup_s": warm, "results": []}
    for c in COMMANDS[:2]:
        t = time.perf_counter()
        a = ns["parse_command"](c)
        out[label]["results"].append({"cmd": c, "actions": a, "raw": raw.get("content"), "ms": round((time.perf_counter() - t) * 1000)})
    sm.kill_servers()
for label in out:
    print(label, "warm-up parse", out[label]["warmup_s"], "s")
for i in range(2):
    o, n = out["old"]["results"][i], out["new"]["results"][i]
    print(f"{i + 1}. {o['cmd']!r}  {'SAME' if o['actions'] == n['actions'] and o['raw'] == n['raw'] else 'DIFFERENT'} ({o['ms']} / {n['ms']} ms)")
    print("   old:", json.dumps(o["actions"])); print("   new:", json.dumps(n["actions"]))
