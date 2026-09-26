"""Diagnose qwen35_4b dying on A7: fresh server, A6 then A7 x3, report exit status and free memory."""
import json, sys, time, urllib.request
sys.path.insert(0, "/termux-home/ladder")
import conv_speed as cs
b = cs.bench
gguf, fam, think = cs.MODELS["qwen35_4b"]
mem = lambda: next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
log = open(sys.argv[1], "w")
proc, load = cs.start(gguf, log)
print(f"loaded {load:.1f}s MemAvailable {mem()} MiB", flush=True)
try:
    for pid in ("A6", "A7", "A7", "A7"):
        p = next(x for x in b.PROMPTS_A if x["id"] == pid)
        prompt, stop = b.format_prompt(b.SYS_CHAT, p["text"], fam, think)
        res = b.generate_completion(prompt, stop, 0.1)
        print(pid, "error" if "error" in res else "ok", res.get("error", repr(res.get("content", ""))[:80]),
              "| server exit", proc.poll(), "| MemAvailable", mem(), "MiB | VmRSS", b.get_vmrss(proc.pid) // 1024 if proc.poll() is None else "-", flush=True)
        if proc.poll() is not None:
            break
finally:
    if proc.poll() is None:
        cs.stop(proc)
