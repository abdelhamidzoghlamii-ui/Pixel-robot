"""Diagnose qwen35_4b dying after ~25 bench turns: replay bench's turn order (A1.. x3 runs) on server_manager flags
and sample memory each turn. Stops at MemAvailable < 1000 MiB or after 20 turns (before the low-memory killer)."""
import sys, time
sys.path.insert(0, "/termux-home/ladder")
import conv_speed as cs
b = cs.bench
extra = sys.argv[2:]
gguf, fam, think = cs.MODELS["qwen35_4b"]
field = lambda pid, k: next(int(l.split()[1]) // 1024 for l in open(f"/proc/{pid}/status") if l.startswith(k + ":"))
avail = lambda: next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
cmd = cs.server_cmd
cs.server_cmd = lambda g, port=cs.PORT: cmd(g, port) + extra
proc, load = cs.start(gguf, open(sys.argv[1], "w"))
print(f"extra args {extra}; loaded; MemAvailable {avail()} MiB; RssAnon {field(proc.pid, 'RssAnon')} RssFile {field(proc.pid, 'RssFile')}", flush=True)
n = 0
try:
    for p in b.PROMPTS_A + b.PROMPTS_B:
        for run in range(3):
            text = [{"user": t["user"]} for t in p["text"]] if p.get("multi") else p["text"]
            prompt, stop = b.format_prompt(b.SYS_CHAT, text, fam, think)
            res = b.generate_completion(prompt, stop, 0.1)
            n += 1
            alive = proc.poll() is None
            print(f"{n:2} {p['id']} r{run + 1} {'ok ' if 'error' not in res else 'ERR'} MemAvailable {avail():5} MiB  server "
                  + (f"RssAnon {field(proc.pid, 'RssAnon'):5} RssFile {field(proc.pid, 'RssFile'):5}" if alive else f"EXITED {proc.poll()}"), flush=True)
            if not alive or avail() < 1000 or n >= 20:
                raise StopIteration
except StopIteration:
    pass
finally:
    if proc.poll() is None:
        cs.stop(proc)
