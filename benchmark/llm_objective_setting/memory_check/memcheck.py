#!/usr/bin/env python3
"""Server memory per turn: llama-server with server_manager.py's flags, with or without --cache-ram 0.

Starts one server, sends N unique robot voice commands in main.py's parse_command() prompt shape
(PARSE_SYS read from main.py source, not imported), or with --shape conv N unique chat questions with
no shared system prefix (the conversation role), and after every turn records RssAnon and VmRSS from
/proc/<pid>/status plus the server's own timings. Stops the server at the end. No motors, camera or
network beyond 127.0.0.1.

  python3 memcheck.py OUTDIR [--cache-ram-0] [--turns 40] [--shape robot|conv]
"""
import ast
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = "/data/data/com.termux/files/home"
SERVER = HOME + "/llama.cpp-b1609-dotprod/build/bin/llama-server"
MODEL = HOME + "/models/gemma-4-E2B-it-Q4_0.gguf"
MAIN_PY = HOME + "/robot/main.py"
THERMAL = HOME + "/ladder/thermal.log"  # written every 5 s by a native-Termux root logger
PORT = 8080
# server_manager.start_server() flags, verbatim
FLAGS = ["--port", str(PORT), "--ctx-size", "2048", "--threads", "4", "--threads-batch", "4",
         "--parallel", "1", "--swa-full", "--host", "127.0.0.1"]

SUBJECTS = ["Chiara", "Abdel", "someone", "the kids", "Mama", "the guest", "Luca", "Sara"]
ROOMS = ["kitchen", "bedroom", "bathroom", "living room", "hallway"]
OBJECTS = ["keys", "phone", "remote", "glasses", "wallet", "charger", "book", "cup"]


def commands(n):
    out = []
    for i in range(n):
        s, r, o = SUBJECTS[i % 8], ROOMS[i % 5], OBJECTS[(i * 3) % 8]
        out.append([f"Find {s} and tell them dinner is ready in {i + 2} minutes",
                    f"Go to the {r} and look for my {o}, item {i}",
                    f"Say hello number {i} to everyone in the {r}",
                    f"Patrol the {r} and the {ROOMS[(i + 2) % 5]} then come back, round {i}"][i % 4])
    assert len(set(out)) == n
    return out


TOPICS = ["the moon", "bread", "rain", "bees", "trains", "volcanoes", "chess", "coffee", "tides", "owls"]


def questions(n):
    out = [f"{['Tell me', 'Explain', 'Give me', 'Describe'][i % 4]} one short interesting fact about "
           f"{TOPICS[i % 10]}, in {i % 3 + 2} sentences, for question {i}." for i in range(n)]
    assert len(set(out)) == n
    return out


def parse_sys():
    for node in ast.parse(Path(MAIN_PY).read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PARSE_SYS":
            return ast.literal_eval(node.value)
    raise SystemExit("PARSE_SYS not found in main.py")


def mem(pid):
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("RssAnon", "VmRSS", "RssFile"):
            fields[k + "_kB"] = int(v.split()[0])
    return fields


def thermal():
    try:
        return Path(THERMAL).read_text().splitlines()[-1]
    except (OSError, IndexError):
        return None


def post(path, body, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    out = Path(sys.argv[1])
    cache_ram0 = "--cache-ram-0" in sys.argv
    turns = int(sys.argv[sys.argv.index("--turns") + 1]) if "--turns" in sys.argv else 40
    shape = sys.argv[sys.argv.index("--shape") + 1] if "--shape" in sys.argv else "robot"
    assert shape in ("robot", "conv")
    out.mkdir(parents=True, exist_ok=False)
    cmd = [SERVER, "-m", MODEL, *FLAGS] + (["--cache-ram", "0"] if cache_ram0 else [])
    meta = {"cmd": cmd, "cache_ram_0": cache_ram0, "turns": turns, "shape": shape,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "main_py_sha256": hashlib.sha256(Path(MAIN_PY).read_bytes()).hexdigest(),
            "cpus_allowed": [l for l in Path("/proc/self/status").read_text().splitlines()
                             if l.startswith("Cpus_allowed_list")][0].split(":")[1].strip(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    time.sleep(2)
    log = open(out / "server.log", "w")
    began = time.perf_counter()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            time.sleep(1)
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                    break
            except Exception:
                pass
        else:
            raise SystemExit("server did not become healthy in 180 s")
        rows = [{"turn": 0, "load_s": round(time.perf_counter() - began, 1), **mem(srv.pid),
                 "thermal": thermal()}]
        print(json.dumps(rows[0]), flush=True)
        sys_text = parse_sys()
        for i, text in enumerate(commands(turns) if shape == "robot" else questions(turns), 1):
            if shape == "robot":
                prompt = ('<start_of_turn>user\n' + sys_text + '\nCommand: "' + text + '"'
                          '<end_of_turn>\n<start_of_turn>model\n[')
                body = {"n_predict": 300, "temperature": 0.05, "stop": ["<end_of_turn>", "\n\n"]}
            else:
                prompt = '<start_of_turn>user\n' + text + '<end_of_turn>\n<start_of_turn>model\n'
                body = {"n_predict": 128, "temperature": 0.1, "stop": ["<end_of_turn>"]}
            t0 = time.perf_counter()
            r = post("/completion", {"prompt": prompt, **body})
            wall = (time.perf_counter() - t0) * 1000
            t = r.get("timings", {})
            row = {"turn": i, "command": text, "wall_ms": round(wall, 1),
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                   "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"),
                   "predicted_per_second": t.get("predicted_per_second"),
                   **mem(srv.pid), "thermal": thermal(), "reply": ("[" if shape == "robot" else "") + r.get("content", "")}
            rows.append(row)
            print(json.dumps(row), flush=True)
        (out / "turns.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    finally:
        srv.terminate()
        try:
            srv.wait(20)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()


if __name__ == "__main__":
    main()
