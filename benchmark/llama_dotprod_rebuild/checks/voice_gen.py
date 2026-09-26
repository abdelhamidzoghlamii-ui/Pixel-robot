"""3b + 3c. Servers start through server_manager.start_server (the robot's exact flags, port 8080) with only
LLAMA_SERVER swapped; parse_command and PARSE_SYS are exec'd from main.py's source (main.py is not imported,
so no camera/detector modules load). Order: old, new, old, new (generation rounds alternate to spread drift)."""
import ast, json, statistics, sys, time
import requests
sys.path.insert(0, "/termux-home/robot")
import server_manager as sm

OLD = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-server"  # pinned: server_manager is edited later
NEW = "/data/data/com.termux/files/home/llama.cpp-b1609-dotprod/build/bin/llama-server"
MODEL = sm.HOME + "/models/gemma-4-e2b-it-q4_k_m.gguf"  # what setup_q4 starts
src = open("/termux-home/robot/main.py").read()
tree = ast.parse(src)
ns = {"requests": requests, "json": json, "GEMMA_URL": "http://127.0.0.1:8080/completion", "__builtins__": __builtins__}
for node in tree.body:
    if (isinstance(node, ast.Assign) and node.targets[0].id == "PARSE_SYS") or \
       (isinstance(node, ast.FunctionDef) and node.name == "parse_command"):
        exec(compile(ast.Module([node], []), "main.py", "exec"), ns)
raw = {}
real_post = requests.post
def post(url, json=None, **kw):  # capture the model's raw text next to parse_command's result
    r = real_post(url, json=json, **kw)
    raw["content"] = r.json().get("content")
    return r
ns["requests"] = type("R", (), {"post": staticmethod(post)})

COMMANDS = [
    "Find Chiara and tell her the pizza is here",
    "Go to the bathroom",
    "Patrol the whole apartment",
    "Find my phone in the living room",
    "Come back",
    "Tell Abdel the meeting starts in ten minutes",
    "Go to the kitchen and say lunch is ready",
    "Check the bedroom and the hallway",
    "Where are my glasses",
    "Say good night",
    "Find someone and tell them the door is open",
    "Go to the living room then come back",
    "Look for the cat in the bedroom",
    "Tell Chiara I love her",
    "Blue banana seven",
]
GEN = {"prompt": "<start_of_turn>user\nTell me a long story about a small robot exploring a quiet house at night."
                 "<end_of_turn>\n<start_of_turn>model\n",
       "n_predict": 200, "temperature": 0, "ignore_eos": True, "cache_prompt": False}

def run(binary, parse):
    sm.LLAMA_SERVER = binary
    assert sm.start_setup("setup_q4"), "server did not start"
    out = {"parse": [], "gen": []}
    if parse:
        for c in COMMANDS:
            t = time.perf_counter()
            actions = ns["parse_command"](c)
            out["parse"].append({"cmd": c, "actions": actions, "raw": raw.get("content"),
                                 "ms": round((time.perf_counter() - t) * 1000)})
    for _ in range(3):
        tm = requests.post("http://127.0.0.1:8080/completion", json=GEN, timeout=300).json()["timings"]
        out["gen"].append({"n": tm["predicted_n"], "tok_s": tm["predicted_per_second"], "prompt_tok_s": tm["prompt_per_second"]})
    sm.kill_servers()
    return out

res = {"old_1": run(OLD, True), "new_1": run(NEW, True), "old_2": run(OLD, False), "new_2": run(NEW, False)}
json.dump(res, open(sys.argv[1], "w"), indent=1)
for i, c in enumerate(COMMANDS):
    o, n = res["old_1"]["parse"][i], res["new_1"]["parse"][i]
    print(f"{i + 1:2}. {c!r}  {'SAME' if o['actions'] == n['actions'] and o['raw'] == n['raw'] else 'DIFFERENT'}"
          f"  ({o['ms']} ms old / {n['ms']} ms new)")
    print(f"    old: {json.dumps(o['actions'])}")
    print(f"    new: {json.dumps(n['actions'])}")
for b in ("old", "new"):
    g = res[f"{b}_1"]["gen"] + res[f"{b}_2"]["gen"]
    print(f"{b}: generation {[round(x['tok_s'], 2) for x in g]} tok/s (n={[x['n'] for x in g]}), median "
          f"{statistics.median(x['tok_s'] for x in g):.2f}; prompt eval median {statistics.median(x['prompt_tok_s'] for x in g):.1f} tok/s")
