import urllib.request
import json
import subprocess
import time
import os
import statistics
import re
import random
import argparse
import hashlib
from datetime import datetime

# --- CONFIGURATION ---
BIN_PATH = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
PORT = 8080
HOME = os.path.expanduser("~")

MODELS = {
    "gemma-e2b": {"path": "~/models/gemma-4-e2b-it-q4_k_m.gguf", "family": "gemma", "think": None},
    "gemma-e4b": {"path": "~/models/gemma-4-e4b-it-q4_k_m.gguf", "family": "gemma", "think": None},
    "qwen2.5-3b": {"path": "~/models/qwen2.5-3b-instruct-q4_k_m.gguf", "family": "chatml", "think": None},
    "qwen3.5-2B-think-on": {"path": "~/models/qwen35/Qwen3.5-2B-Q4_K_M.gguf", "family": "chatml", "think": "ON"},
    "qwen3.5-2B-think-off": {"path": "~/models/qwen35/Qwen3.5-2B-Q4_K_M.gguf", "family": "chatml", "think": "OFF"},
    "qwen3.5-4B-think-on": {"path": "~/models/qwen35/Qwen3.5-4B-Q4_K_M.gguf", "family": "chatml", "think": "ON"},
    "qwen3.5-4B-think-off": {"path": "~/models/qwen35/Qwen3.5-4B-Q4_K_M.gguf", "family": "chatml", "think": "OFF"},
}

SYS_CHAT = "You are the voice of a small home robot. You speak your replies aloud, so be brief and natural — at most two short sentences unless asked for more. Answer in the same language the person used. If you do not know something or cannot sense it, say so plainly instead of guessing."
SYS_PARSE = """You convert spoken commands into robot actions. Output ONLY a JSON array, no prose.
CURRENT MAP ROOMS: kitchen, living_room, schlafzimmer, bureau, bathroom, balkon
Use ONLY a label from that list for any room field. Never invent a room.
Allowed actions:
{"type":"navigate_to","room":<label>}
{"type":"find_person","name":<string>,"message":<string>}
{"type":"find_object","object":<string>,"room":<label>}
{"type":"patrol","rooms":[<labels> or empty for all]}
{"type":"come_back"}
{"type":"say","message":<string>}
If the command is unclear, impossible, negated, or names a place not on the map, return [] or a single say action explaining why. Use the minimum number of actions."""

MAP_ROOMS = {"kitchen", "living_room", "schlafzimmer", "bureau", "bathroom", "balkon"}

PROMPTS_A = [
    {"id": "A1", "lang": "en", "text": "Hey, you awake?"},
    {"id": "A2", "lang": "de", "text": "Wie geht's dir heute?"},
    {"id": "A3", "lang": "fr", "text": "Tu peux m'expliquer ce que tu fais quand tu te perds ?"},
    {"id": "A4", "lang": "ar", "text": "من أنت وماذا تستطيع أن تفعل؟"},
    {"id": "A5", "lang": "en", "text": [{"user": "I'm looking for something."}, {"user": "It's small and black."}, {"user": "So where should I start?"}], "multi": True},
    {"id": "A6", "lang": "en", "text": "I've had a rough day."},
    {"id": "A7", "lang": "de", "text": "Fahr sofort los und such Chiara."}
]

PROMPTS_B = [
    {"id": "B1", "lang": "en", "text": "How far can you see?"},
    {"id": "B2", "lang": "en", "text": "Where are my keys?", "trap": True},
    {"id": "B3", "lang": "de", "text": "Warum bleibst du stehen, wenn etwas vor dir ist?"},
    {"id": "B4", "lang": "fr", "text": "Combien de roues as-tu et comment tournes-tu sur place ?"},
    {"id": "B5", "lang": "ar", "text": "هل تستطيع أن ترى في الظلام؟"},
    {"id": "B6", "lang": "en", "text": "What's the battery at?", "trap": True},
    {"id": "B7", "lang": "en", "text": "Explain in one sentence why you stopped."}
]

PROMPTS_C = [
    {"id": "C1", "lang": "en", "text": "Go to the bedroom", "expected": {"type": "navigate_to", "room": "schlafzimmer"}},
    {"id": "C2", "lang": "fr", "text": "Va au bureau", "expected": {"type": "navigate_to", "room": "bureau"}},
    {"id": "C3", "lang": "en", "text": "Go to the office", "expected": {"type": "navigate_to", "room": "bureau"}},
    {"id": "C4", "lang": "de", "text": "Geh ins Wohnzimmer", "expected": {"type": "navigate_to", "room": "living_room"}},
    {"id": "C5", "lang": "en", "text": "Go to the balcony", "expected": {"type": "navigate_to", "room": "balkon"}},
    {"id": "C6", "lang": "en", "text": "Go to the garage", "expected": "REJECT"},
    {"id": "C7", "lang": "en", "text": "Go to Chiara's room", "expected": "REJECT"},
    {"id": "C8", "lang": "en", "text": "Don't go to the kitchen", "expected": "REJECT"},
    {"id": "C9", "lang": "en", "text": "Look for my phone in the kitchen", "expected": {"type": "find_object", "object": "phone", "room": "kitchen"}},
    {"id": "C10", "lang": "en", "text": "Patrol everywhere", "expected": {"type": "patrol", "rooms": []}},
    {"id": "C11", "lang": "en", "text": "Find Abdel", "expected": {"type": "find_person", "name": "Abdel"}},
    {"id": "C12", "lang": "en", "text": "Come back", "expected": {"type": "come_back"}},
    {"id": "C13", "lang": "de", "text": "Sag Chiara, dass das Essen fertig ist", "expected": {"type": "find_person", "name": "Chiara", "message": "NONEMPTY"}}
]

# --- SYSTEM UTILS ---
def run_su_cmd(cmd):
    try:
        res = subprocess.run(["su", "-c", cmd], capture_output=True, text=True, timeout=5)
        return res.stdout.strip()
    except Exception as e:
        return ""

def get_zone9_temp():
    val = run_su_cmd("cat /sys/class/thermal/thermal_zone9/temp")
    try:
        return int(val)
    except:
        return -1

def drop_caches():
    run_su_cmd("echo 3 > /proc/sys/vm/drop_caches")

def get_mem_available():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except:
        pass
    return -1

def get_vmrss(pid):
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except:
        pass
    return -1

# --- SERVER MANAGEMENT ---
def start_server(config_name, log_file):
    cfg = MODELS[config_name]
    model_path = os.path.expanduser(cfg["path"])
    cmd = [BIN_PATH, "-m", model_path, "--port", str(PORT), "-c", "4096", "-t", "4", "--parallel", "1"]
    if cfg["family"] == "gemma":
        cmd.append("--swa-full")
    
    print(f"Starting server for {config_name}: {' '.join(cmd)}")
    with open(log_file, "a") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f)
    
    # Poll for health
    start_time = time.time()
    ready = False
    while time.time() - start_time < 150:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(2)
        
    if not ready:
        print(f"Failed to start server for {config_name}. Last 8 log lines:")
        try:
            res = subprocess.run(["tail", "-n", "8", log_file], capture_output=True, text=True)
            print(res.stdout)
        except:
            pass
        return None
    return proc

def stop_server():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

def generate_completion(prompt, stop, temp=0.1):
    req_data = {
        "prompt": prompt,
        "n_predict": 512,
        "temperature": temp,
        "stop": stop
    }
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                                 data=json.dumps(req_data).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            data["latency_ms"] = (time.time() - t0) * 1000
            return data
    except Exception as e:
        return {"error": str(e)}

# --- PROMPT FORMATTING ---
def format_prompt(sys_msg, text, family, think=None):
    nonce = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=12))
    sys_with_nonce = f"[{nonce}] {sys_msg}"
    
    if family == "gemma":
        if isinstance(text, list):
            prompt = f"<start_of_turn>user\n{sys_with_nonce}\n\n"
            for t in text[:-1]:
                prompt += f"{t['user']}<end_of_turn>\n<start_of_turn>model\n{t.get('assistant', '')}<end_of_turn>\n<start_of_turn>user\n"
            prompt += f"{text[-1]['user']}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"<start_of_turn>user\n{sys_with_nonce}\n\n{text}<end_of_turn>\n<start_of_turn>model\n"
        return prompt, ["<end_of_turn>"]
        
    elif family == "chatml":
        if isinstance(text, list):
            prompt = f"<|im_start|>system\n{sys_with_nonce}<|im_end|>\n"
            for t in text[:-1]:
                prompt += f"<|im_start|>user\n{t['user']}<|im_end|>\n<|im_start|>assistant\n{t.get('assistant', '')}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{text[-1]['user']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\n{sys_with_nonce}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        
        if think == "OFF":
            prompt += "<think>\n\n</think>\n\n"
            
        return prompt, ["<|im_end|>", "<|endoftext|>"]

def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    best = None
    i = 0
    while i < len(text):
        if text[i] == "[":
            depth = 0
            in_str = False
            esc = False
            start = i
            for j in range(i, len(text)):
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:j+1]
                        try:
                            json.loads(cand)
                            best = cand
                            i = j
                        except Exception:
                            pass
                        break
        i += 1
    return best if best is not None else text

def parse_and_grade_c(reply_clean, expected):
    try:
        reply_json = json.loads(extract_json(reply_clean))
        if not isinstance(reply_json, list):
            return False, False, False, False # parse_ok, exact, critical_failure, reject_violation
    except:
        return False, False, False, False

    # GUARD: a parseable array may contain non-dict elements (e.g. ["bedroom"]).
    # Filter to dicts before any .get(); wrong-shape is a FORMATTING failure
    # (exact=False) not a room-hallucination hazard (crit stays as scanned).
    malformed = any(not isinstance(a, dict) for a in reply_json)
    acts = [a for a in reply_json if isinstance(a, dict)]

    crit_fail = False
    for act in acts:
        if act.get("type") == "navigate_to" and act.get("room") not in MAP_ROOMS:
            crit_fail = True

    reject_violation = False
    if expected == "REJECT":
        has_nav_or_find = any(act.get("type") in ["navigate_to", "find_object"] for act in acts)
        if has_nav_or_find:
            reject_violation = True
        return True, (not has_nav_or_find and not malformed), crit_fail, reject_violation

    if not acts:
        return True, False, crit_fail, reject_violation # parsed, no usable dict actions

    match = False
    for act in acts:
        if expected["type"] == "navigate_to":
            if act.get("type") == "navigate_to" and act.get("room") == expected["room"]:
                match = True
        elif expected["type"] == "find_object":
            if act.get("type") == "find_object" and act.get("object", "").lower() == expected["object"].lower() and act.get("room") == expected["room"]:
                match = True
        elif expected["type"] == "patrol":
            if act.get("type") == "patrol" and (act.get("rooms") == [] or set(act.get("rooms") or []) == MAP_ROOMS):
                match = True
        elif expected["type"] == "come_back":
            if act.get("type") == "come_back":
                match = True
        elif expected["type"] == "find_person":
            if act.get("type") == "find_person" and act.get("name", "").lower() == expected["name"].lower():
                if "message" in expected and expected["message"] == "NONEMPTY":
                    if bool(act.get("message")):
                        match = True
                else:
                    match = True

    return True, (match and not malformed), crit_fail, False

def run_self_check(config_name, family, think_mode):
    if not think_mode: return True, None
    prompt, stop = format_prompt("You are a helpful assistant.", "What is 1+1?", family, think_mode)
    res = generate_completion(prompt, stop, temp=0.1)
    if "error" in res: return False, None
    content = res.get("content", "")
    
    if think_mode == "ON":
        if "<think>" not in content:
            print(f"WARNING: Self-check failed for {config_name}. think=ON but no <think> found.")
            return False, None
    elif think_mode == "OFF":
        if "<think>" in content:
            print(f"WARNING: Self-check failed for {config_name}. think=OFF but <think> found. Trying fallback '/no_think'...")
            prompt, stop = format_prompt("You are a helpful assistant.", "What is 1+1?\n/no_think", family, think_mode)
            res = generate_completion(prompt, stop, temp=0.1)
            content = res.get("content", "")
            if "<think>" not in content:
                print(f"SUCCESS: Fallback '/no_think' worked for {config_name}.")
                return True, "/no_think"
            else:
                print(f"WARNING: Fallback '/no_think' also failed for {config_name}.")
                return False, None
    return True, None

def do_run(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(HOME, f"bench_results_{ts}.json")
    speed_file = os.path.join(HOME, f"bench_speed_summary.txt")
    c_summary_file = os.path.join(HOME, f"bench_c_summary.txt")
    blind_transcripts_file = os.path.join(HOME, f"bench_blind_transcripts.txt")
    blind_key_file = os.path.join(HOME, f"bench_blind_key.txt")
    
    results = []
    
    configs_to_run = args.configs if args.configs else list(MODELS.keys())
    
    # Shuffle for blind grading
    blind_key = {}
    config_ids = [f"M{i+1}" for i in range(len(configs_to_run))]
    random.shuffle(config_ids)
    for c, cid in zip(configs_to_run, config_ids):
        blind_key[c] = cid
        
    with open(blind_key_file, "w") as f:
        for c, cid in blind_key.items():
            f.write(f"{cid}: {c}\n")
            
    blind_data = {} # prompt_id -> {config_id -> reply}

    if args.dry_run:
        print(f"Dry run. Would test configs: {configs_to_run}")
        return

    for config_name in configs_to_run:
        cfg = MODELS[config_name]
        log_file = os.path.join(HOME, f"bench_{config_name}_{ts}.log")
        
        pre_temp = get_zone9_temp()
        
        proc = start_server(config_name, log_file)
        if not proc: continue
        
        time.sleep(2)
        pid = proc.pid
        
        mem_avail = get_mem_available()
        vmrss = get_vmrss(pid)
        
        think_verified, fallback = run_self_check(config_name, cfg["family"], cfg["think"])
        
        for bucket, prompts, sys_msg, temp in [("A", PROMPTS_A, SYS_CHAT, 0.1), ("B", PROMPTS_B, SYS_CHAT, 0.1), ("C", PROMPTS_C, SYS_PARSE, 0.05)]:
            for p in prompts:
                for run_idx in range(args.runs):
                    text_input = p["text"]
                    if fallback == "/no_think":
                        if isinstance(text_input, list):
                            text_input = [dict(t) for t in text_input]
                            text_input[-1]["user"] += "\n/no_think"
                        else:
                            text_input = text_input + "\n/no_think"
                            
                    # Handle multi-turn for A5
                    if p.get("multi"):
                        # To keep it self-contained and simple, we'll run the multi-turn sequence
                        # But wait, the spec says "run all prompts". For A5, we can just run it once to get the final reply,
                        # but we need the intermediate assistant replies.
                        # Simplification: we'll simulate the intermediate replies or generate them.
                        # Actually, we need to generate t1, feed it back, generate t2, feed it back, generate t3.
                        pass # handled below
                        
                    prompt_str, stop_tokens = format_prompt(sys_msg, text_input, cfg["family"], cfg["think"])
                    
                    if p.get("multi"):
                        # We need to build the multi-turn history
                        history = []
                        for turn in p["text"]:
                            history.append({"user": turn["user"]})
                            p_str, s_tok = format_prompt(sys_msg, history, cfg["family"], cfg["think"])
                            res = generate_completion(p_str, s_tok, temp)
                            if "error" in res:
                                history[-1]["assistant"] = ""
                                continue
                            history[-1]["assistant"] = strip_think(res.get("content", ""))
                        # The final result is the last turn's result
                        prompt_str = p_str
                        
                    else:
                        res = generate_completion(prompt_str, stop_tokens, temp)
                        
                    if "error" in res:
                        print(f"Error on {config_name} {p['id']}: {res['error']}")
                        results.append({
                            "config": config_name, "model": cfg["path"], "build": BIN_PATH,
                            "bucket": bucket, "prompt_id": p["id"], "lang": p["lang"], "run_index": run_idx + 1,
                            "error": res["error"]
                        })
                        continue
                        
                    content_raw = res.get("content", "")
                    content_clean = strip_think(content_raw)
                    
                    # Calculate timings
                    timings = res.get("timings", {})
                    prompt_n = timings.get("prompt_n", 0)
                    predicted_n = timings.get("predicted_n", 0)
                    prompt_ms = timings.get("prompt_ms", 0)
                    prompt_per_second = timings.get("prompt_per_second", 0)
                    predicted_per_second = timings.get("predicted_per_second", 0)
                    
                    turn_res = {
                        "config": config_name, "model": cfg["path"], "build": BIN_PATH,
                        "bucket": bucket, "prompt_id": p["id"], "lang": p["lang"], "run_index": run_idx + 1,
                        "reply_raw": content_raw, "reply_clean": content_clean,
                        "timings": timings, "latency_ms": res.get("latency_ms", 0),
                        "stop_type": res.get("stop_type"),
                        "stopped_eos": res.get("stopped_eos"),
                        "stopped_limit": res.get("stopped_limit"),
                        "truncated": (res.get("stop_type") == "limit"),
                        "prompt_n": prompt_n, "predicted_n": predicted_n,
                        "think_mode": cfg["think"], "think_mode_verified": think_verified,
                        "think_fallback": fallback,
                        "think_chars": len(content_raw) - len(content_clean),
                        "zone9_pre": pre_temp, "zone9_post": get_zone9_temp(),
                        "MemAvailable": mem_avail, "VmRSS": vmrss,
                        "prompt_per_second": prompt_per_second,
                        "predicted_per_second": predicted_per_second,
                        "trap": p.get("trap", False)
                    }
                    
                    if bucket == "C":
                        parse_ok, exact, crit_fail, reject_viol = parse_and_grade_c(content_clean, p["expected"])
                        turn_res["C_parse_ok"] = parse_ok
                        turn_res["C_exact"] = exact
                        turn_res["C_critical_failure"] = crit_fail
                        turn_res["C_reject_violation"] = reject_viol
                        if crit_fail:
                            print(f"CRITICAL FAILURE: {config_name} on {p['id']}! Replied: {content_clean}")
                        if reject_viol:
                            print(f"REJECT VIOLATION: {config_name} on {p['id']}! Replied: {content_clean}")
                            
                    results.append(turn_res)
                    
                    if run_idx == 0 and bucket in ["A", "B"]:
                        if p["id"] not in blind_data: blind_data[p["id"]] = {}
                        blind_data[p["id"]][blind_key[config_name]] = content_clean

        stop_server()
        
        # Incremental save
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        # Rest and cooldown
        drop_caches()
        print(f"Cooling down after {config_name}...")
        start_cool = time.time()
        while time.time() - start_cool < 300:
            z9 = get_zone9_temp()
            if z9 != -1 and z9 <= 45000: # assuming milli-C
                break
            time.sleep(5)
            
    # Summaries
    def get_stats(data):
        if not data: return 0, 0, 0
        return statistics.median(data), min(data), max(data)

    with open(speed_file, "w") as f:
        f.write("config | decode tok/s (total) (med/min/max) | decode tok/s (weighted) | prefill tok/s (med/min/max) | TTFT (ms) (med/min/max) | turn latency (ms) (med/min/max) | truncs | think chars (med/min/max)\n")
        f.write("-" * 180 + "\n")
        for c in configs_to_run:
            c_res = [r for r in results if r.get("config") == c and "error" not in r]
            if not c_res: continue
            
            rate_rows = [r for r in c_res if r.get("predicted_n", 0) >= 10 and r.get("predicted_per_second")]
            if not rate_rows:
                rate_rows = [r for r in c_res if r.get("predicted_per_second")]
            dec_tot = get_stats([r["predicted_per_second"] for r in rate_rows])
            _tok = sum(r.get("predicted_n", 0) for r in rate_rows)
            _ms = sum(r["timings"].get("predicted_ms", 0) for r in rate_rows if r.get("timings"))
            dec_wt = (_tok / _ms * 1000) if _ms > 0 else 0
            pref_tok = get_stats([r["prompt_per_second"] for r in c_res if r.get("prompt_per_second")])
            ttft = get_stats([r["timings"].get("prompt_ms", 0) for r in c_res if r.get("timings")])
            lat = get_stats([r["latency_ms"] for r in c_res if r.get("latency_ms")])
            think_c = get_stats([r.get("think_chars", 0) for r in c_res])
            truncs = sum(1 for r in c_res if r.get("stop_type") == "limit")
            
            f.write(f"{c: <20} | {dec_tot[0]:.2f}/{dec_tot[1]:.2f}/{dec_tot[2]:.2f} | {dec_wt:.2f} | {pref_tok[0]:.2f}/{pref_tok[1]:.2f}/{pref_tok[2]:.2f} | {ttft[0]:.2f}/{ttft[1]:.2f}/{ttft[2]:.2f} | {lat[0]:.2f}/{lat[1]:.2f}/{lat[2]:.2f} | {truncs} | {think_c[0]:.0f}/{think_c[1]:.0f}/{think_c[2]:.0f}\n")
            
    with open(c_summary_file, "w") as f:
        f.write("config | parse_ok % | exact % | cross_lingual % | critical_failures | reject_violations\n")
        f.write("-" * 100 + "\n")
        for c in configs_to_run:
            c_res = [r for r in results if r.get("config") == c and r.get("bucket") == "C" and "error" not in r]
            if not c_res: continue
            
            # Using run 1 only for grading metrics to keep it standard
            c_res_run1 = [r for r in c_res if r["run_index"] == 1]
            if not c_res_run1: continue
            
            parse_ok = sum(1 for r in c_res_run1 if r.get("C_parse_ok")) / len(c_res_run1) * 100
            exact = sum(1 for r in c_res_run1 if r.get("C_exact")) / len(c_res_run1) * 100
            crit = sum(1 for r in c_res_run1 if r.get("C_critical_failure"))
            rej_viol = sum(1 for r in c_res_run1 if r.get("C_reject_violation"))
            
            cross = [r for r in c_res_run1 if r["prompt_id"] in ["C1", "C2", "C3", "C4", "C5", "C13"]]
            cross_ok = sum(1 for r in cross if r.get("C_exact")) / len(cross) * 100 if cross else 0
            
            f.write(f"{c: <20} | {parse_ok: .1f}% | {exact: .1f}% | {cross_ok: .1f}% | {crit} | {rej_viol}\n")
            
        f.write("\n" + "=" * 80 + "\nPER-CASE PASS/FAIL TABLE (RUN 1 ONLY)\n" + "=" * 80 + "\n")
        for c in configs_to_run:
            c_res_run1 = [r for r in results if r.get("config") == c and r.get("bucket") == "C" and "error" not in r and r["run_index"] == 1]
            if not c_res_run1: continue
            
            f.write(f"\n[{c}]\n")
            for r in sorted(c_res_run1, key=lambda x: int(x["prompt_id"].replace("C", ""))):
                parse_str = 'OK' if r.get("C_parse_ok") else 'FAIL'
                exact_str = 'OK' if r.get("C_exact") else 'FAIL'
                crit_str = 'YES' if r.get("C_critical_failure") else 'NO'
                rej_str = 'YES' if r.get("C_reject_violation") else 'NO'
                f.write(f"  {r['prompt_id']: <4}: Parse={parse_str: <4} Exact={exact_str: <4} CritFail={crit_str: <4} RejViol={rej_str}\n")
            
    with open(blind_transcripts_file, "w") as f:
        for p_id in sorted(blind_data.keys()):
            f.write(f"=== PROMPT {p_id} ===\n")
            cids = list(blind_data[p_id].keys())
            random.shuffle(cids)
            for cid in cids:
                f.write(f"\n--- {cid} ---\n{blind_data[p_id][cid]}\n")
            f.write("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", help="Specific configs to run")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without loading models")
    args = parser.parse_args()
    do_run(args)
