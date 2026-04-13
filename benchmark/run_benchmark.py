import sys, os, time, json, requests, glob
sys.path.insert(0, "/data/data/com.termux/files/home/robot")
from detect_person import detect_person

HOME        = "/data/data/com.termux/files/home"
RESULTS_DIR = HOME + "/robot/results"
PERSON_PHOTO = HOME + "/robot/test_photos/chiara_close_left.jpg"
EMPTY_PHOTO  = HOME + "/robot/test_photos/empty_1.jpg"
GRAMMAR      = open(HOME + "/actions.gbnf").read()
EASY_CMD     = "Go to the kitchen and tell Chiara dinner is ready"
HARD_CMD     = "Find Abdel and tell him that Chiara needs help in the kitchen as soon as possible"
os.makedirs(RESULTS_DIR, exist_ok=True)

QWEN_SYSTEM = (
    "You are a robot controller. Output ONLY a JSON array of actions. "
    "Use exactly these formats: {type: navigate_to, room: <room>} or "
    "{type: find_person, name: <person>} or {type: say, message: <text>}. "
    "Use minimum actions needed."
)
GEMMA_SYSTEM = (
    "Output ONLY a JSON array of robot actions. "
    "Use exactly these formats and no other fields: "
    "{\"type\": \"navigate_to\", \"room\": \"<room>\"} or "
    "{\"type\": \"find_person\", \"name\": \"<name>\"} or "
    "{\"type\": \"say\", \"message\": \"<text>\"}. "
    "Example: [{\"type\": \"navigate_to\", \"room\": \"kitchen\"}, {\"type\": \"say\", \"message\": \"dinner is ready\"}]"
)

def qwen_prompt(text):
    return "<|im_start|>system\n" + QWEN_SYSTEM + "<|im_end|>\n<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"

def gemma_prompt(text):
    return "<start_of_turn>user\n" + GEMMA_SYSTEM + "\n" + text + "<end_of_turn>\n<start_of_turn>model\n["

def gemma_vision_prompt(photo, text):
    return "<start_of_turn>user\n" + GEMMA_SYSTEM + "\n<img src=file://" + photo + ">\n" + text + "<end_of_turn>\n<start_of_turn>model\n["

def parse_json(raw, is_gemma=False):
    if is_gemma and not raw.strip().startswith("["):
        raw = "[" + raw
    s = raw.find("[")
    e = raw.rfind("]") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(raw[s:e])
    except:
        return None

def llm_query(text, setup_name, image_path=None):
    is_gemma = setup_name in ("setup_b", "setup_c")
    if is_gemma:
        prompt = gemma_vision_prompt(image_path, text) if image_path else gemma_prompt(text)
        payload = {"prompt": prompt, "n_predict": 200, "temperature": 0.1, "stop": ["<end_of_turn>"]}
    else:
        prompt = qwen_prompt(text)
        payload = {"prompt": prompt, "n_predict": 200, "temperature": 0.1, "grammar": GRAMMAR, "stop": ["<|im_end|>"]}
    resp = requests.post("http://127.0.0.1:8080/completion", json=payload, timeout=120)
    raw = resp.json()["content"]
    actions = parse_json(raw, is_gemma)
    return actions, raw[:200]

def gemma_detect(photo, port=8080):
    base = "http://127.0.0.1:" + str(port) + "/completion"
    prompt = "<start_of_turn>user\n<img src=file://" + photo + ">\nWhat is the main subject of this photo? A person, an object, or an empty room?<end_of_turn>\n<start_of_turn>model\n"
    r = requests.post(base, json={
        "prompt": prompt, "n_predict": 30, "temperature": 0.1, "stop": ["<end_of_turn>", "\n"]
    }, timeout=60)
    raw = r.json()["content"].strip().lower()
    detected = "person" in raw
    if detected:
        p2 = "<start_of_turn>user\n<img src=file://" + photo + ">\nWhere is the person in the frame? One word: left, center, or right.<end_of_turn>\n<start_of_turn>model\n"
        r2 = requests.post(base, json={
            "prompt": p2, "n_predict": 5, "temperature": 0.1, "stop": ["<end_of_turn>", "\n"]
        }, timeout=60)
        pos = r2.json()["content"].strip().lower()
        pos = "left" if "left" in pos else "right" if "right" in pos else "center"
        return True, pos
    return False, "none"

def e2b_describe(photo):
    prompt = "<start_of_turn>user\n<img src=file://" + photo + ">\nDescribe the person you see in one sentence: where they are in the frame, and what they look like.<end_of_turn>\n<start_of_turn>model\n"
    r = requests.post("http://127.0.0.1:8080/completion", json={
        "prompt": prompt, "n_predict": 60, "temperature": 0.1, "stop": ["<end_of_turn>", "\n\n"]
    }, timeout=60)
    return r.json()["content"].strip()

def run_vision(setup_name, photo):
    t0 = time.time()
    if setup_name == "setup_d":
        # YOLO detects, E2B describes
        found, conf, pos = detect_person(photo)
        if found:
            description = e2b_describe(photo)
            result = "detected=True pos=" + pos + " | " + description[:80]
        else:
            result = "detected=False conf=" + str(round(conf,2))
    elif setup_name in ("setup_b", "setup_c"):
        detected, pos = gemma_detect(photo)
        result = '{"person_detected": ' + str(detected).lower() + ', "position": "' + pos + '"}'
    else:
        found, conf, pos = detect_person(photo)
        result = "detected=" + str(found) + " conf=" + str(round(conf,2)) + " pos=" + pos
    return result, round(time.time() - t0, 2)

def run_test(label, command, setup_name):
    print("\n  [" + label.upper() + "]")
    print("  Command: \"" + command + "\"")
    t_start = time.time()
    t0 = time.time()
    actions, raw = llm_query(command, setup_name)
    t_llm = round(time.time() - t0, 2)
    print("  JSON:    " + str(raw[:100]))
    t_vision, vision_result = 0, "n/a"
    if actions and any(isinstance(a, dict) and a.get("type") == "find_person" for a in actions):
        vision_result, t_vision = run_vision(setup_name, PERSON_PHOTO)
        print("  Vision:  " + vision_result)
    t_total = round(time.time() - t_start, 2)
    print("  LLM=" + str(t_llm) + "s  Vision=" + str(t_vision) + "s  TOTAL=" + str(t_total) + "s")
    return {"label": label, "command": command, "actions": actions, "json_raw": raw,
            "json_ok": actions is not None, "t_llm": t_llm, "t_vision": t_vision,
            "t_total": t_total, "vision_result": vision_result}

def run_false_positive(setup_name):
    print("\n  [FALSE POSITIVE CHECK]")
    result, elapsed = run_vision(setup_name, EMPTY_PHOTO)
    print("  Result: " + result + " (" + str(elapsed) + "s)")
    return result

def run_benchmark(setup_name):
    print("\n" + "="*55)
    print("  BENCHMARK: " + setup_name.upper())
    print("="*55)
    tests = [run_test(label, cmd, setup_name) for label, cmd in [("easy", EASY_CMD), ("hard", HARD_CMD)]]
    fp = run_false_positive(setup_name)
    results = {"setup": setup_name, "timestamp": time.strftime("%Y-%m-%d %H:%M"), "tests": tests, "false_positive": fp}
    out = RESULTS_DIR + "/" + setup_name + "_" + time.strftime("%Y%m%d_%H%M") + ".json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: " + out)
    return out

def print_comparison():
    files = sorted(glob.glob(RESULTS_DIR + "/setup_*.json"))
    # keep only latest per setup
    latest = {}
    for f in files:
        key = os.path.basename(f)[:7]
        latest[key] = f
    files = sorted(latest.values())
    if not files:
        print("No results yet.")
        return
    print("\n" + "="*70)
    print("  COMPARISON TABLE")
    print("="*70)
    fmt = "  {:<20} {:<8} {:<10} {:<10} {:<8} {:<6}"
    print(fmt.format("Setup", "Test", "LLM(s)", "Vision(s)", "Total(s)", "JSON?"))
    print("  " + "-"*68)
    for path in files:
        with open(path) as f:
            r = json.load(f)
        for t in r["tests"]:
            print(fmt.format(r["setup"], t["label"], str(t["t_llm"]),
                             str(t["t_vision"]), str(t["t_total"]),
                             "OK" if t["json_ok"] else "FAIL"))
        print("  FP: " + str(r.get("false_positive", "n/a")))
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        print_comparison()
    else:
        setup = sys.argv[1] if len(sys.argv) > 1 else "setup_a"
        run_benchmark(setup)
        print_comparison()
