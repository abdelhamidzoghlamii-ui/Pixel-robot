import requests, time, os, sys, random
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

MODEL  = "/data/data/com.termux/files/home/models/qwen2.5-3b-instruct-q4_k_m.gguf"
SERVER = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-server"

WARN_TEMP  = 75000  # pause 5s
KILL_TEMP  = 88000  # emergency stop
COOL_TEMP  = 48000  # target after each cycle
CYCLE_REST = 4.0    # base rest between cycles (tune this)
BIG_CORES  = "/sys/class/thermal/thermal_zone9/temp"

SYSTEM = """You are a robot. Reply with ONE word: FORWARD, LEFT, RIGHT, or BACK.
Examples:
Dist:80cm clear. Sees:hallway. -> FORWARD
Dist:15cm blocked. Sees:wall space-left. -> LEFT
Dist:20cm blocked. Sees:chair space-right. -> RIGHT
Dist:45cm clear. Sees:person left. -> LEFT
Dist:10cm blocked. Sees:wall left wall right. -> BACK
Dist:90cm clear. Sees:open room sofa. -> FORWARD"""

def get_temp():
    try:
        return int(os.popen('su -c "cat ' + BIG_CORES + '"').read().strip())
    except:
        return 0

def check_thermal():
    t = get_temp()
    if t >= KILL_TEMP:
        print(f"\n  KILL: {t//1000}C — stopping!")
        os.system("pkill -f llama-server 2>/dev/null")
        return "kill"
    elif t >= WARN_TEMP:
        print(f"\n  WARN: {t//1000}C — pause 5s")
        time.sleep(5)
        return "warn"
    return "ok"

def kill_chrome():
    # Force stop all Chrome processes
    os.system("am force-stop com.android.chrome 2>/dev/null")
    os.system("am force-stop com.google.android.chrome 2>/dev/null")
    # Kill any remaining chrome processes
    os.system("su -c 'pkill -f chrome' 2>/dev/null")
    time.sleep(1)
    # Verify
    check = os.popen("su -c 'top -n 1 -b | grep -i chrome | grep -v grep'").read()
    if "chrome" in check.lower():
        print("[SETUP] WARNING: Chrome still running!")
    else:
        print("[SETUP] Chrome killed OK")

def robot_mode_on():
    print("[SETUP] Starting robot mode...")

    print("[SETUP] Killing unnecessary apps...")
    kill_chrome()
    os.system("am force-stop com.google.android.youtube 2>/dev/null")
    os.system("am force-stop com.google.android.apps.maps 2>/dev/null")
    os.system("su -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null")
    time.sleep(1)

    mem = os.popen("free -h | grep Mem").read().strip()
    print(f"[SETUP] RAM: {mem}")

    print("[SETUP] Starting LLM server...")
    os.system("pkill -f llama-server 2>/dev/null")
    time.sleep(1)
    os.system(SERVER + " -m " + MODEL + " --port 8080 --ctx-size 2048 --host 127.0.0.1 2>/dev/null &")
    import urllib.request
    for i in range(30):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
            print(f"[SETUP] Server ready after {i+1}s | Temp: {get_temp()//1000}C")
            break
        except:
            pass
    print("[SETUP] Robot mode active\n")

def robot_mode_off():
    print("\n[SETUP] Shutting down...")
    os.system("pkill -f llama-server 2>/dev/null")
    print(f"[SETUP] Server killed | Final temp: {get_temp()//1000}C")

def decide(situation):
    prompt = (
        "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n" + situation + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    t0 = time.time()
    resp = requests.post("http://127.0.0.1:8080/completion", json={
        "prompt": prompt, "n_predict": 40,
        "temperature": 0.1, "stop": ["<|im_end|>", "\n\n"]
    }, timeout=30)
    elapsed = round(time.time() - t0, 2)
    return resp.json()["content"].strip(), elapsed

scenes = [
    ("Hallway ahead clear 80cm. Mission: find kitchen. Last: {last}.",                        "hallway",   "FORWARD"),
    ("Chair blocking path 20cm, open space to your RIGHT. Mission: find kitchen. Last: {last}.", "chair R", "RIGHT"),
    ("Living room with sofa and TV, 60cm clear. Mission: find kitchen. Last: {last}.",         "living rm", "FORWARD"),
    ("Person is standing to your LEFT 45cm away. Mission: find Chiara. Last: {last}.",         "person L",  "LEFT"),
    ("Person is standing to your RIGHT 50cm away. Mission: find Abdel. Last: {last}.",         "person R",  "RIGHT"),
    ("Door ahead with bright room beyond, 90cm clear. Mission: find kitchen. Last: {last}.",   "door",      "FORWARD"),
    ("Path blocked 15cm, open space to your LEFT. Mission: find kitchen. Last: {last}.",       "wall->L",   "LEFT"),
    ("Fridge and counter visible ahead 70cm. Mission: find kitchen. Last: {last}.",            "kitchen",   "FORWARD"),
    ("Bedroom detected - wrong room, must reverse. Mission: find kitchen. Last: {last}.",      "bedroom",   "BACK"),
    ("Dead end walls on all sides, must reverse. Mission: find kitchen. Last: {last}.",        "dead end",  "BACK"),
]

# ── Main ─────────────────────────────────────────────
robot_mode_on()

print("="*55)
print("  1-MINUTE NAVIGATION SIMULATION")
print("  Model: Qwen 3B | Vision: YOLO sim")
print("="*55)
print(f"  Start temp: {get_temp()//1000}C")
print(f"  Cycle rest: {CYCLE_REST}s | Warn: {WARN_TEMP//1000}C | Kill: {KILL_TEMP//1000}C\n")

start = time.time()
cycle = 0
last_moves = ["START"]
total_llm = 0
decisions = {}
score = []
thermal_pauses = 0
temps = []

while time.time() - start < 60:
    cycle += 1
    elapsed = round(time.time() - start, 1)
    temp_c = get_temp() // 1000
    temps.append(temp_c)

    status = check_thermal()
    if status == "kill":
        break
    elif status == "warn":
        thermal_pauses += 1
        continue

    scene_template, label, correct = random.choice(scenes)
    situation = scene_template.format(last=",".join(last_moves[-3:]))
    print(f"  Situation: {situation}")
    decision, llm_time = decide(situation)
    total_llm += llm_time

    move = "FORWARD"
    for word in decision.upper().split():
        if word.strip(".,") in ["FORWARD","LEFT","RIGHT","BACK"]:
            move = word.strip(".,")
            break

    decisions[move] = decisions.get(move, 0) + 1
    mark = "OK" if move == correct else "XX"
    score.append(move == correct)
    last_moves.append(move)
    if len(last_moves) > 5:
        last_moves.pop(0)

    print(f"  [{cycle:2d}] {elapsed}s {temp_c}C | {label:<16} -> {move} {mark} (exp:{correct}) {llm_time}s | {decision[:40]}")

    # Motor execution = natural cooling window
    # Robot moves for 2.5s, CPU idles and cools
    time.sleep(2.5)
    
    # After movement check temp — if still high add small extra cooldown
    t_after = get_temp()
    if t_after > 70000:
        extra = 0
        while get_temp() > 65000 and extra < 5:
            time.sleep(1)
            extra += 1
        if extra > 0:
            print(f"  [+{extra}s cooling after spike]")

duration = round(time.time() - start, 1)
accuracy = round(sum(score)/max(len(score),1)*100)

print(f"\n{'='*55}")
print(f"  RESULTS")
print(f"{'='*55}")
print(f"  Duration:  {duration}s")
print(f"  Cycles:    {cycle}")
print(f"  Avg LLM:   {round(total_llm/max(cycle,1),2)}s")
print(f"  Accuracy:  {accuracy}% ({sum(score)}/{len(score)} correct)")
print(f"  Pauses:    {thermal_pauses}")
print(f"  Avg temp:  {round(sum(temps)/max(len(temps),1))}C")
print(f"  Max temp:  {max(temps) if temps else 0}C")
print(f"\n  DECISIONS:")
for m,c in sorted(decisions.items(), key=lambda x:-x[1]):
    print(f"    {m:<8}: {c}")
print(f"\n  THERMAL REPORT:")
zones = {"BIG":"/sys/class/thermal/thermal_zone9/temp",
         "MID":"/sys/class/thermal/thermal_zone10/temp",
         "LITTLE":"/sys/class/thermal/thermal_zone11/temp",
         "GPU":"/sys/class/thermal/thermal_zone12/temp",
         "Battery":"/sys/class/thermal/thermal_zone25/temp"}
for name, path in zones.items():
    try:
        t = int(os.popen('su -c "cat '+path+'"').read().strip())
        bar = "ok  " if t<50000 else "warm" if t<60000 else "HOT "
        print(f"  {bar} {name:<8}: {t//1000}C")
    except:
        pass

robot_mode_off()
