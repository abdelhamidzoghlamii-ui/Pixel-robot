import requests, time

SYSTEM = "You are a robot navigation controller. Always reply with exactly one word (FORWARD, LEFT, RIGHT, BACK, or LOOK) followed by one short reason. Never repeat instructions."

def test(label, situation):
    prompt = (
        "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n" + situation + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    t0 = time.time()
    resp = requests.post("http://127.0.0.1:8080/completion", json={
        "prompt": prompt,
        "n_predict": 40,
        "temperature": 0.1,
        "stop": ["<|im_end|>", "\n\n"]
    }, timeout=60)
    t = round(time.time() - t0, 2)
    content = resp.json()["content"].strip()
    print(f"\n[{label}] {t}s")
    print(f"  Response: {repr(content[:100])}")
    return t

t1 = test("CLEAR PATH",
    "Dist:45cm clear. Sees:hallway door-left. Mission:find kitchen. Last:FWD FWD.")

t2 = test("OBSTACLE",
    "Dist:18cm blocked. Sees:chair ahead space-right. Mission:find Abdel. Last:FWD FWD FWD.")

t3 = test("PERSON DETECTED",
    "Dist:35cm clear. Sees:person on left. Mission:find Chiara. Last:FWD LOOK.")

t4 = test("AFTER LOOK",
    "Dist:35cm clear. LOOK result:woman dark hair on left. Mission:find Chiara. Last:FWD LOOK.")

t5 = test("UNKNOWN ROOM",
    "Dist:90cm clear. Sees:sofa TV open-room. Mission:find kitchen. Known:bedroom=right. Last:FWD RIGHT FWD.")

print(f"\n=== TIMING SUMMARY ===")
for label, t in [("Clear path",t1),("Obstacle",t2),("Person",t3),("After LOOK",t4),("Unknown room",t5)]:
    print(f"  {label}: {t}s")
print(f"  Average: {round((t1+t2+t3+t4+t5)/5,2)}s")
