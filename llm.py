import requests
import json
import os

HOME = os.path.expanduser("~")
GRAMMAR_FILE = HOME + "/actions.gbnf"
SERVER = "http://127.0.0.1:8080/completion"

SYSTEM = (
    "You are a robot controller. Output ONLY a JSON array of actions. "
    "Use exactly these formats: "
    "{type: navigate_to, room: <room>} or {type: find_person, name: <person>} or {type: say, message: <text>}. "
    "Use minimum actions needed."
)

def parse_command(voice_input):
    prompt = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" + voice_input + "<|im_end|>\n<|im_start|>assistant\n"
    grammar = open(GRAMMAR_FILE).read()
    resp = requests.post(SERVER, json={
        "prompt": prompt,
        "n_predict": 200,
        "temperature": 0.1,
        "grammar": grammar,
        "stop": ["<|im_end|>"],
    }, timeout=30)
    raw = resp.json()["content"]
    clean = raw[:raw.rfind("]") + 1]
    return json.loads(clean)

def get_field(action, *keys):
    for key in keys:
        if key in action:
            return action[key]
    return None

def execute_actions(actions):
    for action in actions:
        t = action.get("type")
        if t == "navigate_to":
            room = get_field(action, "room", "location", "value")
            print("[ROBOT] Navigating to: " + str(room))
        elif t == "find_person":
            name = get_field(action, "name", "person", "value")
            print("[ROBOT] Looking for: " + str(name))
        elif t == "say":
            msg = get_field(action, "message", "text", "value")
            if msg:
                print("[ROBOT] Saying: " + msg)
                os.system("termux-tts-speak \"" + msg + "\"")
            else:
                print("[ROBOT] say action missing message field: " + json.dumps(action))

if __name__ == "__main__":
    tests = [
        "Find Chiara in the kitchen and tell her Abdel is hungry",
        "Go to the living room",
        "Tell everyone dinner is ready",
    ]
    for test in tests:
        print("\nCommand: " + test)
        actions = parse_command(test)
        print("JSON: " + json.dumps(actions))
        execute_actions(actions)
