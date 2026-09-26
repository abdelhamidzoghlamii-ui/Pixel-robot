#!/usr/bin/env python3
"""robot-jevlike: interactive selector playground. Research only, offline, no motors.

Each model runs in its own venv as worker.py (JSON lines over stdin/stdout).
At most one worker is alive at a time; it is stopped before the next loads and on exit.
"""
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = str(HERE.parents[1] / "v3")
sys.path.insert(0, V3)
from measure import ENVS, THREADS, allowed_cpus  # v3 harness: venv + HF cache per candidate

STATE = Path("/termux-home/jevlike")  # outside the repo: worker logs and your last request
LOGS = STATE / "logs"
REQUEST = STATE / "request.txt"  # the last request you wrote, reopened next time
CORES = "4-7"

CAND, OUT, DROP = "CANDIDATE", "OUT (speed)", "DROPPED"
# menu name -> (v3 adapter, python, HF_HOME, status)
MODELS = {
    "laya_en": ("laya_en_onnx", *ENVS["laya_en_onnx"], CAND),
    "von11": ("von11", *ENVS["von11"], CAND),
    "s1o": ("s1o", *ENVS["s1o"], OUT),
    "laya_multi": ("laya_multi", *ENVS["laya_multi"], DROP),
    "laya_micro": ("laya_micro", *ENVS["laya_micro"], DROP),
    # von-sdk 1.2.0 through the same Von11 adapter, in its own venv and HF cache
    "von12": ("von11", "/termux-home/von12-test/venv/bin/python", "/termux-home/von12-test/hf-cache", DROP),
    "von10_nli": ("von10_nli", *ENVS["von10_nli"], DROP),
}
LABEL = {"laya_en": "laya_en (ONNX)", "von10_nli": "von10 NLI"}

TEMPLATE_HELP = """\
# robot-jevlike request (like a Jev request: one INPUT, any number of QUESTIONs).
# Write the situation after INPUT:, then each QUESTION: with its instruction and
# one choice per line starting with "-". Lines starting with # are ignored.
# Save and exit (nano: Ctrl-O Enter, Ctrl-X), then compile or add a question.
#
# No limit is enforced. Advised maximums (choices per question / questions per run):
#   laya_en, laya_multi, laya_micro   10 / 10
#       instruction + all choices share a 192-token head, each choice is cut at 48
#       tokens; past the budget choices get truncated. Calibrated for 2, 3-5, 6-10, 11+.
#       Whole sequence 512 tokens: a long INPUT gets cut at the end. ~0.3-0.6 s/question.
#   von11, von12                      8 / 6
#       one 2048-token sequence; calibration trained on up to ~8 options. ~0.8 s/question.
#   von10 NLI                         6 / 3
#       one 512-token pair per choice, time grows with choices (~0.45 s per choice).
#   s1o                               12 / 1-2
#       hard stop at 52 choices (A-Z, a-z); only A-L are checked single-token.
#       2048-token context. ~5 s per question.
"""

BLANK = TEMPLATE_HELP + """
INPUT:


QUESTION:
-
-
"""

# Newly written for this tool; not from the v3 splits or selector-testset.
DEMOS = [
    """INPUT:
The robot is in the garage. Its battery reads 9 percent and the charging dock is two rooms
away in the laundry room. Nobody has asked it for anything.

QUESTION: What should the robot do next?
- drive to the charging dock
- keep exploring the garage
- call out for the owner
- power down where it stands

QUESTION: Should the robot tell someone about its battery?
- yes, announce it out loud
- no, handle it silently
""",
    """INPUT:
A child in the living room just said 'my ball rolled under the sofa'. The robot is at the
doorway of the living room and its camera is working.

QUESTION: What should the robot do next?
- look under the sofa
- go to the kitchen
- say it cannot help
- return to the charging dock
""",
    """INPUT:
The robot was sent to find the cat. It has searched the bedroom and the bathroom with no
sign of it. It just heard a meow from the direction of the study.

QUESTION: Where should the robot search next?
- the study
- the bedroom again
- the bathroom again
- the hallway

QUESTION: How sure is the robot that the cat is in the study?
- very sure
- somewhat sure
- not sure at all
""",
    """INPUT:
The front door is wide open, it is raining outside, and the robot's owner left for work an
hour ago. The robot is in the hallway next to the door.

QUESTION: What should the robot do next?
- send an alert to the owner's phone
- go outside to look around
- ignore it and resume cleaning
- turn on the radio
""",
]

C = not os.environ.get("NO_COLOR")
def paint(code, s): return f"\033[{code}m{s}\033[0m" if C else s
green, bold, dim, yellow, red, cyan = (lambda s, c=c: paint(c, s) for c in ("32", "1", "2", "33", "31", "36"))


class Quit(Exception):
    pass


def ask(prompt):
    try:
        s = input(prompt)
    except EOFError:
        raise Quit
    if s.strip().lower() == "q":
        raise Quit
    return s


def free_mib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024


def tag(name):
    st = MODELS[name][3]
    return green(st) if st == CAND else yellow(st) if st == OUT else dim(st)


def label(name):
    s = LABEL.get(name, name)
    return green(bold(s)) if MODELS[name][3] == CAND else dim(s) if MODELS[name][3] == DROP else s


def installed(name):
    return os.path.exists(MODELS[name][1])


class Worker:
    def __init__(self, name):
        adapter, python, hf, _ = MODELS[name]
        self.name = name
        LOGS.mkdir(parents=True, exist_ok=True)
        server_log = LOGS / f"{name}.server.log"
        server_log.unlink(missing_ok=True)  # s1o opens it exclusive-create
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   USE_TF="0", OMP_NUM_THREADS=THREADS, V3_THREADS=THREADS, S1O_SERVER_LOG=str(server_log))
        if hf:
            env.update(HF_HOME=hf, HF_HUB_CACHE=f"{hf}/hub")
        cmd = [python, str(HERE / "worker.py"), adapter]
        if set(range(4, 8)) <= allowed_cpus():
            cmd = ["taskset", "-c", CORES] + cmd
        began = time.perf_counter()
        # Own session: Ctrl-C reaches only this UI, and killpg also reaches llama-server.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
                                     stderr=open(LOGS / f"{name}.log", "w"), start_new_session=True)
        try:
            ready = self._read()
        except BaseException:
            self.stop()
            raise
        if "error" in ready:
            self.stop()
            raise RuntimeError(ready["error"])
        self.wall_s = time.perf_counter() - began
        self.load_ms, self.warmup_ms = ready["load_ms"], ready["warmup_ms"]

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited (see {LOGS / (self.name + '.log')})")
        return json.loads(line)

    def decide(self, state, instruction, options, text):
        try:
            self.proc.stdin.write(json.dumps({"state": state, "options": {o: "" for o in options},
                                              "instruction": instruction, "text": text}) + "\n")
            self.proc.stdin.flush()
            out = self._read()
        except BaseException:
            self.stop()  # a half-finished reply would desync the pipe
            raise
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def stop(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # EOF -> adapter.close() -> llama-server stops
            except OSError:
                pass
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)  # backstop for anything left in the group
        except ProcessLookupError:
            pass
        self.proc.wait()


current = None


def load(name):
    """Return (worker, reused). Keeps at most one model in memory."""
    global current
    if current and current.name == name:
        return current, True
    unload()
    print(dim(f"  loading {name}... free RAM {free_mib()} MiB"), flush=True)
    current = Worker(name)
    print(dim(f"  {name} ready in {current.wall_s:.1f} s, free RAM now {free_mib()} MiB"))
    return current, False


def unload():
    global current
    if current:
        name, w = current.name, current
        current = None
        w.stop()
        print(dim(f"  unloaded {name}, free RAM {free_mib()} MiB"))


def run(name, state, questions, flags):
    """One model, every question: normal order, plus reversed if toggled."""
    began = time.perf_counter()
    w, reused = load(name)
    res = {"name": name, "reused": reused, "wall_s": w.wall_s, "load_ms": w.load_ms, "warmup_ms": w.warmup_ms,
           "answers": []}
    for instruction, options in questions:
        a = {"normal": w.decide(state, instruction, options, flags["text"])}
        if flags["reverse"]:
            a["reverse"] = w.decide(state, instruction, options[::-1], flags["text"])
        res["answers"].append(a)
    res["total_s"] = time.perf_counter() - began
    return res


def choice(dist):
    top = max(dist.values())
    return next(k for k in dist if dist[k] == top)  # ties: first offered, as in run.py


def load_str(r):
    if r["reused"]:
        return "already loaded"
    return f"{r['wall_s']:.1f} s (adapter {r['load_ms'] / 1000:.1f} s + warm-up {r['warmup_ms'] / 1000:.1f} s)"


def show_ranked(dist):
    pick = choice(dist)
    for k in sorted(dist, key=dist.get, reverse=True):
        bar = "#" * round(dist[k] * 20)
        line = f"  {dist[k] * 100:5.1f}%  {bar:<20}  {k}"
        print(green(bold(line + "  <= choice")) if k == pick else line)


def margin(dist):
    """Top choice minus runner-up, in percentage points."""
    top = sorted(dist.values(), reverse=True)
    return (top[0] - top[1]) * 100


def confidence(dist):
    """1 - normalised entropy: 1 = all mass on one choice, 0 = uniform."""
    h = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return 1 - h / math.log(len(dist))


def tok_s(ans):
    return ans["tokens"] / (ans["ms"] / 1000)


def truncated(ans):
    return ans["longest"] >= ans["limit"]


def show_stats(ans):
    print(dim(f"  {ans['ms']:.0f} ms | {ans['tokens']} tokens read (limit {ans['limit']}) | {tok_s(ans):.0f} tok/s"
              f" | margin {margin(ans['dist']):.1f} pts | confidence {confidence(ans['dist']):.2f}"))
    if truncated(ans):
        print(red(f"  input hit the {ans['limit']}-token limit: the end of it was cut and never read"))


def show_summary(r):
    q_ms = sum(a[o]["ms"] for a in r["answers"] for o in a)
    last = r["answers"][-1]["normal"]
    load = "0 s (already loaded)" if r["reused"] else f"{r['wall_s']:.1f} s"
    print(bold(f"\n  compile: {r['total_s']:.1f} s total") +
          f" = load {load} + {sum(len(a) for a in r['answers'])} decision(s) {q_ms / 1000:.2f} s + overhead")
    print(f"  memory: model uses {last['rss_mib']} MiB now, peak {last['peak_mib']} MiB; free RAM {free_mib()} MiB")
    print(dim("  tok/s = prompt tokens read per second (these models score the choices, they generate no text)"))


def show_single(r, questions):
    print(f"\n{label(r['name'])}  [{tag(r['name'])}]")
    print(f"  load: {load_str(r)}")
    for i, ((instruction, _), a) in enumerate(zip(questions, r["answers"]), 1):
        print(bold(f"\n  Q{i}: {instruction}"))
        show_ranked(a["normal"]["dist"])
        show_stats(a["normal"])
        if "reverse" in a:
            x, y = choice(a["normal"]["dist"]), choice(a["reverse"]["dist"])
            print("  reversed order:")
            show_ranked(a["reverse"]["dist"])
            show_stats(a["reverse"])
            print("  choice changed: " + (red(bold("YES")) + f"  ({x} -> {y})" if x != y else green("no")))
        show_text(r["name"], i, a)
    show_summary(r)


def show_text(name, i, a):
    for order in ("normal", "reverse"):
        if order in a and a[order].get("text") is not None:
            print(cyan(f"\n  --- exact text sent to {name}, Q{i} ({order} order) ---"))
            print(a[order]["text"])
            print(cyan("  --- end ---"))


def show_compare(results, questions, flags):
    """Models as columns, split into blocks that fit the terminal width."""
    O = max(18, min(26, max(len(o) for _, opts in questions for o in opts) + 2))  # row-label width
    W = 12                                                                        # column width
    per = max(1, (shutil.get_terminal_size().columns - O) // W)
    for start in range(0, len(results), per):
        compare_block(results[start:start + per], questions, flags, O, W)
    print(dim("  * = choice. Probabilities in the order written. CUT = input hit the token limit."))
    print(dim("  margin = top minus runner-up; confidence = 1 - normalised entropy (1 sure, 0 uniform)."))
    print(dim("  tok/s = prompt tokens read per second (these models score the choices, they generate no text)."))
    print(dim("  Models run one after another, so RAM figures are each model on its own."))


def compare_block(rs, questions, flags, O, W):
    short = {CAND: "CANDIDATE", OUT: "OUT(speed)", DROP: "DROPPED"}
    def head(text, fn=None):  # pad the plain text first, then colour, so ANSI codes don't break alignment
        text = text if len(text) <= O - 2 else text[:O - 3] + "~"
        return fn(text.ljust(O)) if fn else text.ljust(O)
    def cell(text, fn=None):
        text = text.rjust(W - 2)[:W - 2] + "  "
        return fn(text) if fn else text
    def row(label, cells, fn=dim):
        print(head(label, fn) + "".join(cells))
    rule = lambda ch="─": print(dim(ch * (O + W * len(rs))))
    colour = {CAND: lambda t: green(bold(t)), OUT: yellow, DROP: dim}

    print()
    row("", [cell(r["name"], colour[MODELS[r["name"]][3]]) for r in rs])
    row("", [cell(short[MODELS[r["name"]][3]], colour[MODELS[r["name"]][3]]) for r in rs])
    for i, (instruction, options) in enumerate(questions):
        rule()
        print(bold(textwrap.fill(f"Q{i + 1}: {instruction}", O + W * len(rs))))
        ans = [r["answers"][i]["normal"] for r in rs]
        for o in options:
            row(o, [cell(("* " if choice(a["dist"]) == o else "") + f"{a['dist'][o] * 100:.1f}%",
                         (lambda t: green(bold(t))) if choice(a["dist"]) == o else None) for a in ans], fn=None)
        rule("·")
        row("decision ms", [cell(f"{a['ms']:.0f}") for a in ans])
        row("tokens read", [cell(f"{a['tokens']}" + (" CUT" if truncated(a) else ""),
                                 red if truncated(a) else None) for a in ans])
        row("tok/s", [cell(f"{tok_s(a):.0f}") for a in ans])
        row("margin pts", [cell(f"{margin(a['dist']):.1f}") for a in ans])
        row("confidence", [cell(f"{confidence(a['dist']):.2f}") for a in ans])
        if flags["reverse"]:
            cells = []
            for r in rs:
                a = r["answers"][i]
                changed = choice(a["normal"]["dist"]) != choice(a["reverse"]["dist"])
                cells.append(cell("CHANGED", lambda t: red(bold(t))) if changed else cell("same", green))
            row("reversed order", cells)
            for r in rs:
                a = r["answers"][i]
                if choice(a["normal"]["dist"]) != choice(a["reverse"]["dist"]):
                    print(dim(f"  {r['name']} reversed -> {choice(a['reverse']['dist'])}"))
    rule()
    row("load s", [cell("loaded" if r["reused"] else f"{r['wall_s']:.1f}") for r in rs])
    row("compile total s", [cell(f"{r['total_s']:.1f}") for r in rs], fn=bold)
    row("model RAM MiB", [cell(f"{r['answers'][-1]['normal']['rss_mib']}") for r in rs])
    row("peak RAM MiB", [cell(f"{r['answers'][-1]['normal']['peak_mib']}") for r in rs])
    rule()
    for r in rs:
        for i, a in enumerate(r["answers"], 1):
            show_text(r["name"], i, a)


def pick_models(multi):
    names = list(MODELS)
    print()
    for i, n in enumerate(names, 1):
        extra = "" if installed(n) else red("  not installed")
        print(f"  [{i}] {label(n)}{' ' * (16 - len(LABEL.get(n, n)))} {tag(n)}{extra}")
    while True:
        prompt = ("Models (numbers separated by spaces, 'c' = both candidates): " if multi else "Model number: ")
        s = ask(prompt).strip().lower()
        try:
            chosen = ([n for n in names if MODELS[n][3] == CAND] if multi and s == "c"
                      else [names[int(x) - 1] for x in s.split()])
        except (ValueError, IndexError):
            chosen = []
        chosen = list(dict.fromkeys(chosen))
        if chosen and (multi or len(chosen) == 1) and all(installed(n) for n in chosen):
            order = {CAND: 0, OUT: 1, DROP: 2}
            return sorted(chosen, key=lambda n: order[MODELS[n][3]])  # candidates first
        print(red("  invalid choice"))


def parse(text):
    """INPUT: + QUESTION: blocks -> (state, [(instruction, [choices])]). Raises ValueError."""
    state, questions, cur = [], [], None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        head, _, rest = line.partition(":")
        if head.upper() == "INPUT":
            cur = state
            line = rest.strip()
        elif head.upper() == "QUESTION":
            questions.append(([], []))
            cur = questions[-1][0]
            line = rest.strip()
        elif line.startswith("-") and questions:
            if line[1:].strip():
                questions[-1][1].append(line[1:].strip())
            continue
        if line and cur is None:
            raise ValueError(f"text before INPUT: or QUESTION: -> {line!r}")
        if line:
            cur.append(line)
    state = " ".join(state)
    questions = [(" ".join(ins), list(dict.fromkeys(ch))) for ins, ch in questions]
    if not state:
        raise ValueError("INPUT is empty")
    if not questions:
        raise ValueError("no QUESTION")
    for i, (ins, ch) in enumerate(questions, 1):
        if not ins:
            raise ValueError(f"Q{i} has no instruction after QUESTION:")
        if len(ch) < 2:
            raise ValueError(f"Q{i} needs at least 2 distinct choices")
    return state, questions


def edit(text, line=1):
    """Open the request in $EDITOR (default nano) and return what was saved."""
    REQUEST.parent.mkdir(parents=True, exist_ok=True)
    REQUEST.write_text(text)
    subprocess.call(shlex.split(os.environ.get("EDITOR") or "nano") + [f"+{line}", str(REQUEST)])
    return REQUEST.read_text()


def request_screen():
    """Write the request, then compile or add questions. Returns (state, questions)."""
    text = REQUEST.read_text() if REQUEST.exists() else edit(BLANK, BLANK.count("\n") - 4)
    while True:
        try:
            state, questions = parse(text)
            print(bold(f"\n  Request: {len(questions)} question(s)"))
            width = shutil.get_terminal_size().columns - 2
            print(textwrap.fill(state, width, initial_indent="   INPUT: ", subsequent_indent="          "))
            for i, (ins, ch) in enumerate(questions, 1):
                print(f"   Q{i} {ins}  " + dim(f"({len(ch)} choices)"))
            ok = True
        except ValueError as e:
            print(red(f"\n  request not valid: {e}"))
            ok = False
        print(("  [c] compile   " if ok else "  ") + "[a] add a question   [e] edit   [d] load a demo   [b] back")
        cmd = ask("> ").strip().lower()
        if cmd == "c" and ok:
            return state, questions
        if cmd == "a":
            text = text.rstrip("\n") + "\n\nQUESTION: \n-\n-\n"
            text = edit(text, text.count("\n") - 2)
        elif cmd == "e":
            text = edit(text)
        elif cmd == "d":
            for i, demo in enumerate(DEMOS, 1):
                print(f"  [{i}] {demo.splitlines()[1][:70]}...")
            try:
                text = edit(TEMPLATE_HELP + "\n" + DEMOS[int(ask("Demo number: ")) - 1])
            except (ValueError, IndexError):
                print(red("  invalid choice"))
        elif cmd == "b":
            return None


def menu(flags):
    onoff = lambda b: green("ON") if b else dim("off")
    loaded = current.name if current else "none"
    print(bold("\nrobot-jevlike") + dim("  selector playground, research only, offline"))
    print(f"  free RAM {free_mib()} MiB   loaded: {loaded}")
    print("  [1] single model   [2] compare models")
    print(f"  [r] reversed-order rerun: {onoff(flags['reverse'])}   [t] show exact text: {onoff(flags['text'])}")
    print("  [q] quit (works at any prompt; Ctrl-C returns here)")
    return ask("> ").strip().lower()


def main():
    flags = {"reverse": False, "text": False}
    while True:
        try:
            cmd = menu(flags)
            if cmd == "r":
                flags["reverse"] = not flags["reverse"]
            elif cmd == "t":
                flags["text"] = not flags["text"]
            elif cmd in ("1", "2"):
                names = pick_models(cmd == "2")
                while (req := request_screen()):  # after results, back here to edit and recompile
                    state, questions = req
                    results = []
                    for n in names:
                        try:
                            results.append(run(n, state, questions, flags))
                        except RuntimeError as e:
                            print(red(f"  {n} failed: {e}"))
                    if results and cmd == "1":
                        show_single(results[0], questions)
                    elif results:
                        show_compare(results, questions, flags)
        except KeyboardInterrupt:
            print(yellow("\n  interrupted, back to menu"))


def cleanup(*_):
    unload()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGHUP, cleanup)
    try:
        main()
    except Quit:
        pass
    finally:
        unload()
        print("bye")
