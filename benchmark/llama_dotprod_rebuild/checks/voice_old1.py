"""3b follow-up: command 1 on the old binary after its warm-up request has finished server-side."""
import json, sys, time
sys.argv = [sys.argv[0], "/dev/null"]
src = open(__file__.replace("voice_old1.py", "voice_gen.py")).read()
exec(src[:src.index("res = {")])
sm.LLAMA_SERVER = OLD
assert sm.start_setup("setup_q4")
ns["parse_command"]("Come back")
time.sleep(90)
t = time.perf_counter(); a = ns["parse_command"](COMMANDS[0]); ms = round((time.perf_counter() - t) * 1000)
sm.kill_servers()
print(f"old {COMMANDS[0]!r} ({ms} ms): {json.dumps(a)}   raw={raw.get('content')!r}")
