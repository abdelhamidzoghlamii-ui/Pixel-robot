"""Smoke: the edited server_manager.start_setup('setup_q4') + one parse_command (from main.py source)."""
import sys, time
sys.argv = [sys.argv[0], "/dev/null"]
src = open(__file__.replace("smoke.py", "voice_gen.py")).read()
exec(src[:src.index("OLD = ")] + src[src.index("MODEL = "):src.index("res = {")])  # server_manager's own LLAMA_SERVER
print("LLAMA_SERVER", sm.LLAMA_SERVER)
assert sm.start_setup("setup_q4")
import subprocess
print(subprocess.run("pgrep -af llama-server | grep -v pgrep", shell=True, capture_output=True, text=True).stdout.strip())
t = time.perf_counter(); a = ns["parse_command"]("Go to the kitchen"); print(a, round(time.perf_counter() - t, 1), "s")
sm.start_setup("stop")
