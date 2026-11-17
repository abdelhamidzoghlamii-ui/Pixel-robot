import os
import shlex
import shutil
import subprocess
from pathlib import Path
import time
import urllib.request

HOME = "/data/data/com.termux/files/home"
LLAMA_SERVER = HOME + "/llama.cpp-b1609-dotprod/build/bin/llama-server"

# Optimal config (benchmarked on Pixel 7 Tensor G2):
# --threads 4 --threads-batch 4 → ~12 tok/s for E2B Q4_0 (dotprod build, cores 4-7, 2026-09-26)
# --parallel 1                  → no slot splitting
# --swa-full                    → full-size SWA cache, so the fixed prompt prefix is reused
# --cache-ram 0                 → no host-RAM prompt cache; it grew server memory per turn on
#                                 prompts without a shared prefix (benchmark/llm_objective_setting)
THREADS = "4"
BATCH_THREADS = "4"

# MTP drafter, opt-in through setup_q4_mtp; not the robot default. Measured in DECISIONS #120 (temperature 0,
# ctx 2048): +46% gen tok/s, +609 MiB peak, identical replies. chat.py uses it for its Gemma E2B Q4_0 entry.
MTP_DRAFT = HOME + "/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf"
MTP_DRAFT_SIZE = 97817664  # bytes; SHA-256 9eba8199... (conv_mtp/ARTIFACTS.md)
MTP_ARGS = ["--model-draft", MTP_DRAFT, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]

def mtp_draft_ok():
    return os.path.isfile(MTP_DRAFT) and os.path.getsize(MTP_DRAFT) == MTP_DRAFT_SIZE

E4B_MTP_DRAFT = HOME + "/models/mtp/mtp-gemma-4-E4B-it-Q8_0.gguf"
E4B_MTP_DRAFT_SIZE = 98653248
E4B_MTP_ARGS = ["--model-draft", E4B_MTP_DRAFT, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]

def e4b_mtp_draft_ok():
    return os.path.isfile(E4B_MTP_DRAFT) and os.path.getsize(E4B_MTP_DRAFT) == E4B_MTP_DRAFT_SIZE

def kill_servers():
    os.system("pkill -f llama-server 2>/dev/null")
    time.sleep(2)
    print("[SERVER] All servers stopped")

def wait_for_server(port=8080, timeout=60, process=None):
    for i in range(timeout):
        if process is not None and process.poll() is not None:
            print(f'[SERVER] ERROR: launched process exited: {process.returncode}')
            return False
        time.sleep(1)
        try:
            r = urllib.request.urlopen(
                "http://127.0.0.1:" + str(port) + "/health", timeout=2)
            if r.status == 200:
                print("[SERVER] Ready after " + str(i+1) + "s")
                return True
        except:
            pass
    print("[SERVER] ERROR: failed to load in " + str(timeout) + "s")
    return False

def verify_server_affinity(proc):
    # taskset execs llama-server, so Popen's PID identifies the exact instance.
    root = Path('/proc') / str(proc.pid)
    if proc.poll() is not None or os.path.realpath(root / 'exe') != os.path.realpath(LLAMA_SERVER):
        raise RuntimeError('launched llama-server is not running')
    tasks = root / 'task'
    tids = sorted(int(p.name) for p in tasks.iterdir())
    if not tids:
        raise RuntimeError('server thread list is empty')
    for tid in tids:
        try:
            cpuset = (tasks / str(tid) / 'cpuset').read_text().strip()
        except OSError as e:
            cpuset = 'unreadable: ' + str(e)
        mask = os.sched_getaffinity(tid)
        if mask != {4, 5, 6, 7}:
            raise RuntimeError(f'thread {tid} CPUs {sorted(mask)}, expected 4-7; cpuset {cpuset}')
        print(f'[SERVER] thread {tid}: CPUs 4-7; cpuset {cpuset}')
    if proc.poll() is not None or tids != sorted(int(p.name) for p in tasks.iterdir()):
        raise RuntimeError('server exited or thread list changed during affinity verification')


def start_server(model_path, port=8080, ctx=2048, extra_args=""):
    if not shutil.which('taskset'):
        print('[SERVER] ERROR: taskset unavailable; nothing started')
        return False
    cmd = ['taskset', '-c', '4-7', LLAMA_SERVER, '-m', model_path,
           '--port', str(port), '--ctx-size', str(ctx), '--threads', THREADS,
           '--threads-batch', BATCH_THREADS, '--parallel', '1', '--swa-full',
           '--cache-ram', '0', '--host', '127.0.0.1'] + shlex.split(extra_args)
    kill_servers()
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                start_new_session=True)
        print("[SERVER] Starting: " + model_path.split("/")[-1])
        print("[SERVER] Waiting...")
        if not wait_for_server(port, process=proc):
            raise RuntimeError('server did not become ready')
        verify_server_affinity(proc)
        print('[SERVER] Launch affinity verified for all threads')
        return True
    except (OSError, RuntimeError) as e:
        print(f'[SERVER] ERROR: {e}')
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                print(f'[SERVER] ERROR: failed to stop rejected server: {cleanup_error}')
        return False

def start_setup(setup_name):
    print("\n" + "="*50)
    print("  Starting: " + setup_name)
    print("="*50)

    # ── MAIN ROBOT MODEL ──────────────────────────────
    if setup_name == "setup_q4":
        # E2B Q4_0 — default robot model
        # Speed: ~12 tok/s (dotprod build) | Size: 2.8GB
        return start_server(
            HOME + "/models/gemma-4-E2B-it-Q4_0.gguf",
            port=8080, ctx=2048
        )

    # ── MAIN ROBOT MODEL + MTP DRAFTER (opt-in) ───────
    elif setup_name == "setup_q4_mtp":
        # setup_q4 plus the DECISIONS #120 drafter flags; not the robot default
        if not mtp_draft_ok():
            print("[SERVER] ERROR: MTP drafter missing or not " + str(MTP_DRAFT_SIZE) +
                  " bytes: " + MTP_DRAFT + "; nothing started")
            return False
        return start_server(
            HOME + "/models/gemma-4-E2B-it-Q4_0.gguf",
            port=8080, ctx=2048, extra_args=" ".join(MTP_ARGS)
        )

    # ── QUALITY MODE ──────────────────────────────────
    elif setup_name == "setup_e4b":
        # E4B Q4_K_M — smarter but slower
        # Speed: ~5.4 tok/s (dotprod build) | Size: 5.0GB
        return start_server(
            HOME + "/models/gemma-4-e4b-it-q4_k_m.gguf",
            port=8080, ctx=2048
        )

    # E4B Q4_0: opt-in, not the robot default.
    elif setup_name == "setup_e4b_q4":
        return start_server(
            HOME + "/models/gemma-4-E4B-it-Q4_0.gguf",
            port=8080, ctx=2048
        )

    elif setup_name == "setup_e4b_q4_mtp":
        if not e4b_mtp_draft_ok():
            print("[SERVER] ERROR: MTP drafter missing or not " + str(E4B_MTP_DRAFT_SIZE) +
                  " bytes: " + E4B_MTP_DRAFT + "; nothing started")
            return False
        return start_server(
            HOME + "/models/gemma-4-E4B-it-Q4_0.gguf",
            port=8080, ctx=2048, extra_args=" ".join(E4B_MTP_ARGS)
        )

    # ── STOP ─────────────────────────────────────────
    elif setup_name == "stop":
        kill_servers()
        return True

    else:
        print("[SERVER] Unknown setup: " + setup_name)
        print("[SERVER] Available: setup_q4, setup_q4_mtp, setup_e4b, stop")
        print("[SERVER] Opt-in: setup_e4b_q4, setup_e4b_q4_mtp")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 server_manager.py [setup_q4|setup_q4_mtp|setup_e4b|setup_e4b_q4|setup_e4b_q4_mtp|stop]")
        sys.exit(1)
    sys.exit(0 if start_setup(sys.argv[1]) else 1)
