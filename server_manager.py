import os
import time
import urllib.request

HOME = "/data/data/com.termux/files/home"
LLAMA_SERVER = HOME + "/llama.cpp/build/bin/llama-server"

# Optimal config (benchmarked on Pixel 7 Tensor G2):
# --threads 4 --threads-batch 4 → 11-12 tok/s for E2B
# --parallel 1                  → no slot splitting
# --cache-ram 0                 → disables broken SWA cache
THREADS = "4"
BATCH_THREADS = "4"

def kill_servers():
    os.system("pkill -f llama-server 2>/dev/null")
    time.sleep(2)
    print("[SERVER] All servers stopped")

def wait_for_server(port=8080, timeout=60):
    for i in range(timeout):
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

def start_server(model_path, port=8080, ctx=2048, extra_args=""):
    kill_servers()
    cmd = (
        LLAMA_SERVER +
        " -m " + model_path +
        " --port " + str(port) +
        " --ctx-size " + str(ctx) +
        " --threads " + THREADS +
        " --threads-batch " + BATCH_THREADS +
        " --parallel 1" +
        " --cache-ram 0" +
        " --host 127.0.0.1 " +
        extra_args +
        " 2>/dev/null &"
    )
    os.system(cmd)
    print("[SERVER] Starting: " + model_path.split("/")[-1])
    print("[SERVER] Waiting...")
    return wait_for_server(port)

def start_setup(setup_name):
    print("\n" + "="*50)
    print("  Starting: " + setup_name)
    print("="*50)

    # ── MAIN ROBOT MODEL ──────────────────────────────
    if setup_name == "setup_q4":
        # E2B Q4_K_M — default robot model
        # Speed: 11-12 tok/s | Size: 3.3GB
        return start_server(
            HOME + "/models/gemma-4-e2b-it-q4_k_m.gguf",
            port=8080, ctx=2048
        )

    # ── QUALITY MODE ──────────────────────────────────
    elif setup_name == "setup_e4b":
        # E4B Q4_K_M — smarter but slower
        # Speed: 7.2 tok/s | Size: 5.0GB
        return start_server(
            HOME + "/models/gemma-4-e4b-it-q4_k_m.gguf",
            port=8080, ctx=2048
        )

    # ── LIGHTWEIGHT MODELS ────────────────────────────
    elif setup_name == "setup_qwen3b":
        # Qwen 2.5 3B — fast fallback
        return start_server(
            HOME + "/models/qwen2.5-3b-instruct-q4_k_m.gguf",
            port=8080, ctx=2048
        )

    elif setup_name == "setup_qwen1b":
        # Qwen 2.5 1.5B — fastest, lowest quality
        return start_server(
            HOME + "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            port=8080, ctx=2048
        )

    # ── STOP ─────────────────────────────────────────
    elif setup_name == "stop":
        kill_servers()
        return True

    else:
        print("[SERVER] Unknown setup: " + setup_name)
        print("[SERVER] Available: setup_q4, setup_e4b, setup_qwen3b, setup_qwen1b, stop")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 server_manager.py [setup_q4|setup_e4b|setup_qwen3b|setup_qwen1b|stop]")
        sys.exit(1)
    start_setup(sys.argv[1])
