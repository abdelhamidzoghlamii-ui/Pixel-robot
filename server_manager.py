import os
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
        " --swa-full" +
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
        # E2B Q4_0 — default robot model
        # Speed: ~12 tok/s (dotprod build) | Size: 2.8GB
        return start_server(
            HOME + "/models/gemma-4-E2B-it-Q4_0.gguf",
            port=8080, ctx=2048
        )

    # ── QUALITY MODE ──────────────────────────────────
    elif setup_name == "setup_e4b":
        # E4B Q4_K_M — smarter but slower
        # Speed: ~5.4 tok/s (dotprod build) | Size: 5.0GB
        return start_server(
            HOME + "/models/gemma-4-e4b-it-q4_k_m.gguf",
            port=8080, ctx=2048
        )

    # ── STOP ─────────────────────────────────────────
    elif setup_name == "stop":
        kill_servers()
        return True

    else:
        print("[SERVER] Unknown setup: " + setup_name)
        print("[SERVER] Available: setup_q4, setup_e4b, stop")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 server_manager.py [setup_q4|setup_e4b|stop]")
        sys.exit(1)
    start_setup(sys.argv[1])
