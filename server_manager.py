import os
import time
import subprocess

HOME = "/data/data/com.termux/files/home"
LLAMA_SERVER = HOME + "/llama.cpp/build/bin/llama-server"

def kill_servers():
    os.system("pkill -f llama-server 2>/dev/null")
    time.sleep(2)
    print("[SERVER] All servers stopped")

def start_server(model_path, port=8080, ctx=2048, extra_args=""):
    kill_servers()
    cmd = (
        LLAMA_SERVER +
        " -m " + model_path +
        " --port " + str(port) +
        " --ctx-size " + str(ctx) +
        " --host 127.0.0.1 " +
        extra_args +
        " 2>/dev/null &"
    )
    os.system(cmd)
    print("[SERVER] Starting model: " + model_path.split("/")[-1])
    print("[SERVER] Waiting for model to load...")
    for i in range(60):
        time.sleep(1)
        try:
            import urllib.request
            r = urllib.request.urlopen("http://127.0.0.1:" + str(port) + "/health", timeout=2)
            if r.status == 200:
                print("[SERVER] Ready after " + str(i+1) + "s")
                return True
        except:
            pass
    print("[SERVER] ERROR: model failed to load in 60s")
    return False

def start_setup(setup_name):
    print("\n" + "="*50)
    print("  Starting: " + setup_name)
    print("="*50)

    if setup_name == "setup_a":
        return start_server(
            HOME + "/models/qwen2.5-3b-instruct-q4_k_m.gguf",
            port=8080, ctx=2048
        )
    elif setup_name == "setup_b":
        return start_server(
            HOME + "/models/gemma-4-e4b-it-q4_k_m.gguf",
            port=8080, ctx=2048
        )
    elif setup_name == "setup_q4":
        start_server(
            HOME + "/models/gemma-4-e2b-it-q4_k_m.gguf",
            port=8080, ctx=2048
        )
    elif setup_name == "setup_q4_no_mmproj":
        start_server(
            HOME + "/models/gemma-4-e2b-it-q4_k_m.gguf",
            port=8080, ctx=2048
        )
    elif setup_name == "setup_c":
        return start_server(
            HOME + "/models/gemma-4-e2b-it-q8_0.gguf",
            port=8080, ctx=2048
        )
    elif setup_name == "setup_d_old":
        # Qwen on 8080 for LLM, E2B on 8081 for vision
        kill_servers()
        cmd1 = LLAMA_SERVER + " -m " + HOME + "/models/qwen2.5-3b-instruct-q4_k_m.gguf --port 8080 --ctx-size 2048 --threads 8 --threads-batch 8 --host 127.0.0.1 2>/dev/null &"
        os.system(cmd1)
        cmd2 = LLAMA_SERVER + " -m " + HOME + "/models/gemma-4-e2b-it-q8_0.gguf --port 8081 --ctx-size 2048 --threads 8 --threads-batch 8 --host 127.0.0.1 2>/dev/null &"
        os.system(cmd2)
        print("[SERVER] Starting Qwen 3B on 8080 + E2B on 8081...")
        import urllib.request
        for port in [8080, 8081]:
            for i in range(60):
                time.sleep(1)
                try:
                    r = urllib.request.urlopen("http://127.0.0.1:" + str(port) + "/health", timeout=2)
                    if r.status == 200:
                        print("[SERVER] Port " + str(port) + " ready after " + str(i+1) + "s")
                        break
                except:
                    pass
        return True
    elif setup_name == "setup_f":
        return start_server(
            HOME + "/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            port=8080, ctx=2048
        )
    elif setup_name == "setup_e":
        return start_server(
            HOME + "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            port=8080, ctx=2048
        )
    elif setup_name == "setup_d":
        # YOLO detection + E2B for rich description
        return start_server(
            HOME + "/models/gemma-4-e2b-it-q8_0.gguf",
            port=8080, ctx=2048
        )
    else:
        print("[SERVER] Unknown setup: " + setup_name)
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 server_manager.py [setup_a|setup_b|setup_c|setup_d|stop]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "stop":
        kill_servers()
    else:
        start_setup(cmd)
