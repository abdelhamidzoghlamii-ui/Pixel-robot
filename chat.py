import sys, os, requests, time, subprocess, shutil
from pathlib import Path
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

HOME   = '/data/data/com.termux/files/home'
MODEL_FILES = (
    ('Gemma 4 E2B Q4_0', 'gemma-4-E2B-it-Q4_0.gguf', 'gemma', 4096),
    ('Gemma 4 E2B Q4_K_M', 'gemma-4-e2b-it-q4_k_m.gguf', 'gemma', 4096),
    ('Gemma 4 E2B Q8_0', 'gemma-4-e2b-it-q8_0.gguf', 'gemma', 2048),
    ('Gemma 4 E4B Q4_K_M', 'gemma-4-e4b-it-q4_k_m.gguf', 'gemma', 2048),
    ('Qwen3.5 2B Q4_K_M', 'qwen35/Qwen3.5-2B-Q4_K_M.gguf', 'qwen', 4096),
    ('Qwen3.5 4B Q4_K_M', 'qwen35/Qwen3.5-4B-Q4_K_M.gguf', 'qwen', 2048),
    ('Qwen3.5 4B Q5_K_M', 'qwen35/Qwen3.5-4B-Q5_K_M.gguf', 'qwen', 2048),
    ('Qwen3.5 4B Q6_K', 'qwen35/Qwen3.5-4B-Q6_K.gguf', 'qwen', 2048),
    ('Mistral 7B Q4_K_M', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf', 'mistral', 2048),
)
available_models = [
    entry for entry in MODEL_FILES
    if (Path(HOME) / 'models' / entry[1]).is_file()
    and (Path(HOME) / 'models' / entry[1]).stat().st_size > 1000
]
MODELS = {
    str(i): {'name': name, 'file': file, 'type': family, 'ctx': ctx}
    for i, (name, file, family, ctx) in enumerate(available_models, 1)
}

SERVER  = HOME + '/llama.cpp-b1609-dotprod/build/bin/llama-server'
SERVER_LOG = HOME + '/robot-chat-server.log'
URL     = 'http://127.0.0.1:8080/completion'
current_model = None
history = []

# ── Colors ────────────────────────────────────────────
class C:
    BLUE    = '\033[94m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    CYAN    = '\033[96m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    RESET   = '\033[0m'

def colored(text, color):
    return f'{color}{text}{C.RESET}'

# ── Temperature ───────────────────────────────────────
def get_ram():
    try:
        with open('/proc/meminfo') as info:
            kb = next(int(line.split()[1]) for line in info if line.startswith('MemAvailable:'))
        return f'RAM:{kb / 1024 / 1024:.1f}GiB available'
    except (OSError, StopIteration, ValueError):
        return ''

def get_temp():
    ram = get_ram()
    try:
        t = int(os.popen('su -c "cat /sys/class/thermal/thermal_zone9/temp"').read().strip()) // 1000
        b = int(os.popen('su -c "cat /sys/class/thermal/thermal_zone25/temp"').read().strip()) // 1000
        cpu_color = C.RED if t > 80 else C.YELLOW if t > 65 else C.GREEN
        return f'CPU:{colored(str(t)+"°C", cpu_color)} Batt:{b}°C {ram}'
    except:
        return ram

# ── Server ────────────────────────────────────────────
def kill_server():
    subprocess.run(['pkill', '-f', '^' + SERVER + ' '], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    time.sleep(1)
    print(colored('  Server killed', C.YELLOW))

def start_server(model_key):
    global current_model
    m = MODELS[model_key]
    model_path = f'{HOME}/models/{m["file"]}'

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        print(colored(f'  Model file not found or incomplete: {m["file"]}', C.RED))
        return False

    kill_server()
    time.sleep(1)
    if server_running():
        print(colored('  Port 8080 is still in use by another server', C.RED))
        return False

    print(colored(f'  Loading {m["name"]}...', C.CYAN))
    cmd = [
        SERVER, '-m', model_path, '--port', '8080',
        '--ctx-size', str(m['ctx']), '--host', '127.0.0.1',
        '--threads', '4', '--threads-batch', '4', '--parallel', '1',
        '--swa-full', '--cache-ram', '0',
    ]
    if not shutil.which('taskset'):
        print(colored('  taskset unavailable; using Android CPU scheduling', C.YELLOW))
    elif {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
        cmd[:0] = ['taskset', '-c', '4-7']
    else:
        print(colored('  Cores 4-7 unavailable; using Android CPU scheduling', C.YELLOW))
    try:
        with open(SERVER_LOG, 'w') as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
    except OSError as e:
        print(colored(f'  Failed to launch server: {e}', C.RED))
        return False

    # Wait for ready
    import urllib.request
    for i in range(40):
        time.sleep(1)
        if server.poll() is not None:
            break
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2) as response:
                if response.status != 200:
                    continue
            print(colored(f'  Ready after {i+1}s | {get_temp()}', C.GREEN))
            current_model = model_key
            return True
        except:
            if i % 5 == 0:
                print(f'  Starting... {i+1}s')
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    print(colored(f'  Failed to start server (exit {server.returncode}); see {SERVER_LOG}', C.RED))
    return False

def server_running():
    try:
        requests.get('http://127.0.0.1:8080/health', timeout=2)
        return True
    except:
        return False

# ── Prompt builders ───────────────────────────────────
def build_prompt(model_type, messages):
    if model_type == 'gemma':
        prompt = ''
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n'
            elif role == 'user':
                prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n'
            elif role == 'assistant':
                prompt += f'<start_of_turn>model\n{content}<end_of_turn>\n'
        prompt += '<start_of_turn>model\n'
        return prompt

    elif model_type == 'qwen':
        prompt = ''
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt += f'<|im_start|>system\n{content}<|im_end|>\n'
            elif role == 'user':
                prompt += f'<|im_start|>user\n{content}<|im_end|>\n'
            elif role == 'assistant':
                prompt += f'<|im_start|>assistant\n{content}<|im_end|>\n'
        prompt += '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return prompt

    elif model_type == 'mistral':
        prompt = ''
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                prompt += f'[INST] {content} [/INST]'
            elif role == 'assistant':
                prompt += f' {content}</s>'
        return prompt

    return ''

def get_stop_tokens(model_type):
    stops = {
        'gemma':   ['<end_of_turn>'],
        'qwen':    ['<|im_end|>'],
        'mistral': ['</s>', '[INST]'],
    }
    return stops.get(model_type, [])

# ── Chat ──────────────────────────────────────────────
def chat(user_input):
    m = MODELS[current_model]

    # Add to history
    history.append({'role': 'user', 'content': user_input})

    # Build prompt
    prompt = build_prompt(m['type'], history)

    payload = {
        'prompt':      prompt,
        'n_predict':   512,
        'temperature': 0.7,
        'stop':        get_stop_tokens(m['type']),
        'stream':      False,
    }

    # Call
    t0 = time.time()
    try:
        resp = requests.post(URL, json=payload, timeout=120)
        data = resp.json()
        if 'content' not in data:
            raise ValueError(f"{data.get('error', data)}; use /clear to start a fresh conversation")
        reply = data['content'].strip()
        elapsed = round(time.time()-t0, 1)
        tok_s = round(data['timings']['predicted_per_second'], 1)

        history.append({'role': 'assistant', 'content': reply})
        return reply, elapsed, tok_s
    except Exception as e:
        history.pop()
        return f'Error: {e}', 0, 0

# ── UI ────────────────────────────────────────────────
def show_header():
    os.system('clear')
    print(colored('='*50, C.CYAN))
    print(colored('  ROBOT LLM CHAT', C.BOLD + C.CYAN))
    print(colored('='*50, C.CYAN))
    if current_model:
        m = MODELS[current_model]
        print(f'  Model: {colored(m["name"], C.YELLOW)}')
        print(f'  No vision | {get_temp()}')
    print(colored('='*50, C.CYAN))

def show_menu():
    show_header()
    print(f'\n  {colored("COMMANDS:", C.BOLD)}')
    print(f'  {colored("/switch", C.CYAN)}   — switch model')
    print(f'  {colored("/clear", C.CYAN)}    — clear conversation')
    print(f'  {colored("/history", C.CYAN)}  — show conversation')
    print(f'  {colored("/temp", C.CYAN)}     — show temperatures')
    print(f'  {colored("/kill", C.CYAN)}     — kill server')
    print(f'  {colored("/quit", C.CYAN)}     — exit')
    print()

def select_model():
    if not MODELS:
        print(colored('  No supported model files found in ~/models', C.RED))
        return False
    print(f'\n  {colored("AVAILABLE MODELS:", C.BOLD)}')
    for key, m in MODELS.items():
        model_path = f'{HOME}/models/{m["file"]}'
        exists = os.path.exists(model_path) and os.path.getsize(model_path) > 1000
        status = colored('✅', C.GREEN) if exists else colored('❌ missing', C.RED)
        size = f'{round(os.path.getsize(model_path)/1024/1024/1024, 1)}GB' if exists else '?'
        print(f'  {colored(key, C.YELLOW)}) {m["name"]} [{size}] {status}')

    print(f'\n  Enter number (or ENTER to cancel): ', end='')
    choice = input().strip()
    if choice in MODELS:
        model_path = f'{HOME}/models/{MODELS[choice]["file"]}'
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            print(colored('  Model not available', C.RED))
            time.sleep(2)
            return False
        return start_server(choice)
    return None

# ── Main loop ─────────────────────────────────────────
def main():
    global history

    show_header()
    print(colored('\n  Welcome! Select a model to start.\n', C.GREEN))

    if not current_model:
        select_model()

    if not current_model:
        print(colored('  No model loaded. Exiting.', C.RED))
        return

    show_menu()

    while True:
        # Prompt
        model_name = MODELS[current_model]['name'].split('(')[0].strip()
        print(f'\n{colored("You", C.GREEN + C.BOLD)}: ', end='')

        try:
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith('/'):
            cmd = user_input.lower().split()[0]

            if cmd == '/quit' or cmd == '/exit':
                break

            elif cmd == '/kill':
                kill_server()

            elif cmd == '/switch':
                switched = select_model()
                if switched is False:
                    print(colored('  Model switch failed; exiting chat', C.RED))
                    break
                if switched:
                    history = []
                show_menu()

            elif cmd == '/clear':
                history = []
                print(colored('  Conversation cleared', C.YELLOW))

            elif cmd == '/history':
                print()
                for msg in history:
                    role_color = C.GREEN if msg['role'] == 'user' else C.CYAN
                    print(colored(f'{msg["role"].upper()}:', role_color))
                    print(f'  {msg["content"][:200]}...' if len(msg['content']) > 200 else f'  {msg["content"]}')
                    print()

            elif cmd == '/temp':
                print(colored(f'\n  {get_temp()}', C.CYAN))
                # Show all zones
                zones = {'BIG':9,'MID':10,'LITTLE':11,'GPU':12,'Battery':25}
                for name, zone in zones.items():
                    try:
                        t = int(os.popen(f'su -c "cat /sys/class/thermal/thermal_zone{zone}/temp"').read().strip()) // 1000
                        bar = colored('●', C.RED) if t>80 else colored('●', C.YELLOW) if t>65 else colored('●', C.GREEN)
                        print(f'  {bar} {name:<8}: {t}°C')
                    except:
                        pass

            elif cmd in ('/photo', '/camera'):
                print(colored('  Image chat is unavailable with the current local models', C.RED))

            else:
                print(colored(f'  Unknown command: {cmd}', C.RED))

            continue

        # Check server still running
        if not server_running():
            print(colored('  Server died — restarting...', C.RED))
            if not start_server(current_model):
                break

        # Chat
        print(f'\n{colored(model_name, C.CYAN + C.BOLD)}: ', end='', flush=True)
        reply, elapsed, tok_s = chat(user_input)

        print(reply)
        print(colored(f'\n  [{elapsed}s | {tok_s}tok/s | {get_temp()}]', C.DIM))

    print(colored('\n  Goodbye!', C.CYAN))
    print(colored('  Stop the server from Termux with: pkill -f llama-server', C.DIM))

if __name__ == '__main__':
    main()
