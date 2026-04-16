import sys, os, requests, time, base64, subprocess
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

HOME   = '/data/data/com.termux/files/home'
MODELS = {
    '1': {
        'name':    'Gemma 4 E2B Q8 (best quality)',
        'file':    'gemma-4-e2b-it-q8_0.gguf',
        'type':    'gemma',
        'vision':  True,
        'ctx':     4096,
    },
    '2': {
        'name':    'Gemma 4 E2B Q4 (faster)',
        'file':    'gemma-4-e2b-it-q4_k_m.gguf',
        'type':    'gemma',
        'vision':  True,
        'ctx':     4096,
    },
    '3': {
        'name':    'Gemma 4 E4B Q4 (largest)',
        'file':    'gemma-4-e4b-it-q4_k_m.gguf',
        'type':    'gemma',
        'vision':  True,
        'ctx':     4096,
    },
    '4': {
        'name':    'Qwen 2.5 3B (fast)',
        'file':    'qwen2.5-3b-instruct-q4_k_m.gguf',
        'type':    'qwen',
        'vision':  False,
        'ctx':     2048,
    },
    '5': {
        'name':    'Qwen 2.5 1.5B (fastest)',
        'file':    'qwen2.5-1.5b-instruct-q4_k_m.gguf',
        'type':    'qwen',
        'vision':  False,
        'ctx':     2048,
    },
    '6': {
        'name':    'Mistral 7B (powerful)',
        'file':    'mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        'type':    'mistral',
        'vision':  False,
        'ctx':     4096,
    },
}

SERVER  = HOME + '/llama.cpp/build/bin/llama-server'
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
def get_temp():
    try:
        t = int(os.popen('su -c "cat /sys/class/thermal/thermal_zone9/temp"').read().strip()) // 1000
        b = int(os.popen('su -c "cat /sys/class/thermal/thermal_zone25/temp"').read().strip()) // 1000
        cpu_color = C.RED if t > 80 else C.YELLOW if t > 65 else C.GREEN
        return f'CPU:{colored(str(t)+"°C", cpu_color)} Batt:{b}°C'
    except:
        return ''

# ── Server ────────────────────────────────────────────
def kill_server():
    os.system('pkill -f llama-server 2>/dev/null')
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

    print(colored(f'  Loading {m["name"]}...', C.CYAN))
    cmd = f'{SERVER} -m {model_path} --port 8080 --ctx-size {m["ctx"]} --host 127.0.0.1 --threads 8 --threads-batch 8 2>/dev/null &'
    os.system(cmd)

    # Wait for ready
    import urllib.request
    for i in range(40):
        time.sleep(1)
        try:
            urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)
            print(colored(f'  Ready after {i+1}s | {get_temp()}', C.GREEN))
            current_model = model_key
            return True
        except:
            if i % 5 == 0:
                print(f'  Starting... {i+1}s')
    print(colored('  Failed to start server', C.RED))
    return False

def server_running():
    try:
        requests.get('http://127.0.0.1:8080/health', timeout=2)
        return True
    except:
        return False

# ── Prompt builders ───────────────────────────────────
def build_prompt(model_type, messages, image_b64=None):
    if model_type == 'gemma':
        prompt = ''
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n'
            elif role == 'user':
                if image_b64 and msg == messages[-1]:
                    prompt += f'<start_of_turn>user\n[img-1]\n{content}<end_of_turn>\n'
                else:
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
        prompt += '<|im_start|>assistant\n'
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
def chat(user_input, image_path=None):
    m = MODELS[current_model]

    # Add image if provided
    image_b64 = None
    if image_path and m['vision'] and os.path.exists(image_path):
        try:
            from PIL import Image
            import io
            img = Image.open(image_path).convert('RGB')
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=80)
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            print(colored(f'  Image loaded: {image_path}', C.DIM))
        except Exception as e:
            print(colored(f'  Image error: {e}', C.RED))

    # Add to history
    history.append({'role': 'user', 'content': user_input})

    # Build prompt
    prompt = build_prompt(m['type'], history, image_b64)

    payload = {
        'prompt':      prompt,
        'n_predict':   512,
        'temperature': 0.7,
        'stop':        get_stop_tokens(m['type']),
        'stream':      False,
    }

    if image_b64:
        payload['image_data'] = [{'data': image_b64, 'id': 1}]

    # Call
    t0 = time.time()
    try:
        resp = requests.post(URL, json=payload, timeout=120)
        data = resp.json()
        reply = data['content'].strip()
        elapsed = round(time.time()-t0, 1)
        tok_s = round(data['timings']['predicted_per_second'], 1)

        history.append({'role': 'assistant', 'content': reply})
        return reply, elapsed, tok_s
    except Exception as e:
        return f'Error: {e}', 0, 0

# ── UI ────────────────────────────────────────────────
def show_header():
    os.system('clear')
    print(colored('='*50, C.CYAN))
    print(colored('  ROBOT LLM CHAT', C.BOLD + C.CYAN))
    print(colored('='*50, C.CYAN))
    if current_model:
        m = MODELS[current_model]
        vision = colored('👁 Vision ON', C.GREEN) if m['vision'] else colored('No vision', C.DIM)
        print(f'  Model: {colored(m["name"], C.YELLOW)}')
        print(f'  {vision} | {get_temp()}')
    print(colored('='*50, C.CYAN))

def show_menu():
    show_header()
    print(f'\n  {colored("COMMANDS:", C.BOLD)}')
    print(f'  {colored("/switch", C.CYAN)}   — switch model')
    print(f'  {colored("/photo", C.CYAN)}    — attach photo (Gemma only)')
    print(f'  {colored("/camera", C.CYAN)}   — take photo with phone camera')
    print(f'  {colored("/clear", C.CYAN)}    — clear conversation')
    print(f'  {colored("/history", C.CYAN)}  — show conversation')
    print(f'  {colored("/temp", C.CYAN)}     — show temperatures')
    print(f'  {colored("/kill", C.CYAN)}     — kill server')
    print(f'  {colored("/quit", C.CYAN)}     — exit')
    print()

def select_model():
    print(f'\n  {colored("AVAILABLE MODELS:", C.BOLD)}')
    for key, m in MODELS.items():
        model_path = f'{HOME}/models/{m["file"]}'
        exists = os.path.exists(model_path) and os.path.getsize(model_path) > 1000
        status = colored('✅', C.GREEN) if exists else colored('❌ missing', C.RED)
        vision = colored(' 👁', C.CYAN) if m['vision'] else ''
        size = f'{round(os.path.getsize(model_path)/1024/1024/1024, 1)}GB' if exists else '?'
        print(f'  {colored(key, C.YELLOW)}) {m["name"]}{vision} [{size}] {status}')

    print(f'\n  Enter number (or ENTER to cancel): ', end='')
    choice = input().strip()
    if choice in MODELS:
        model_path = f'{HOME}/models/{MODELS[choice]["file"]}'
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            print(colored('  Model not available', C.RED))
            time.sleep(2)
            return False
        return start_server(choice)
    return False

def take_camera_photo():
    path = HOME + '/chat_photo.jpg'
    print(colored('  Taking photo...', C.CYAN))
    os.system(f'termux-camera-photo {path} 2>/dev/null')
    time.sleep(1)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(colored(f'  Photo taken: {path}', C.GREEN))
        return path
    print(colored('  Photo failed', C.RED))
    return None

# ── Main loop ─────────────────────────────────────────
def main():
    global history

    show_header()
    print(colored('\n  Welcome! Select a model to start.\n', C.GREEN))

    # Auto-select if server already running
    if server_running() and current_model is None:
        print(colored('  Server already running — attach to it? (y/n): ', C.YELLOW), end='')
        if input().strip().lower() == 'y':
            # Try to detect which model
            for key in MODELS:
                pass  # just use it as-is
            globals()['current_model'] = '1'  # assume Gemma E2B

    if not current_model:
        select_model()

    if not current_model:
        print(colored('  No model loaded. Exiting.', C.RED))
        return

    show_menu()

    pending_image = None

    while True:
        # Prompt
        temp = get_temp()
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
                history = []
                pending_image = None
                select_model()
                show_menu()

            elif cmd == '/clear':
                history = []
                pending_image = None
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

            elif cmd == '/photo':
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    path = parts[1].strip()
                    if os.path.exists(path):
                        pending_image = path
                        print(colored(f'  Image queued: {path}', C.GREEN))
                    else:
                        print(colored(f'  File not found: {path}', C.RED))
                else:
                    print(colored('  Usage: /photo /path/to/image.jpg', C.YELLOW))

            elif cmd == '/camera':
                if MODELS[current_model]['vision']:
                    pending_image = take_camera_photo()
                else:
                    print(colored('  Current model does not support vision', C.RED))

            else:
                print(colored(f'  Unknown command: {cmd}', C.RED))

            continue

        # Check server still running
        if not server_running():
            print(colored('  Server died — restarting...', C.RED))
            start_server(current_model)

        # Chat
        print(f'\n{colored(model_name, C.CYAN + C.BOLD)}: ', end='', flush=True)
        reply, elapsed, tok_s = chat(user_input, pending_image)
        pending_image = None

        print(reply)
        print(colored(f'\n  [{elapsed}s | {tok_s}tok/s | {get_temp()}]', C.DIM))

    print(colored('\n  Goodbye!', C.CYAN))
    print(colored('  Server left running. Use /kill or: pkill -f llama-server', C.DIM))

if __name__ == '__main__':
    main()
