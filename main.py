import sys, os, time, json, threading, requests
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

from detect_person import detect_scene, scene_to_text, person_direction
from stereo_depth import stereo_scan, scene_with_depth, estimate_distance_single
from stereo_depth import stereo_scan, scene_with_depth

HOME    = '/data/data/com.termux/files/home'
PHOTO_A = HOME + '/robot_photo_a.jpg'
PHOTO_B = HOME + '/robot_photo_b.jpg'

# ── Config ────────────────────────────────────────────
OBSTACLE_DIST    = 25    # cm — stop if closer
PERSON_STOP_DIST = 80    # cm — stop when person this close
GEMMA_INTERVAL   = 10    # cycles between Gemma checks
STEREO_BASELINE  = 5.0   # cm — strafe for depth
MOTOR_SPEED      = 130   # default motor speed
CYCLE_MOVE_TIME  = 1.5   # seconds per move

# ── Thermal ───────────────────────────────────────────
def get_temp():
    try:
        return int(os.popen('su -c "cat /sys/class/thermal/thermal_zone9/temp"').read().strip()) // 1000
    except:
        return 0

# ── Camera ────────────────────────────────────────────
def take_photo(path):
    os.system(f'termux-camera-photo {path} 2>/dev/null')
    time.sleep(0.5)
    return os.path.exists(path) and os.path.getsize(path) > 1000

# ── LLM (Gemma E2B) ───────────────────────────────────
GEMMA_URL = 'http://127.0.0.1:8080/completion'
GEMMA_SYS = (
    'You are a robot assistant. Reply with ONE word: FORWARD, LEFT, RIGHT, BACK, STOP, or SPEAK. '
    'Then one short sentence explaining why. '
    'Use STOP if mission is complete. Use SPEAK to say something to a person.'
)

def gemma_decide(context):
    prompt = (
        '<start_of_turn>user\n' + GEMMA_SYS + '\n\n' + context +
        '<end_of_turn>\n<start_of_turn>model\n'
    )
    try:
        resp = requests.post(GEMMA_URL, json={
            'prompt': prompt, 'n_predict': 40,
            'temperature': 0.1, 'stop': ['<end_of_turn>']
        }, timeout=30)
        return resp.json()['content'].strip()
    except:
        return 'FORWARD default'

def gemma_identify(scene_desc, mission):
    prompt = (
        '<start_of_turn>user\n'
        f'Mission: {mission}\n'
        f'I see: {scene_desc}\n'
        'Is the mission complete? Reply YES or NO and why.'
        '<end_of_turn>\n<start_of_turn>model\n'
    )
    try:
        resp = requests.post(GEMMA_URL, json={
            'prompt': prompt, 'n_predict': 60,
            'temperature': 0.1, 'stop': ['<end_of_turn>']
        }, timeout=30)
        return resp.json()['content'].strip()
    except:
        return 'NO cannot connect'

# ── Voice (Whisper) ───────────────────────────────────
WHISPER_BIN   = HOME + '/whisper.cpp/build/bin/whisper-cli'
WHISPER_MODEL = HOME + '/whisper.cpp/models/ggml-base.bin'
RAW_FILE      = HOME + '/rec_raw.amr'
WAV_FILE      = HOME + '/rec.wav'

def listen():
    import subprocess
    for f in [RAW_FILE, WAV_FILE]:
        if os.path.exists(f): os.remove(f)
    subprocess.Popen(['termux-microphone-record', '-f', RAW_FILE],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('[VOICE] Listening 5s...')
    time.sleep(5)
    subprocess.run(['termux-microphone-record', '-q'], capture_output=True)
    time.sleep(0.5)
    subprocess.run(['ffmpeg', '-i', RAW_FILE, '-ar', '16000', '-ac', '1',
                    '-c:a', 'pcm_s16le', WAV_FILE, '-y'], capture_output=True)
    r = subprocess.run([WHISPER_BIN, '-m', WHISPER_MODEL, '-f', WAV_FILE,
                        '--no-timestamps', '-l', 'en'], capture_output=True, text=True)
    text = ' '.join(l.strip() for l in r.stdout.splitlines()
                    if l.strip() and not l.startswith('[') and not l.startswith('whisper'))
    return text.strip()

# ── Command parser (Gemma E2B on port 8080) ──────────
# Gemma handles everything — no separate Qwen needed
PARSE_SYS = (
    'You are a robot controller. Output ONLY a JSON array of actions. '
    'Use exactly these formats: '
    '{"type":"navigate_to","room":"<room>"} or '
    '{"type":"find_person","name":"<name>"} or '
    '{"type":"say","message":"<text>"}. '
    'Use minimum actions needed. Output JSON only, no explanation.'
)

def parse_command(text):
    prompt = (
        '<start_of_turn>user\n' + PARSE_SYS +
        '\nCommand: ' + text +
        '<end_of_turn>\n<start_of_turn>model\n'
    )
    try:
        resp = requests.post(GEMMA_URL, json={
            'prompt': prompt, 'n_predict': 200,
            'temperature': 0.1, 'stop': ['<end_of_turn>']
        }, timeout=15)
        raw = resp.json()['content']
        s = raw.find('['); e = raw.rfind(']') + 1
        if s >= 0 and e > 0:
            return json.loads(raw[s:e])
    except:
        pass
    return []

# ── TTS ───────────────────────────────────────────────
def speak(text):
    print(f'[SPEAK] {text}')
    os.system(f'termux-tts-speak "{text}" &')

# ── Main Robot Class ──────────────────────────────────
class Robot:
    def __init__(self, motors=None):
        self.motors      = motors
        self.cycle       = 0
        self.mission     = None
        self.target      = None   # person name or room
        self.last_moves  = []
        self.scene_log   = []
        self.known_rooms = {}
        self.running     = False
        self.state       = 'idle'  # idle / navigating / searching / found

    def move(self, cmd, duration=CYCLE_MOVE_TIME):
        print(f'  [MOTOR] {cmd}')
        if self.motors:
            if cmd == 'FORWARD':  self.motors.forward(MOTOR_SPEED, duration)
            elif cmd == 'LEFT':   self.motors.rotate_left(120, duration)
            elif cmd == 'RIGHT':  self.motors.rotate_right(120, duration)
            elif cmd == 'BACK':   self.motors.backward(MOTOR_SPEED, duration)
            elif cmd == 'STOP':   self.motors.stop()
            elif cmd == 'STRAFE_RIGHT': self.motors.strafe_right(100, duration)
            elif cmd == 'STRAFE_LEFT':  self.motors.strafe_left(100, duration)
        else:
            time.sleep(duration)
        self.last_moves.append(cmd)
        if len(self.last_moves) > 10:
            self.last_moves.pop(0)

    def get_distance(self):
        if self.motors:
            return self.motors.get_distance()
        return 999

    def navigate_rules(self, results, distance):
        """Fast Python navigation — no LLM."""
        # Safety first
        if distance < 15:
            return 'BACK'
        if distance < OBSTACLE_DIST:
            return 'LEFT'

        # Person found — move toward them
        labels = [r[0] for r in results]
        if 'person' in labels:
            direction = person_direction(results)
            if direction:
                # Check if close enough
                for r in results:
                    if r[0] == 'person':
                        dist_est = estimate_distance_single('person', r[7])
                        if dist_est and dist_est < PERSON_STOP_DIST:
                            return 'STOP'
                return direction

        # Room detection
        if 'refrigerator' in labels:
            room = 'kitchen'
        elif 'couch' in labels or 'tv' in labels:
            room = 'living room'
        elif 'bed' in labels:
            room = 'bedroom'
        elif 'toilet' in labels:
            room = 'bathroom'
        else:
            room = None

        if room and room not in self.known_rooms:
            self.known_rooms[room] = time.strftime('%H:%M')
            print(f'  [MAP] Found: {room}')

        return 'FORWARD'

    def gemma_context(self, scene):
        last5 = self.scene_log[-5:] if self.scene_log else ['none']
        return (
            f"Mission: {self.mission}\n"
            f"Target: {self.target}\n"
            f"Current scene: {scene}\n"
            f"Last 5 scenes:\n" +
            '\n'.join([f'  - {s}' for s in last5]) +
            f"\nLast moves: {', '.join(self.last_moves[-5:])}\n"
            f"Known rooms: {self.known_rooms}\n"
            f"Distance ahead: {self.get_distance()}cm"
        )

    def run_cycle(self, photo_path=None):
        self.cycle += 1
        temp = get_temp()
        distance = self.get_distance()
        print(f'\n── Cycle {self.cycle} | {temp}°C | dist:{distance}cm ──')

        # Vision
        if photo_path:
            path = photo_path
        else:
            if not take_photo(PHOTO_A):
                print('  [CAM] Photo failed')
                return 'FORWARD'
            path = PHOTO_A

        results = detect_scene(path)
        scene = scene_to_text(results)
        self.scene_log.append(scene)
        if len(self.scene_log) > 10:
            self.scene_log.pop(0)
        print(f'  [YOLO] {scene}')

        # Fast navigation rules
        move = self.navigate_rules(results, distance)
        print(f'  [NAV] {move}')

        # Gemma check — every N cycles or when person found
        person_found = any(r[0] == 'person' for r in results)
        use_gemma = (self.cycle % GEMMA_INTERVAL == 0) or (person_found and move == 'STOP')

        if use_gemma and self.mission:
            print(f'  [GEMMA] Consulting...')
            context = self.gemma_context(scene)
            response = gemma_decide(context)
            print(f'  [GEMMA] {response}')
            words = response.upper().split()
            for w in words:
                if w in ['FORWARD','LEFT','RIGHT','BACK','STOP','SPEAK']:
                    move = w
                    break
            if move == 'SPEAK':
                # Extract message after first word
                msg_parts = response.split(' ', 1)
                if len(msg_parts) > 1:
                    speak(msg_parts[1])
                move = 'STOP'

        # Execute move (motors cool during movement)
        if move != 'STOP':
            self.move(move)
        else:
            print('  [STOP] Mission complete or waiting')
            self.state = 'found'

        return move

    def run_mission(self, mission_text):
        """Run until mission complete or stopped."""
        self.mission = mission_text
        self.state = 'navigating'
        self.running = True
        print(f'\n[ROBOT] Mission: {mission_text}')

        while self.running and self.state != 'found':
            # Thermal protection
            if get_temp() > 80:
                print('[THERMAL] Too hot, pausing 5s...')
                if self.motors: self.motors.stop()
                time.sleep(5)
                continue

            move = self.run_cycle()

            if move == 'STOP':
                break

        print('[ROBOT] Mission ended')
        if self.motors: self.motors.stop()

    def stereo_depth_scan(self):
        """Strafe 5cm right, take two photos, return depth data."""
        print('[STEREO] Starting depth scan...')
        take_photo(PHOTO_A)
        self.move('STRAFE_RIGHT', duration=0.3)  # ~5cm
        time.sleep(0.3)
        take_photo(PHOTO_B)
        self.move('STRAFE_LEFT', duration=0.3)   # return
        enhanced = stereo_scan(PHOTO_A, PHOTO_B, STEREO_BASELINE)
        depth_scene = scene_with_depth(enhanced)
        print(f'[STEREO] {depth_scene}')
        return enhanced, depth_scene

    def voice_command(self):
        """Listen for voice command and parse it."""
        speak('Ready')
        text = listen()
        print(f'[VOICE] Heard: {text}')
        if not text:
            return []
        actions = parse_command(text)
        print(f'[VOICE] Actions: {actions}')
        return actions

if __name__ == '__main__':
    print('Robot system initialized')
    print('Testing scene detection...')
    robot = Robot()

    # Test with scene photo
    if os.path.exists('test_photos/scene_test.jpg'):
        result = robot.run_cycle('test_photos/scene_test.jpg')
        print(f'Test cycle result: {result}')
    else:
        print('No test photo found')

    print(f'\nKnown rooms: {robot.known_rooms}')
    print(f'Last moves: {robot.last_moves}')
    print('\nRobot ready. Connect motors and run mission.')
