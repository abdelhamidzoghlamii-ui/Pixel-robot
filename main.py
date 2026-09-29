import sys, os, time, json, threading, requests, subprocess
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

from detect_person import get_detector, detect_scene, scene_from_detections, scene_to_text, person_direction
from detector_size_policy import SizePolicy
from robotcam_reader import read_frame
from stereo_depth import stereo_scan, scene_with_depth, estimate_distance_single

HOME    = '/data/data/com.termux/files/home'
PHOTO_A = HOME + '/robot_photo_a.jpg'
PHOTO_B = HOME + '/robot_photo_b.jpg'

# ── Config ────────────────────────────────────────────
OBSTACLE_DIST    = 25    # cm — stop if closer
PERSON_STOP_DIST = 80    # cm — stop when person this close
STEREO_BASELINE  = 5.0   # cm — strafe for depth
MOTOR_SPEED      = 130   # default motor speed
CYCLE_MOVE_TIME  = 1.5   # seconds per move
CAMERA_SOURCE    = 'robotcam'  # 'termux_photo' is the rollback source
LARGE_FRAME_INTERVAL_S = 5
PERSON_HEIGHT_FRACTION = 1/3

# ── Thermal ───────────────────────────────────────────
def get_temp():
    try:
        return int(os.popen('su -c "cat /sys/class/thermal/thermal_zone9/temp"').read().strip()) // 1000
    except KeyboardInterrupt:
        raise
    except Exception:
        return 0

# ── Camera ────────────────────────────────────────────
def take_photo(path):
    try: os.remove(path)
    except FileNotFoundError: pass
    except OSError: return False
    try:
        result = subprocess.run(['termux-camera-photo', path], stderr=subprocess.DEVNULL)
    except OSError:
        return False
    time.sleep(0.5)
    return result.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1000

# ── LLM (Gemma E2B) ───────────────────────────────────
GEMMA_URL = 'http://127.0.0.1:8080/completion'

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
PARSE_SYS = """You are a home robot command parser.
Parse voice commands into a JSON array of actions.
Output ONLY valid JSON array, no explanation, no markdown.

AVAILABLE ACTIONS:
{"type":"find_person","name":"NAME","message":"MSG or empty"}
  → Find a person and optionally deliver a message
  → Examples: find Chiara, find Abdel, find someone

{"type":"navigate_to","room":"ROOM"}
  → Go to a room: hallway, kitchen, living_room, bedroom, bathroom
  → Examples: go to kitchen, go to bedroom

{"type":"say","message":"TEXT"}
  → Speak a message out loud where robot currently is

{"type":"patrol","rooms":["room1","room2"]}
  → Visit a list of rooms, or all rooms if empty list

{"type":"find_object","object":"OBJECT","room":"ROOM or empty"}
  → Look for an object, optionally in a specific room

{"type":"come_back"}
  → Return to starting position or last known person location

RULES:
- Chain actions when needed: find person + say message = one find_person with message
- "Tell X that Y" = find_person X with message Y
- "Go to kitchen and say hello" = navigate_to + say
- "Find my keys in bedroom" = find_object keys in bedroom
- Always use the minimum number of actions
- Names: capitalize first letter (Chiara, Abdel)
- Rooms: use underscore format (living_room, not living room)
- Output [] if command is unclear

EXAMPLES:
"Find Chiara and tell her dinner is ready"
→ [{"type":"find_person","name":"Chiara","message":"dinner is ready"}]

"Go to the kitchen"
→ [{"type":"navigate_to","room":"kitchen"}]

"Tell Abdel his coffee is cold"
→ [{"type":"find_person","name":"Abdel","message":"your coffee is getting cold"}]

"Patrol the apartment"
→ [{"type":"patrol","rooms":[]}]

"Find my keys in the bedroom"
→ [{"type":"find_object","object":"keys","room":"bedroom"}]

"Go to the living room and say good morning"
→ [{"type":"navigate_to","room":"living_room"},{"type":"say","message":"good morning"}]

"Come back"
→ [{"type":"come_back"}]
"""

def parse_command(text, timeout=40):
    prompt = (
        '<start_of_turn>user\n' + PARSE_SYS +
        '\nCommand: "' + text + '"'
        '<end_of_turn>\n<start_of_turn>model\n[' 
    )
    try:
        resp = requests.post(GEMMA_URL, json={
            'prompt': prompt, 'n_predict': 300,
            'temperature': 0.05, 'stop': ['<end_of_turn>', '\n\n']
        }, timeout=timeout)
        raw = resp.json()['content']
        # Model starts after [ which we injected
        raw = '[' + raw
        s = raw.find('['); e = raw.rfind(']') + 1
        if s >= 0 and e > 0:
            actions = json.loads(raw[s:e])
            # Validate each action has required fields
            valid = []
            for a in actions:
                if isinstance(a, dict) and 'type' in a:
                    valid.append(a)
            return valid
    except Exception as ex:
        print(f'[PARSE] Error: {ex}')
    return []

def warm_up():
    """Once the server answers /health (up to 60 s), send one parse and discard it,
    so the first real command skips the cold prompt (a cold first parse took 39 s)."""
    health = GEMMA_URL.replace('/completion', '/health')
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if requests.get(health, timeout=max(0.1, min(2, deadline - time.time()))).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(max(0, min(1, deadline - time.time())))
    else:
        print('[WARMUP] server not ready after 60 s; skipped')
        return
    t = time.time()
    parse_command('Go to the kitchen', timeout=60)
    print(f'[WARMUP] done in {time.time() - t:.1f} s')

# ── TTS ───────────────────────────────────────────────
def speak(text):
    print(f'[SPEAK] {text}')
    os.system(f'termux-tts-speak "{text}" &')

# ── Main Robot Class ──────────────────────────────────
class Robot:
    def __init__(self, motors=None, camera_source=CAMERA_SOURCE):
        self.motors      = motors
        if camera_source not in ('robotcam', 'termux_photo'):
            raise ValueError(camera_source)
        self.camera_source = camera_source
        self.detector = None
        self.size_policy = SizePolicy(LARGE_FRAME_INTERVAL_S, PERSON_HEIGHT_FRACTION)
        self.camera_session = None
        self.camera_frame = None
        self.cycle       = 0
        self.mission     = None
        self.target      = None   # person name or room
        self.last_moves  = []
        self.scene_log   = []
        self.known_rooms = {}
        self.running     = False
        self.state       = 'idle'  # idle / navigating / searching / found

    def start_camera(self):
        self.detector = get_detector()  # shared by detect_scene and the robot loop
        if self.camera_source == 'robotcam':
            self.camera_session = None
            self.camera_frame = None
            started_boot_s = time.clock_gettime(time.CLOCK_BOOTTIME)
            subprocess.run(['am', 'start', '-n', 'com.pixelrobot.robotcam/.StartActivity',
                            '--es', 'mode', 'B', '--ei', 'rate', '2'],
                           check=True, timeout=10, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                result = read_frame(min_capture_boot_s=started_boot_s)
                if result['status'] == 'ok':
                    self.camera_session = result['session']
                    return
                time.sleep(0.1)
            raise RuntimeError('RobotCam did not publish a usable frame')

    def stop_camera(self):
        if self.camera_source == 'robotcam':
            subprocess.run(['am', 'broadcast', '-n', 'com.pixelrobot.robotcam/.ControlReceiver',
                            '-a', 'com.pixelrobot.robotcam.STOP'],
                           check=True, timeout=5, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        """cm ahead. 999 = no usable reading (treated as clear).
        Firmware sends -1 for no echo, i.e. nothing within ~4m."""
        if not self.motors:
            return 999
        d = self.motors.get_distance()
        if time.time() - self.motors.state.get("dist_at", 0) > 1.0:
            return 999          # stale or never received
        if d < 0:
            return 400          # no echo = clear to sensor max range
        return d

    def navigate_rules(self, results, distance):
        """Fast Python navigation — no LLM. Owns safety; Gemma never overrides.

        Blocked path: strafe first (mecanum holds heading, so YOLO keeps the
        same view), rotate on axis if strafing isn't clearing it, and set
        nav_stuck so run_cycle hands the strategy call to Gemma.
        """
        self.nav_stuck = False
        if not hasattr(self, 'avoid_side'):
            self.avoid_side = 'LEFT'
        if not hasattr(self, 'blocked_n'):
            self.blocked_n = 0
        if not hasattr(self, 'asked_gemma'):
            self.asked_gemma = False
        if not hasattr(self, 'blocked_since'):
            self.blocked_since = None

        ESCAPE_TIMEOUT = 60.0   # s of continuous blockage before giving up

        if distance < OBSTACLE_DIST and self.blocked_since is None:
            self.blocked_since = time.time()

        # Very close: rotate to sweep the sensor, but give up on the same
        # timeout as the main ladder rather than turning forever.
        if distance < 15:
            self.blocked_n += 1
            if time.time() - self.blocked_since >= ESCAPE_TIMEOUT:
                print(f'  [NAV] cornered for {ESCAPE_TIMEOUT:.0f}s — giving up')
                return 'STOP'
            return self.avoid_side

        # ── Safety ──────────────────────────────────────────
        if distance < OBSTACLE_DIST:
            self.blocked_n += 1
            n = self.blocked_n

            if n <= 2:                      # strafe — holds heading for YOLO
                return 'STRAFE_' + self.avoid_side
            if n <= 6:                      # sweep this side, sensor leads
                return self.avoid_side
            if n == 7:                      # committed flip, ask Gemma once
                self.avoid_side = 'RIGHT' if self.avoid_side == 'LEFT' else 'LEFT'
                if not self.asked_gemma:
                    self.nav_stuck = True
                    self.asked_gemma = True
                return self.avoid_side
            if n <= 13:                     # sweep back through and past centre
                return self.avoid_side

            # A full sweep found nothing. Keep trying until the timeout, then
            # declare defeat rather than spinning indefinitely.
            if time.time() - self.blocked_since < ESCAPE_TIMEOUT:
                self.blocked_n = 0          # restart the ladder
                return 'STRAFE_' + self.avoid_side
            print(f'  [NAV] boxed in for {ESCAPE_TIMEOUT:.0f}s — giving up')
            return 'STOP'

        # path is clear — reset the avoidance state machine
        self.blocked_n = 0
        self.asked_gemma = False
        self.blocked_since = None

        # ── Person ──────────────────────────────────────────
        labels = [r[0] for r in results]
        if 'person' in labels:
            for r in results:
                if r[0] == 'person':
                    dist_est = estimate_distance_single('person', r[7])
                    if dist_est and dist_est < PERSON_STOP_DIST:
                        return 'STOP'
            direction = person_direction(results)
            if direction:
                return direction

        # ── Room signature ──────────────────────────────────
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

    def run_cycle(self, photo_path=None):
        self.cycle += 1
        temp = get_temp()
        distance = self.get_distance()
        print(f'\n── Cycle {self.cycle} | {temp}°C | dist:{distance}cm ──')

        # Vision: every cycle must have a fresh usable frame before any movement.
        if photo_path:
            path = photo_path
        elif self.camera_source == 'robotcam':
            frame_result = read_frame(session=self.camera_session)
            if frame_result['status'] != 'ok' or (self.camera_frame is not None and
                    frame_result['frame'] <= self.camera_frame):
                reason = frame_result['status'] if frame_result['status'] != 'ok' else 'no new'
                print(f'  [CAM] {reason} frame — stop')
                if self.motors: self.motors.stop()
                return 'STOP'
            self.camera_session = frame_result['session']
            self.camera_frame = frame_result['frame']
            frame = frame_result['image']
        else:
            if not take_photo(PHOTO_A):
                print('  [CAM] Photo failed')
                if self.motors: self.motors.stop()
                return 'STOP'
            path = PHOTO_A

        try:
            if photo_path:
                results = detect_scene(path)
            else:
                if self.detector is None: self.detector = get_detector()
                size = self.size_policy.next_size()
                if self.camera_source == 'termux_photo':
                    from PIL import Image, ImageOps
                    with Image.open(path) as source:
                        frame = ImageOps.exif_transpose(source).convert('RGB')
                detections = self.detector.detect(frame, size)
                self.size_policy.observe(detections, frame.height)
                results = scene_from_detections(detections, *frame.size)
                print(f'  [YOLO] size={size} person={any(d["class_name"] == "person" for d in detections)}')
        except Exception as exc:
            print(f'  [VISION] {exc} — stop')
            if self.motors: self.motors.stop()
            return 'STOP'
        scene = scene_to_text(results)
        self.scene_log.append(scene)
        if len(self.scene_log) > 10:
            self.scene_log.pop(0)
        print(f'  [YOLO] {scene}')

        # Fast navigation rules
        move = self.navigate_rules(results, distance)
        print(f'  [NAV] {move}')

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
    warm_up()

    # Test with scene photo
    if os.path.exists('test_photos/scene_test.jpg'):
        result = robot.run_cycle('test_photos/scene_test.jpg')
        print(f'Test cycle result: {result}')
    else:
        print('No test photo found')

    print(f'\nKnown rooms: {robot.known_rooms}')
    print(f'Last moves: {robot.last_moves}')
    print('\nRobot ready. Connect motors and run mission.')
