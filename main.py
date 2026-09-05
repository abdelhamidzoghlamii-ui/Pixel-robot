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
    except KeyboardInterrupt:
        raise
    except Exception:
        return 0

# ── Camera ────────────────────────────────────────────
def take_photo(path):
    os.system(f'termux-camera-photo {path} 2>/dev/null')
    time.sleep(0.5)
    return os.path.exists(path) and os.path.getsize(path) > 1000

# ── LLM (Gemma E2B) ───────────────────────────────────
GEMMA_URL = 'http://127.0.0.1:8080/completion'
GEMMA_SYS = """You are a robot navigation AI controlling a 4WD mecanum wheel robot.
Reply with ONE word: FORWARD, LEFT, RIGHT, BACK, or STOP.
Then one sentence explanation.

STRICT RULES — follow exactly:
- Person center + distance < 100cm → STOP (reached person)
- Person center + distance > 100cm → FORWARD (approach)
- Person visible LEFT → LEFT (turn toward them)
- Person visible RIGHT → RIGHT (turn toward them)
- Obstacle center + distance < 80cm → BACK (reverse away)
- Obstacle LEFT → RIGHT (avoid by going right)
- Obstacle RIGHT → LEFT (avoid by going left)
- Refrigerator/sink visible + mission kitchen → FORWARD
- Bed/wardrobe visible + mission bedroom → FORWARD
- Toilet visible + mission bathroom → FORWARD
- Couch/tv visible + mission living_room → FORWARD
- Room signature visible + distance < 100cm → STOP (arrived)
- Same direction 4+ times + empty scene → turn LEFT or RIGHT
- All 5 rooms visited + person not found → STOP (give up)
- Patrol + all 5 rooms visited → STOP (complete)"""

def gemma_decide(context, image_path=None):
    """
    Call Gemma with text context and optional image.
    image_path: if provided Gemma sees the actual photo
    """
    import base64

    # Build text prompt
    prompt = (
        '<start_of_turn>user\n' + GEMMA_SYS + '\n\n' + context +
        '<end_of_turn>\n<start_of_turn>model\n'
    )

    payload = {
        'prompt': prompt,
        'n_predict': 40,
        'temperature': 0.1,
        'stop': ['<end_of_turn>']
    }

    # Add image if provided and file exists
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            payload['image_data'] = [{'data': img_b64, 'id': 1}]
            payload['prompt'] = payload['prompt'].replace(
                '<start_of_turn>user\n',
                '<start_of_turn>user\n[img-1]\n'
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            pass  # fall back to text only if image fails

    try:
        resp = requests.post(GEMMA_URL, json=payload, timeout=45)
        return resp.json()['content'].strip()
    except KeyboardInterrupt:
        raise
    except Exception:
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
    except KeyboardInterrupt:
        raise
    except Exception:
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

def parse_command(text):
    prompt = (
        '<start_of_turn>user\n' + PARSE_SYS +
        '\nCommand: "' + text + '"'
        '<end_of_turn>\n<start_of_turn>model\n[' 
    )
    try:
        resp = requests.post(GEMMA_URL, json={
            'prompt': prompt, 'n_predict': 300,
            'temperature': 0.05, 'stop': ['<end_of_turn>', '\n\n']
        }, timeout=20)
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

        # Gemma check — every N cycles or triggered
        person_found  = any(r[0] == 'person' for r in results)
        every_5       = (self.cycle % 5 == 0)
        every_3       = (self.cycle % GEMMA_INTERVAL == 0)
        goal_reached  = (move == 'STOP')
        new_room      = len(self.known_rooms) > getattr(self, '_prev_rooms', 0)
        self._prev_rooms = len(self.known_rooms)

        # DECISIONS #19: Python owns safety. A safety move is never handed to
        # Gemma, so the model cannot override an obstacle stop.
        safety_move = move in ('BACK', 'LEFT', 'RIGHT',
                               'STRAFE_LEFT', 'STRAFE_RIGHT')
        stuck       = getattr(self, 'nav_stuck', False)

        use_gemma  = (every_3 or every_5 or person_found or goal_reached
                      or new_room or stuck) and not (safety_move and not stuck)
        use_vision = every_5 or person_found or goal_reached or new_room

        if use_gemma and self.mission:
            trigger = []
            if every_3:      trigger.append('interval')
            if every_5:      trigger.append('vision')
            if person_found: trigger.append('person')
            if goal_reached: trigger.append('goal')
            if new_room:     trigger.append('new_room')
            if stuck:        trigger.append('stuck')
            print(f'  [GEMMA] Consulting ({"+".join(trigger)})...')
            context = self.gemma_context(scene)
            response = gemma_decide(context, image_path=path if use_vision else None)
            print(f'  [GEMMA] {response}')
            words = response.upper().split()
            for w in words:
                if w in ['FORWARD','LEFT','RIGHT','BACK','STOP','SPEAK',
                         'STRAFE_LEFT','STRAFE_RIGHT']:
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
