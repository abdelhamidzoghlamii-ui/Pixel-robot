import sys, os, time
sys.path.insert(0, '/data/data/com.termux/files/home/robot')
from main import Robot
from detect_person import detect_scene, scene_to_text, person_direction
from stereo_depth import estimate_distance_single

def get_temp():
    try:
        return int(os.popen('su -c "cat /sys/class/thermal/thermal_zone9/temp"').read().strip()) // 1000
    except:
        return 0

class SimulatedMotors:
    def __init__(self):
        self.last_cmd = None

    def get_distance(self):
        dist = input('  [ARDUINO] Distance (cm) or ENTER for clear: ').strip()
        if not dist:
            return 999
        return int(dist)

    def forward(self, speed, duration):
        print(f'  [ARDUINO] >> FORWARD speed={speed}')
        input('  [ARDUINO] Press ENTER when movement done...')

    def rotate_left(self, speed, duration):
        print(f'  [ARDUINO] >> ROTATE LEFT speed={speed}')
        input('  [ARDUINO] Press ENTER when movement done...')

    def rotate_right(self, speed, duration):
        print(f'  [ARDUINO] >> ROTATE RIGHT speed={speed}')
        input('  [ARDUINO] Press ENTER when movement done...')

    def backward(self, speed, duration):
        print(f'  [ARDUINO] >> BACKWARD speed={speed}')
        input('  [ARDUINO] Press ENTER when movement done...')

    def strafe_right(self, speed, duration):
        print(f'  [ARDUINO] >> STRAFE RIGHT speed={speed}')
        input('  [ARDUINO] Press ENTER when movement done...')

    def strafe_left(self, speed, duration):
        print(f'  [ARDUINO] >> STRAFE LEFT speed={speed}')
        input('  [ARDUINO] Press ENTER when movement done...')

    def stop(self):
        print(f'  [ARDUINO] >> STOP')

def take_photo_interactive(path):
    input(f'\n  [CAM] Press ENTER to take photo...')
    os.system(f'termux-camera-photo {path} 2>/dev/null')
    time.sleep(0.8)
    return os.path.exists(path) and os.path.getsize(path) > 1000

PHOTO = '/data/data/com.termux/files/home/robot_photo.jpg'

print('='*55)
print('  ROBOT MISSION SIMULATION')
print('='*55)
print('  Mission: Find Chiara in the toilet and tell her')
print('  "Abdel told you to come to the sleeping room ;)"')
print('='*55)
print()
print('  You are playing the role of:')
print('  - Arduino (type distance or ENTER for clear path)')
print('  - Camera operator (press ENTER to take each photo)')
print()
input('  Press ENTER to start mission...')

motors = SimulatedMotors()
robot = Robot(motors=motors)
robot.mission = 'find Chiara in the toilet'
robot.target = 'Chiara'

cycle = 0
found = False
person_seen_count = 0

while not found and cycle < 30:
    cycle += 1
    temp = get_temp()
    print(f'\n{"="*55}')
    print(f'  CYCLE {cycle} | Temp: {temp}°C')
    print(f'{"="*55}')

    # Thermal check
    if temp > 80:
        print('  [THERMAL] Too hot! Pausing 10s...')
        time.sleep(10)
        continue

    # Take photo
    ok = take_photo_interactive(PHOTO)
    if not ok:
        print('  [CAM] Photo failed, skipping...')
        continue

    # YOLO detection
    print('  [YOLO] Analyzing...')
    results = detect_scene(PHOTO)
    scene = scene_to_text(results)
    robot.scene_log.append(scene)
    print(f'  [YOLO] {scene}')

    # Distance from Arduino
    distance = motors.get_distance()
    print(f'  [DIST] {distance}cm')

    # Navigate
    move = robot.navigate_rules(results, distance)

    # Check if person found and close
    person_found = any(r[0] == 'person' for r in results)
    if person_found:
        for r in results:
            if r[0] == 'person':
                dist_est = estimate_distance_single('person', r[7])
                pos = r[2]
                print(f'  [PERSON] Detected! Position:{pos} EstDist:{dist_est}cm')
                person_seen_count += 1
                print(f'  [PERSON] Seen {person_seen_count} times')
                if dist_est and dist_est < 200 or person_seen_count >= 3:
                    print('  [MISSION] Person close enough — delivering message!')
                    print()
                    print('  *** Abdel told you to come to the sleeping room ;) ***')
                    print()
                    os.system('termux-tts-speak "Chiara, Abdel told you to come to the sleeping room" &')
                    found = True
                    move = 'STOP'

    # Gemma triggers: every 3 cycles, person detected, obstacle, new room
    gemma_trigger = (
        cycle % 3 == 0 or
        person_found or
        distance < 50 or
        len(robot.known_rooms) > len(getattr(robot, '_last_rooms', {}))
    )
    robot._last_rooms = dict(robot.known_rooms)

    if gemma_trigger and robot.mission:
        print('  [GEMMA] Consulting...')
        context = (
            f"Mission: {robot.mission}\n"
            f"Current scene: {scene}\n"
            f"Distance ahead: {distance}cm\n"
            f"Person visible: {person_found}\n"
            f"Last 3 moves: {robot.last_moves[-3:]}\n"
            f"Known rooms: {robot.known_rooms}"
        )
        try:
            import requests
            prompt = (
                '<start_of_turn>user\n'
                'You are a robot. ONE word: FORWARD LEFT RIGHT BACK STOP SPEAK.\n'
                + context +
                '<end_of_turn>\n<start_of_turn>model\n'
            )
            resp = requests.post('http://127.0.0.1:8080/completion', json={
                'prompt': prompt, 'n_predict': 20,
                'temperature': 0.1, 'stop': ['<end_of_turn>']
            }, timeout=10)
            gemma_says = resp.json()['content'].strip().split()[0].upper()
            print(f'  [GEMMA] suggests: {gemma_says}')
            if gemma_says in ['LEFT','RIGHT','BACK','STOP','SPEAK']:
                move = gemma_says
        except:
            print('  [GEMMA] offline - using rules')

    print(f'  [NAV] Decision: {move}')
    robot.last_moves.append(move)

    if move != 'STOP':
        if move == 'FORWARD':   motors.forward(130, 1.5)
        elif move == 'LEFT':    motors.rotate_left(120, 1.0)
        elif move == 'RIGHT':   motors.rotate_right(120, 1.0)
        elif move == 'BACK':    motors.backward(130, 1.0)

print()
print('='*55)
print(f'  MISSION COMPLETE in {cycle} cycles')
print(f'  Rooms discovered: {robot.known_rooms}')
print(f'  Moves made: {robot.last_moves}')
print('='*55)
