"""Autonomous entry point. Needs root (USB) and the LLM server on 8080.

  su -c "LD_LIBRARY_PATH=... python run_mission.py 'find the kitchen'"
  add --dry to run the nav loop with no motor commands.
"""
import sys, time
sys.path.insert(0, "/data/data/com.termux/files/home/robot")

import main as R
from motors import Motors

dry     = "--dry" in sys.argv
args    = [a for a in sys.argv[1:] if not a.startswith("--")]
mission = args[0] if args else "explore and map the rooms"

motors = None
if not dry:
    motors = Motors()
    motors.connect()
    time.sleep(1)
    print("PING:", "ALIVE" if motors.ping() else "NO REPLY")
    print("distance:", motors.get_distance(), "cm")

robot = R.Robot(motors)
print(f"\nmission: {mission}{'   [DRY — no motor commands]' if dry else ''}")
print("ctrl-C to stop\n")

try:
    robot.run_mission(mission)
except KeyboardInterrupt:
    print("\ninterrupted")
finally:
    if motors:
        motors.stop()
        motors.disconnect()
    print(f"rooms: {robot.known_rooms}")
    print(f"moves: {robot.last_moves}")
