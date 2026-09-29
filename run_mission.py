"""Autonomous entry point. --dry uses RobotCam without opening motor USB."""
import signal
import sys
import time

sys.path.insert(0, '/data/data/com.termux/files/home/robot')
import main as R


def exit_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dry = '--dry' in argv
    camera_source = 'termux_photo' if '--termux-photo' in argv else 'robotcam'
    person_search = '--person-search' in argv
    args = [a for a in argv if not a.startswith('--')]
    mission = args[0] if args else 'explore and map the rooms'

    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGHUP, exit_on_signal)
    motors = None
    robot = None
    try:
        if not dry:
            from motors import Motors
            motors = Motors()
            motors.connect()
            time.sleep(1)
            print('PING:', 'ALIVE' if motors.ping() else 'NO REPLY')
            print('distance:', motors.get_distance(), 'cm')
        robot = R.Robot(motors, camera_source=camera_source)
        if person_search:
            robot.size_policy.set_mode('person_search')
        robot.start_camera()
        if not dry:
            R.warm_up()
        print(f"\nmission: {mission}{'   [DRY — no motor commands]' if dry else ''}")
        print('ctrl-C to stop\n')
        robot.run_mission(mission)
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        cleanup_errors = []
        if motors:
            try:
                motors.stop()
            except Exception as exc:
                cleanup_errors.append(f'[MOTOR] stop failed: {exc}')
        if robot:
            try:
                robot.stop_camera()
            except Exception as exc:
                cleanup_errors.append(f'[CAM] stop failed: {exc}')
        if motors:
            try:
                motors.disconnect()
            except Exception as exc:
                cleanup_errors.append(f'[MOTOR] disconnect failed: {exc}')
        for error in cleanup_errors:
            print(error)
        if robot:
            print(f'rooms: {robot.known_rooms}')
            print(f'moves: {robot.last_moves}')


if __name__ == '__main__':
    main()
