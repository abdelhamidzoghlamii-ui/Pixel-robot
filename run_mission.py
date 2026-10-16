"""Autonomous entry point. --dry uses RobotCam without opening motor USB.
--no-picker: today's rules (incl. person approach) and person-stop; neither the Gemma client nor M2 is loaded."""
import signal
import sys
import time

sys.path.insert(0, '/data/data/com.termux/files/home/robot')
import main as R
import heat_control


def exit_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def build_picker():
    """(Picker, None), or (None, reason) when it cannot run: the mission then continues in EXPLORE without picks
    (a fallback, never a stop). The owner starts the server with setup_q4 first."""
    import picker
    import relate
    try:
        gemma = picker.Gemma()
    except Exception as exc:
        return None, f'llama-server at {picker.URL} not ready ({type(exc).__name__}: {exc}); run setup_q4 first'
    try:
        return picker.Picker(gemma, relate.Relate()), None
    except Exception as exc:
        return None, f'M2 not available ({type(exc).__name__}: {exc})'


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dry = '--dry' in argv
    camera_source = 'termux_photo' if '--termux-photo' in argv else 'robotcam'
    person_search = '--person-search' in argv
    use_picker = '--no-picker' not in argv
    args = [a for a in argv if not a.startswith('--')]
    mission = args[0] if args else 'explore and map the rooms'

    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGHUP, exit_on_signal)
    motors = None
    robot = None
    heat = None
    pick = None
    try:
        # Before the motor reader and heat threads exist, so the ORT thread diff sees only ORT workers;
        # raises (mission not started) when the workers or their affinity cannot be verified.
        R.get_detector(pinned=True)
        # Also before those threads: Relate's ORT thread diff must see only its own worker.
        pick, picker_off = build_picker() if use_picker else (None, '--no-picker')
        if use_picker and pick is None:
            print(f'[PICKER] disabled: {picker_off} — the mission continues in EXPLORE')
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
        if use_picker:
            robot.picker = pick
            robot.set_mode('EXPLORE', 'mission start')
        heat = heat_control.HeatControl(heat_control.HeatMonitor())
        heat.start()  # raises without root or a first complete reading: mission not started
        heat.event('picker_config', {'enabled': True, **pick.config()} if pick else {'enabled': False, 'reason': picker_off})
        robot.start_camera(heat)
        if not dry:
            R.warm_up()
        print(f"\nmission: {mission}{'   [DRY — no motor commands]' if dry else ''}")
        print('ctrl-C to stop\n')
        robot.run_mission(mission, heat=heat)
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
        if pick:
            if not pick.stop():
                cleanup_errors.append('[PICKER] worker still running after its timeout')
            if robot:
                try:
                    robot.collect_pick()  # trace a pick that ended during shutdown
                except Exception as exc:
                    cleanup_errors.append(f'[PICKER] last pick not traced: {exc}')
        if heat:
            try:
                heat.stop()
            except Exception as exc:
                cleanup_errors.append(f'[HEAT] stop failed: {exc}')
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
