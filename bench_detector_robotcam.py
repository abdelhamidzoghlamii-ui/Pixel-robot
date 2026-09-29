"""MOTORS OFF functional check. Run from native Termux: python bench_detector_robotcam.py"""
import subprocess
import time

from detect_person import get_detector
from detector_size_policy import SizePolicy
from robotcam_reader import read_frame


START = ['am', 'start', '-n', 'com.pixelrobot.robotcam/.StartActivity',
         '--es', 'mode', 'B', '--ei', 'rate', '2']
STOP = ['am', 'broadcast', '-n', 'com.pixelrobot.robotcam/.ControlReceiver',
        '-a', 'com.pixelrobot.robotcam.STOP']


def main():
    detector = get_detector()
    policy = SizePolicy()
    session = None
    stopped = False
    stop_observed = False
    try:
        started_boot_s = time.clock_gettime(time.CLOCK_BOOTTIME)
        subprocess.run(START, check=True, timeout=10, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start = time.monotonic()
        while time.monotonic() - start < 60:
            if not stopped and time.monotonic() - start >= 30:
                subprocess.run(STOP, check=True, timeout=5, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                stopped = True
                print('RobotCam stopped mid-run', flush=True)
            r = read_frame(session=session,
                           min_capture_boot_s=started_boot_s if session is None else None)
            if r['status'] != 'ok':
                print(f"robot_side=STOP camera_status={r['status']}", flush=True)
                if stopped:
                    stop_observed = True
            else:
                session = r['session']
                size = policy.next_size()
                t0 = time.perf_counter()
                detections = detector.detect(r['image'], size)
                ms = (time.perf_counter() - t0)*1000
                policy.observe(detections, r['image'].height)
                person = any(d['class_name'] == 'person' for d in detections)
                print(f'size_used={size} detect_ms={ms:.1f} person={person}', flush=True)
            time.sleep(0.5)
        print(f'stop_observed_after_robotcam_stop={stop_observed}', flush=True)
        if not stop_observed:
            raise RuntimeError('Robot side did not observe STOP after RobotCam stopped')
    finally:
        subprocess.run(STOP, check=True, timeout=5, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
