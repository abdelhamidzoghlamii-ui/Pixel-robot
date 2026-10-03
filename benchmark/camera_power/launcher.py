"""Reuse DUTY1's reviewed launcher, with camera-power paths and refusal pattern."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def launcher_text():
    text = (HERE.parent / 'duty_cycle' / 'oneshot.sh').read_text()
    # Keep duty_cycle in the competing-runner refusal while adding this runner.
    text = text.replace('duty_cycle\\.py|power_map', 'camera_power\\.py|duty_cycle\\.py|power_map')
    for before, after in (
            ('L=$H/duty_cycle', 'L=$H/camera_power'),
            ('R=$H/robot/benchmark/duty_cycle', 'R=$H/robot/benchmark/camera_power'),
            ('duty_cycle_screen.txt', 'camera_power_screen.txt'),
            ('run_duty_cycle.sh', 'run_camera_power.sh')):
        assert before in text, 'DUTY1 launcher changed: '+before
        text = text.replace(before, after)
    return text


if __name__ == '__main__':
    if Path('/termux-home').is_dir() or os.environ.get('PREFIX') != '/data/data/com.termux/files/usr':
        raise SystemExit('Run from native Termux, not Debian/proot.')
    out = Path('/data/data/com.termux/files/home/camera_power')
    out.mkdir(exist_ok=True)
    script = out / 'launcher.sh'
    script.write_text(launcher_text())
    os.execvp('bash', ['bash', str(script), *sys.argv[1:]])
