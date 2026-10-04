"""Native verification and capture; no timed block/session is ever invoked."""
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[2]
assert sys.platform == 'android'
checks = [
    ('self_check', [str(HERE / 'self_check.py')]),
    ('power_map_units', [str(ROBOT / 'benchmark/power_map/test_power_map.py')]),
    ('coresidency_units', [str(ROBOT / 'benchmark/coresidency/test_coresidency.py')]),
    ('detector_units', [str(ROBOT / 'test_detector_robotcam.py')]),
    ('mission_units', [str(ROBOT / 'test_run_mission.py')]),
    ('cycle_units', [str(ROBOT / 'run_cycle_safety_test.py')]),
]
results = {}
if '--self-only' in sys.argv:
    checks = checks[:1]
    results = json.loads((HERE / 'check_results.json').read_text())
for name, args in checks:
    with (HERE / f'{name}.stdout').open('w') as out, (HERE / f'{name}.stderr').open('w') as err:
        rc = subprocess.run([sys.executable, *args], cwd=ROBOT, stdout=out, stderr=err).returncode
    results[name] = {'command': [sys.executable, *args], 'exit_code': rc}
    print(name, rc, flush=True)
(HERE / 'check_results.json').write_text(json.dumps(results, indent=2) + '\n')
if any(r['exit_code'] for r in results.values()):
    raise SystemExit('checks failed')
