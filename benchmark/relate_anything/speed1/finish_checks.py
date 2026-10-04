"""Capture final quality and the single required session dry-run; never timed."""
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
assert sys.platform == 'android'
for stem in ('quality.json', 'quality.stdout', 'quality.stderr', 'self_check.stdout', 'self_check.stderr', 'check_results.json'):
    p = HERE / stem
    if p.exists():
        p.rename(p.with_name(p.stem + '_initial' + p.suffix))
# Preserve passing direct-script results; rerun only the changed self-check.
(HERE / 'check_results.json').write_bytes((HERE / 'check_results_initial.json').read_bytes())
with (HERE / 'quality.stdout').open('w') as out, (HERE / 'quality.stderr').open('w') as err:
    rc = subprocess.run([sys.executable, '-u', str(HERE / 'quality.py')], stdout=out, stderr=err).returncode
assert rc == 0, 'quality runner failed'
rc = subprocess.run([sys.executable, '-u', str(HERE / 'run_checks.py'), '--self-only']).returncode
assert rc == 0, 'self-check failed'
with (HERE / 'session_dry.stdout').open('w') as out, (HERE / 'session_dry.stderr').open('w') as err:
    command = [sys.executable, '-u', str(HERE / 'session.py'), '--session', '--dry-run',
               '--output', str(HERE / 'session_dry.json')]
    rc = subprocess.run(command, stdout=out, stderr=err).returncode
(HERE / 'dry_check.json').write_text(json.dumps({'command': command, 'exit_code': rc}, indent=2) + '\n')
print('session dry-run:', rc, 'INFORMAL — agents resident, NOT VALID TIMING', flush=True)
assert rc == 0, 'session dry-run failed'
