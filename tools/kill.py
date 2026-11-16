import subprocess, time
def su(cmd, t=15):
    try:
        return subprocess.check_output(['su','-c',cmd], timeout=t,
               stderr=subprocess.DEVNULL).decode(errors='replace').strip()
    except Exception:
        return None
def real_servers():
    # exact process name only — cannot match our own shell commands
    return [p for p in (su('pgrep -x llama-server') or '').split() if p.isdigit()]
before = real_servers()
if not before:
    print('nothing running.')
else:
    print('killing:', ' '.join(before))
    su('pkill -9 -x llama-server')
    time.sleep(2)
    left = real_servers()
    print('remaining:', ' '.join(left) if left else 'none (all killed)')
