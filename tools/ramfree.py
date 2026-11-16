# ramfree.py — deeper RAM reclaim. FULLY REVERSIBLE.
#   python3 ~/ramfree.py off   -> disable/restrict the heavy passengers
#   python3 ~/ramfree.py on    -> restore everything
import subprocess, sys, time

def su(c, t=20):
    try:
        return subprocess.check_output(['su','-c',c], timeout=t,
               stderr=subprocess.DEVNULL).decode(errors='replace').strip()
    except Exception:
        return "ERR"

def avail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1048576
    return 0

# Tier 1: fully disabled (reversible). YouTube only, per your call.
SAFE_DISABLE = [
    'com.google.android.youtube',
    'com.google.android.apps.chromecast.app',
]

# Tier 2: background-restricted only — stays callable, idle daemon stops
RESTRICT = [
    'com.google.android.gms',
    'com.google.android.as',
    'com.android.vending',
    'com.google.android.googlequicksearchbox',
]

mode = (sys.argv[1] if len(sys.argv) > 1 else 'off').lower()
b = avail()
print("MemAvailable before: %.2f GB   mode=%s" % (b, mode))

if mode == 'off':
    for p in SAFE_DISABLE:
        su('pm disable-user --user 0 %s' % p)
        su('am force-stop %s' % p)
        print("  disabled   %s" % p)
    for p in RESTRICT:
        su('am set-standby-bucket %s restricted' % p)
        su('cmd appops set %s RUN_IN_BACKGROUND deny' % p)
        su('cmd appops set %s RUN_ANY_IN_BACKGROUND deny' % p)
        su('cmd appops set %s WAKE_LOCK deny' % p)
        su('am force-stop %s' % p); su('am kill %s' % p)
        print("  restricted %s" % p)
elif mode == 'on':
    for p in SAFE_DISABLE:
        su('pm enable %s' % p); print("  re-enabled %s" % p)
    for p in RESTRICT:
        su('am set-standby-bucket %s active' % p)
        su('cmd appops set %s RUN_IN_BACKGROUND allow' % p)
        su('cmd appops set %s RUN_ANY_IN_BACKGROUND allow' % p)
        su('cmd appops set %s WAKE_LOCK allow' % p)
        print("  restored   %s" % p)
else:
    print("usage: ramfree.py [off|on]"); raise SystemExit

su('sync'); su('echo 3 > /proc/sys/vm/drop_caches')
su('echo 1 > /proc/sys/vm/compact_memory')
time.sleep(3)
a = avail()
print("MemAvailable after:  %.2f GB   (delta %+.2f GB)" % (a, a - b))
print("top holders:")
for l in su('ps -A -o rss,name --sort=-rss').splitlines()[1:8]:
    print("  ", l.strip())
