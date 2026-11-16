# prep.py — free RAM before a robot run.
# HARDCODED list only. No discovery. No system services. Apps reopen when tapped.
import subprocess, time

KILL = [
    'com.google.android.apps.chromecast.app',      # Google Home
    'com.google.android.apps.tips',                # Pixel Tips
    'com.google.android.gm',                       # Gmail
    'com.google.android.apps.messaging',           # Messages
    'com.google.android.apps.maps',                # Maps
    'com.google.android.apps.docs.editors.docs',   # Docs
    'com.google.android.apps.docs',                # Drive
    'com.lunchboxclassx.aiusagemeter',             # AI usage meter
    'com.anthropic.claude',                        # Claude app (kill LAST)
]

# Belt and braces: refuse to touch anything resembling a system process,
# even if the list above were ever edited by mistake.
FORBIDDEN = ('system_server', 'systemui', 'zygote', 'surfaceflinger',
             'inputmethod', 'nexuslauncher', 'com.android.phone',
             'com.termux', 'android.gms', 'android.as', 'vending',
             'googlequicksearchbox', 'GoogleCamera')

def su(c, t=15):
    try:
        return subprocess.check_output(['su','-c',c], timeout=t,
               stderr=subprocess.DEVNULL).decode(errors='replace').strip()
    except Exception:
        return ""

def avail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1048576
    return 0

before = avail()
print("MemAvailable before: %.2f GB" % before)

freed = 0
for pkg in KILL:
    if any(f in pkg for f in FORBIDDEN):
        print("  SKIP (forbidden) %s" % pkg); continue
    pid = su('pidof %s' % pkg).split()
    if not pid:
        continue
    p = pid[0]
    rss = su("awk '/^VmRSS/{print $2}' /proc/%s/status" % p)
    su('kill %s' % p)
    mb = int(rss)/1024 if rss.isdigit() else 0
    freed += mb
    print("  killed %-45s %5.0f MB" % (pkg, mb))

su('sync')
su('echo 3 > /proc/sys/vm/drop_caches')
time.sleep(2)
after = avail()
print("\nreclaimed from apps: %.0f MB" % freed)
print("MemAvailable after:  %.2f GB   (delta %+.2f GB)" % (after, after - before))
