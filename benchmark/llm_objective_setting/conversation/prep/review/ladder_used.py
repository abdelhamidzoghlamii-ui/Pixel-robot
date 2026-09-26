# excerpt of /termux-home/ladder/ladder.py (reviewed and approved earlier; unchanged): the pieces conv_speed.py uses
CORES = "4-7"

class CoresLost(Exception):
    """Android moved Termux off the top-app cpuset mid-block; the block's data is discarded and redone."""

def check_cores(when):
    if not set(range(4, 8)) <= allowed_cpus():
        raise CoresLost(f"cores {CORES} not all allowed {when} (allowed {sorted(allowed_cpus())})")

def wait_cores(label):
    """Pauses until cores 4-7 are allowed again (Termux back in the foreground), rechecking every 10 s."""
    began = time.monotonic()
    while not set(range(4, 8)) <= allowed_cpus():
        print(f"  [{label}] paused: cores {CORES} not allowed (allowed {sorted(allowed_cpus())}); bring Termux to the "
              f"foreground. {time.monotonic() - began:.0f} s", flush=True)
        time.sleep(10)
DROP_CMD = "su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"

def cached_mib():
    return next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("Cached:"))

def evict(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # read-only files: every page is clean
        finally:
            os.close(fd)
DROP_REQUEST, DROP_DONE = HERE / ".drop_request", HERE / ".drop_done"
HANDSHAKE_S = 60

def drop_handshake(name):
    """Unattended cache drop: create .drop_request; a native-Termux watcher drops the page cache as root and
    writes 'ok' or 'failed' to .drop_done. Returns that answer, or None after HANDSHAKE_S seconds."""
    DROP_DONE.unlink(missing_ok=True)
    DROP_REQUEST.touch()
    print(f"\n[{name}] cold load next: handshake, waiting up to {HANDSHAKE_S} s for {DROP_DONE}", flush=True)
    deadline, answer = time.monotonic() + HANDSHAKE_S, None
    try:
        while time.monotonic() < deadline:
            if DROP_DONE.exists() and (text := DROP_DONE.read_text().strip()):  # empty: still being written
                answer = text
                break
            time.sleep(0.5)
    finally:
        DROP_REQUEST.unlink(missing_ok=True)
        DROP_DONE.unlink(missing_ok=True)
    return answer

def make_cold(name, files):
    """Drop the page cache from native Termux (Magisk su does not work in proot) and check /proc/meminfo that
    it happened; otherwise evict only this model's weight files. LADDER_COLD_HANDSHAKE=1 asks through files
    (unattended runs); otherwise the human is prompted on the terminal."""
    before = cached_mib()
    if os.environ.get("LADDER_COLD_HANDSHAKE") == "1":
        answer = drop_handshake(name)
        after = cached_mib()
        if answer == "ok" and after < 0.5 * before:
            return {"mode": "full-cold (page cache dropped by the handshake watcher with drop_caches)",
                    "cached_mib_before": before, "cached_mib_after": after}
        why = (f"handshake timed out after {HANDSHAKE_S} s" if answer is None else
               f"handshake answered {answer!r}" if answer != "ok" else
               f"handshake answered 'ok' but page cache did not drop ({before} -> {after} MiB)")
    elif not sys.stdin.isatty():
        why = "no terminal to prompt on"
    else:
        print(f"\n[{name}] COLD LOAD NEXT. In native Termux (not proot) run:\n    {DROP_CMD}\n"
              "then press Enter here. Type s + Enter to skip (weights-cold fallback).", flush=True)
        answer = input("> ").strip().lower()
        after = cached_mib()
        if answer != "s" and after < 0.5 * before:
            return {"mode": "full-cold (page cache dropped by the human with drop_caches)",
                    "cached_mib_before": before, "cached_mib_after": after}
        why = "skipped by the human" if answer == "s" else f"Enter pressed but page cache did not drop ({before} -> {after} MiB)"
    evict(files)
    return {"mode": "weights-cold (posix_fadvise DONTNEED on the weight files; libraries stay cached)",
            "fallback_reason": why, "cached_mib_before": before}
THERMAL_MAX_AGE_S = 30
GATE_MC = 4000

def read_thermal(path):
    """Last line of the root thermal log: '2026-09-25T21:51:27Z z9=36000 z10=36000 z11=37000' (millidegrees)."""
    last = Path(path).read_text().strip().splitlines()[-1].split()
    stamp = datetime.strptime(last[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if age > THERMAL_MAX_AGE_S:
        raise SystemExit(f"thermal log {path} last line is {age:.0f} s old: the logger is not running")
    z = dict(kv.split("=") for kv in last[1:])
    return {"at": last[0], **{k: int(z[k]) for k in ("z9", "z10", "z11")}}

def thermal_gate(path, idle, label):
    """Waits until z9 <= idle + GATE_MC (cooler than idle is fine); returns the reading the block starts at."""
    began = time.monotonic()
    while (t := read_thermal(path))["z9"] > idle["z9"] + GATE_MC:
        print(f"  [{label}] waiting: z9 {t['z9'] / 1000:.1f} degC, need <= {(idle['z9'] + GATE_MC) / 1000:.1f} "
              f"(idle {idle['z9'] / 1000:.1f} + {GATE_MC / 1000:.0f}), {time.monotonic() - began:.0f} s", flush=True)
        time.sleep(10)
    t["waited_s"] = round(time.monotonic() - began)
    return t

def fmt_thermal(t):
    return " ".join(f"{k} {t[k] / 1000:.1f}" for k in ("z9", "z10", "z11")) + f" degC at {t['at']}"
