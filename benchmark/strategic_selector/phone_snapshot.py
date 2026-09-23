"""Read-only, untimed phone state capture for future selector runs."""
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read(root, source, unit=None):
    path = root / source.lstrip("/")
    measured = utc_now()
    try:
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError("empty")
        return {"status": "available", "value": value, "unit": unit,
                "source": str(path), "measured_utc": measured}
    except (OSError, UnicodeError, ValueError) as exc:
        return {"status": "unavailable", "value": None, "unit": unit,
                "source": str(path), "measured_utc": measured,
                "reason": type(exc).__name__}


def group(root, directory, prefix, fields):
    base = root / directory.lstrip("/")
    entries = sorted(p for p in base.glob(prefix + "[0-9]*") if p.is_dir())
    if not entries:
        return {"status": "unavailable", "reason": "no_matching_paths", "entries": {}}
    return {"status": "available", "entries": {
        entry.name: {name: read(root, f"{directory}/{entry.name}/{name}", unit)
                     for name, unit in fields.items()}
        for entry in entries}}


def memory(root):
    source = "/proc/meminfo"
    measured = utc_now()
    path = root / source.lstrip("/")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = dict(line.split(":", 1) for line in lines if ":" in line)
    except (OSError, UnicodeError):
        values = {}
    return {key: {"status": "available" if key in values else "unavailable",
                  "value": values[key].strip() if key in values else None,
                  "source": str(path), "measured_utc": measured,
                  **({} if key in values else {"reason": "missing_or_unreadable"})}
            for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}


def capture(root=Path("/")):
    root = Path(root)
    snapshot = {"started_utc": utc_now(), "source_root": str(root)}
    snapshot["thermal_zones"] = group(root, "/sys/class/thermal", "thermal_zone",
                                      {"type": None, "temp": "millidegree_C_raw"})
    snapshot["cooling_devices"] = group(root, "/sys/class/thermal", "cooling_device",
                                        {"type": None, "cur_state": "raw_level"})
    snapshot["virtual_skin"] = {"status": "unavailable", "value": None,
                                "reason": "not_directly_read_from_sysfs; no estimate made"}
    snapshot["cpu_frequency"] = group(root, "/sys/devices/system/cpu/cpufreq", "policy",
                                       {"scaling_cur_freq": "kHz", "scaling_max_freq": "kHz",
                                        "cpuinfo_max_freq": "kHz"})
    snapshot["battery"] = {name: read(root, "/sys/class/power_supply/battery/" + name, unit)
                           for name, unit in {"status": None, "capacity": "percent",
                                              "temp": "deci_C_raw", "voltage_now": "microvolt",
                                              "current_now": "microamp"}.items()}
    snapshot["charging_sources"] = {name: read(root, f"/sys/class/power_supply/{name}/online", "raw_boolean")
                                    for name in ("usb", "wireless", "ac")}
    snapshot["memory"] = memory(root)
    snapshot["ended_utc"] = utc_now()
    return snapshot
