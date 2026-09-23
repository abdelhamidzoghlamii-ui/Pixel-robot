"""Offline mock checks for archive provenance and publication state."""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def command(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=True)


class ArchiveRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.top = Path(self.temp.name)
        self.repo = self.top / "robot"
        self.archive = self.repo / "benchmark/strategic_selector"
        self.archive.mkdir(parents=True)
        for name in ("archive_run.py", "phone_snapshot.py", "robot_selector_benchmark.py"):
            shutil.copy2(HERE / name, self.archive / name)
        command("git", "init", "-b", "main", str(self.repo))
        command("git", "config", "user.name", "Archive Test", cwd=self.repo)
        command("git", "config", "user.email", "archive@example.test", cwd=self.repo)
        command("git", "add", "benchmark/strategic_selector", cwd=self.repo)
        command("git", "commit", "-m", "base", cwd=self.repo)
        self.remote = self.top / "origin.git"
        command("git", "init", "--bare", str(self.remote))
        command("git", "remote", "add", "origin", str(self.remote), cwd=self.repo)
        command("git", "push", "-u", "origin", "main", cwd=self.repo)
        sys.path.insert(0, str(self.archive))
        self.addCleanup(lambda: sys.path.remove(str(self.archive)))
        spec = importlib.util.spec_from_file_location("archive_test_module", self.archive / "archive_run.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.probe = self.top / "empty_phone_fixture"
        self.probe.mkdir()

    def make_run(self):
        args = argparse.Namespace(model="mock", quick=True, root=self.archive / "results", probe_root=self.probe)
        with patch("builtins.print"):
            self.assertEqual(self.module.run(args), 0)
        return sorted((self.archive / "results").iterdir())[-1]

    def test_missing_metadata_duplicate_and_corrupt_evidence(self):
        run = self.make_run()
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["phone_state"]["before"]["thermal_zones"]["status"], "unavailable")
        self.assertEqual(manifest["phone_state"]["after"]["cpu_frequency"]["status"], "unavailable")
        self.assertEqual(manifest["phone_state"]["before"]["battery"]["status"]["status"], "unavailable")
        self.assertEqual(manifest["phone_state"]["after"]["memory"]["MemAvailable"]["status"], "unavailable")
        self.assertEqual(manifest["model_provenance"]["status"], "not_applicable")
        self.assertEqual(manifest["phone_snapshot_sha256"], self.module.digest(self.archive / "phone_snapshot.py"))
        with patch.object(self.module.importlib.metadata, "version", side_effect=self.module.importlib.metadata.PackageNotFoundError):
            self.assertTrue(all(item["status"] == "unavailable" for item in self.module.package_versions().values()))
        instant = datetime.strptime(run.name[:18], "%Y-%m-%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        frozen = type("Frozen", (), {"now": staticmethod(lambda tz: instant)})
        with patch.object(self.module, "datetime", frozen), patch.object(self.module.uuid, "uuid4", return_value=types.SimpleNamespace(hex=run.name[-8:])):
            with self.assertRaises(FileExistsError):
                self.module.run(argparse.Namespace(model="mock", quick=True, root=self.archive / "results", probe_root=self.probe))
        with (run / "stdout.txt").open("ab") as stream:
            stream.write(b"tampered")
        with patch("builtins.print"):
            reports = self.module.audit(self.archive / "results")
        self.assertEqual(reports[0]["local_integrity"], "corrupt")

    def test_publication_uses_fresh_remote_not_tracking_ref(self):
        first = self.make_run()
        with patch("builtins.print"):
            local = self.module.audit(self.archive / "results")
        self.assertEqual(local[0]["local_git"], "untracked")
        self.assertEqual(local[0]["origin_main"], "unknown")
        audit_cmd = [sys.executable, str(self.archive / "archive_run.py"), "audit", "--root", str(self.archive / "results")]
        self.assertEqual(subprocess.run(audit_cmd, cwd=self.repo, capture_output=True).returncode, 1)
        command("git", "add", "benchmark/strategic_selector/results", cwd=self.repo)
        command("git", "commit", "-m", "first run", cwd=self.repo)
        command("git", "push", "origin", "main", cwd=self.repo)
        with patch("builtins.print"):
            confirmed = self.module.audit(self.archive / "results", remote=True)
        self.assertEqual(confirmed[0]["origin_main"], "confirmed")
        self.assertEqual(subprocess.run([*audit_cmd, "--remote"], cwd=self.repo, capture_output=True).returncode, 0)
        first_manifest = first / "manifest.json"
        original_manifest = first_manifest.read_bytes()
        first_manifest.write_bytes(original_manifest + b"\n")  # valid JSON, but not committed
        with patch("builtins.print"):
            changed = self.module.audit(self.archive / "results", remote=True)
        self.assertEqual(changed[0]["local_git"], "uncommitted")
        self.assertEqual(changed[0]["origin_main"], "not_confirmed")
        first_manifest.write_bytes(original_manifest)
        second = self.make_run()
        command("git", "add", "benchmark/strategic_selector/results", cwd=self.repo)
        command("git", "commit", "-m", "second run local only", cwd=self.repo)
        command("git", "update-ref", "refs/remotes/origin/main", "HEAD", cwd=self.repo)
        with patch("builtins.print"):
            reports = self.module.audit(self.archive / "results", remote=True)
        self.assertEqual(next(r for r in reports if r["run_id"] == second.name)["origin_main"], "not_confirmed")
        self.assertEqual(subprocess.run([*audit_cmd, "--remote"], cwd=self.repo, capture_output=True).returncode, 1)
        command("git", "remote", "set-url", "origin", str(self.top / "missing.git"), cwd=self.repo)
        with patch("builtins.print"):
            unavailable = self.module.audit(self.archive / "results", remote=True)
        self.assertTrue(all(r["origin_main"] == "unknown" for r in unavailable))

    def test_available_phone_readings_are_timestamped_outside_mock_process(self):
        samples = {
            "sys/class/thermal/thermal_zone9/type": "BIG\n",
            "sys/class/thermal/thermal_zone9/temp": "41000\n",
            "sys/class/thermal/cooling_device0/type": "thermal-cpufreq-2\n",
            "sys/class/thermal/cooling_device0/cur_state": "1\n",
            "sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq": "2200000\n",
            "sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq": "2500000\n",
            "sys/devices/system/cpu/cpufreq/policy6/cpuinfo_max_freq": "2850000\n",
            "sys/class/power_supply/battery/status": "Discharging\n",
            "sys/class/power_supply/battery/capacity": "63\n",
            "sys/class/power_supply/usb/online": "0\n",
            "proc/meminfo": "MemTotal: 7000000 kB\nMemAvailable: 3000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
        }
        for relative, value in samples.items():
            path = self.probe / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        manifest = json.loads((self.make_run() / "manifest.json").read_text())
        before, after = (manifest["phone_state"][when] for when in ("before", "after"))
        for snapshot in (before, after):
            self.assertEqual(snapshot["thermal_zones"]["entries"]["thermal_zone9"]["temp"]["value"], "41000")
            self.assertEqual(snapshot["cpu_frequency"]["entries"]["policy6"]["scaling_cur_freq"]["value"], "2200000")
            self.assertEqual(snapshot["battery"]["status"]["value"], "Discharging")
            self.assertEqual(snapshot["memory"]["MemAvailable"]["value"], "3000000 kB")
            self.assertIn("measured_utc", snapshot["charging_sources"]["usb"])
        self.assertLessEqual(before["ended_utc"], manifest["process_started_utc"])
        self.assertLessEqual(manifest["process_ended_utc"], after["started_utc"])

    def test_von_reported_weight_hash_and_laya_uncertainty(self):
        cache = self.top / "cache"
        revision = "a" * 40
        weight = cache / "hub/models--wfzyx--von/snapshots" / revision / "option_marker.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"fake test weights")
        log = self.top / "stdout.txt"
        log.write_text(f"[von] Loaded von-1.1.0 weights from Hugging Face Hub 'wfzyx/von:option_marker.pt' ({weight}) (input-conditioned calibration map active)\n")
        report = self.module.model_provenance("von", log, cache)
        self.assertEqual(report["runtime_model_label"], "von-1.1.0")
        self.assertEqual(report["checkpoint_revision"], revision)
        self.assertEqual(report["weight_identity"]["sha256_after_run"], self.module.digest(weight))
        self.assertEqual(self.module.model_provenance("laya", log, cache)["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
