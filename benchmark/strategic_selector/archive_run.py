#!/usr/bin/env python3
"""Capture future selector runs and audit local integrity and publication.

Activate the existing phone venv and set HF_HOME before a real run. `audit`
never commits or pushes. `--remote` queries origin/main afresh; local tracking
refs alone are not publication evidence.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from phone_snapshot import capture, utc_now

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "robot_selector_benchmark.py"
RESULTS = BASE / "results"
VON_PATH = re.compile(r"\[von\] Loaded (von-[^\s]+) weights from Hugging Face Hub 'wfzyx/von:option_marker\.pt' \((/[^\n)]+)\)")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, cwd=BASE, timeout=20):
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(["git", *args], cwd=cwd, env=env, text=True,
                              capture_output=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def package_versions():
    names = ("torch", "laya", "von-sdk", "huggingface-hub", "transformers", "safetensors", "tokenizers")
    versions = {}
    for name in names:
        try:
            versions[name] = {"status": "available", "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            versions[name] = {"status": "unavailable", "version": None}
    return versions


def model_provenance(model, stdout, cache_home):
    if model == "mock":
        return {"status": "not_applicable", "requested_model_id": None}
    info = {"requested_model_id": "convaiinnovations/laya" if model == "laya" else "wfzyx/von",
            "checkpoint_revision": None, "weight_identity": None,
            "status": "unverified", "encoder_weights": "unverified"}
    if model == "laya":
        info["reason"] = "The archived v2 loader does not emit its resolved snapshot or weight path; a model ID/cache ref is not proof of loaded bytes."
        return info
    match = VON_PATH.search(stdout.read_text(encoding="utf-8", errors="replace"))
    if not match:
        info["reason"] = "No usable Von loaded-weight path in stdout."
        return info
    info["runtime_model_label"] = match.group(1)
    weight = Path(match.group(2))
    expected = cache_home / "hub/models--wfzyx--von/snapshots"
    if (weight.parent.parent != expected or weight.name != "option_marker.pt"
            or not re.fullmatch(r"[0-9a-f]{40}", weight.parent.name)):
        info["reason"] = "Runtime-reported path is outside the expected Von snapshot layout."
        return info
    try:
        if not weight.resolve(strict=True).is_relative_to(cache_home.resolve(strict=True)):
            raise ValueError("resolved path escapes cache")
        identity = {"path_reported_by_runtime": str(weight),
                    "sha256_after_run": digest(weight), "bytes": weight.stat().st_size,
                    "measured_utc": utc_now(),
                    "evidence": "Von stdout reports this path after its loader; file hashed after process exit."}
    except (OSError, ValueError) as exc:
        info["reason"] = f"Could not verify reported weight file: {type(exc).__name__}"
        return info
    info.update(status="partial_runtime_report_and_hash",
                checkpoint_revision=weight.parent.name, weight_identity=identity,
                reason="Option-marker weight verified from runtime report; base encoder and calibration file paths were not independently observed.")
    return info


def run(args):
    cache_home = None
    if args.model != "mock":
        if not os.environ.get("HF_HOME"):
            raise SystemExit("Set HF_HOME to the existing offline model cache first")
        cache_home = Path(os.environ["HF_HOME"]).resolve()
        if not cache_home.is_dir():
            raise SystemExit(f"Offline cache does not exist: {cache_home}")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ") + f"-{args.model}-{uuid.uuid4().hex[:8]}"
    folder = root / run_id
    folder.mkdir()  # exclusive: a duplicate ID cannot overwrite an old run
    output, stdout, stderr = (folder / name for name in ("results.json", "stdout.txt", "stderr.txt"))
    command = [sys.executable, str(SOURCE), "--model", args.model, "--output", str(output)]
    if args.quick:
        command.append("--quick")
    if args.model != "mock":
        command = ["taskset", "-c", "4-7", *command]
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               USE_TF="0", OMP_NUM_THREADS="4")
    if cache_home:
        env.update(HF_HOME=str(cache_home), HF_HUB_CACHE=str(cache_home / "hub"),
                   HUGGINGFACE_HUB_CACHE=str(cache_home / "hub"))
    source_hash = digest(SOURCE)
    before = capture(args.probe_root)
    started = utc_now()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        try:
            result = subprocess.run(command, cwd=folder, stdout=out, stderr=err, env=env, check=False)
            exit_code = result.returncode
        except OSError as exc:
            err.write(f"EXECUTION_ERROR={type(exc).__name__}: {exc}\n".encode())
            exit_code = 127
    ended = utc_now()
    if exit_code == 0 and not output.exists():
        with stderr.open("ab") as err:
            err.write(b"ARCHIVE_ERROR=successful process produced no results.json\n")
        exit_code = 1
    after = capture(args.probe_root)
    files = {path.name: {"archived_name": path.name, "archived_sha256": digest(path),
                         "bytes": path.stat().st_size}
             for path in (output, stdout, stderr) if path.exists()}
    manifest = {
        "run_id": run_id, "kind": "original", "model": args.model,
        "benchmark_version": 2,
        "framing": "five development frames; development-selected heldout frame; reversed order",
        "fixture_split": "quick: 4 development/2 heldout" if args.quick else "variants 0,1 development; variant 2 heldout; model-specific generated cases",
        "process_started_utc": started, "process_ended_utc": ended, "exit_code": exit_code,
        "command": command, "cwd": str(folder), "python_executable": sys.executable,
        "python_version": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "packages": package_versions(),
        "environment": {key: env.get(key) for key in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                                                    "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS")},
        "source": {"archived_path": SOURCE.name, "archived_sha256": source_hash},
        "source_sha256_after_run": digest(SOURCE),
        "archive_helper_sha256": digest(Path(__file__)),
        "phone_snapshot_sha256": digest(BASE / "phone_snapshot.py"),
        "model_provenance": model_provenance(args.model, stdout, cache_home),
        "phone_state": {"before": before, "after": after},
        "files": files,
    }
    with (folder / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"RUN={folder} EXIT={exit_code}")
    print("LOCAL ARCHIVE ONLY: origin/main has not been checked; review, authorize, publish, then audit --remote.")
    audit(root, remote=False)
    return exit_code


def fresh_remote_sha():
    result = git("ls-remote", "--exit-code", "origin", "refs/heads/main")
    if result is None or result.returncode != 0:
        return None, "origin/main query unavailable"
    parts = result.stdout.strip().split()
    if len(parts) != 2 or parts[1] != "refs/heads/main" or not re.fullmatch(r"[0-9a-f]{40,64}", parts[0]):
        return None, "origin/main response invalid"
    return parts[0], None


def audit(root=RESULTS, remote=False):
    root = Path(root).resolve()
    repo_result = git("rev-parse", "--show-toplevel")
    repo = Path(repo_result.stdout.strip()).resolve() if repo_result and repo_result.returncode == 0 else None
    remote_sha, remote_error = fresh_remote_sha() if remote else (None, "remote not queried")
    reports = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        errors, paths = [], [folder / "manifest.json"]
        manifest = None
        try:
            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            for entry in manifest["files"].values():
                name = entry["archived_name"]
                if Path(name).name != name:
                    raise ValueError("unsafe archived name")
                path = folder / name
                paths.append(path)
                if digest(path) != entry["archived_sha256"]:
                    errors.append(f"hash mismatch: {name}")
            source = manifest["source"]
            path = (BASE / source["archived_path"]).resolve()
            if not path.is_relative_to(BASE):
                raise ValueError("source outside archive")
            paths.append(path)
            if digest(path) != source["archived_sha256"]:
                errors.append("source hash mismatch")
            if manifest.get("source_sha256_after_run", source["archived_sha256"]) != source["archived_sha256"]:
                errors.append("source changed during run")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"missing or invalid manifest/evidence: {type(exc).__name__}")
        tracked, changes = [], []
        inside_repo = repo is not None and folder.is_relative_to(repo)
        if inside_repo:
            for path in paths:
                rel = str(path.relative_to(repo))
                check = git("ls-files", "--error-unmatch", "--", rel, cwd=repo)
                if check is None or check.returncode != 0:
                    tracked.append(rel)
            status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *(str(p.relative_to(repo)) for p in paths), cwd=repo)
            if status and status.returncode == 0:
                changes = status.stdout.splitlines()
            else:
                errors.append("git status unavailable")
        local = "corrupt" if errors else "untracked" if tracked else "uncommitted" if changes else "clean" if inside_repo else "outside_git"
        publication = "unknown"
        reason = remote_error
        if remote and remote_sha and not errors and inside_repo:
            exists = git("cat-file", "-e", f"{remote_sha}^{{commit}}")
            if exists is None or exists.returncode != 0:
                reason = "fresh remote SHA unavailable locally; fetch origin main and rerun audit"
            elif tracked or changes:
                publication, reason = "not_confirmed", "run has untracked or uncommitted files"
            else:
                same = True
                for path in paths:
                    rel = str(path.relative_to(repo))
                    remote_blob = git("rev-parse", f"{remote_sha}:{rel}", cwd=repo)
                    local_blob = git("hash-object", str(path), cwd=repo)
                    if (not remote_blob or remote_blob.returncode != 0 or not local_blob
                            or local_blob.returncode != 0 or remote_blob.stdout.strip() != local_blob.stdout.strip()):
                        same = False
                        break
                publication, reason = ("confirmed", "fresh origin/main tree matches every run file and source") if same else ("not_confirmed", "run bytes absent or different on fresh origin/main")
        if errors:
            next_step = "Restore original evidence, then rerun audit; do not publish corrupted files."
        elif tracked or changes:
            next_step = "Review this run, obtain authorization to commit and push it, then rerun audit --remote."
        elif publication == "confirmed":
            next_step = "Published copy confirmed; retain raw phone originals."
        else:
            next_step = "Run audit --remote; if remote objects are unavailable, fetch origin main and rerun. No backup is confirmed."
        reports.append({"run_id": folder.name, "local_integrity": "corrupt" if errors else "valid",
                        "errors": errors, "untracked_files": tracked, "uncommitted_paths": changes,
                        "local_git": local, "origin_main": publication,
                        "fresh_remote_sha": remote_sha, "remote_reason": reason,
                        "next_step": next_step})
    if not reports:
        raise ValueError(f"No run directories in {root}")
    for report in reports:
        print(json.dumps(report, sort_keys=True))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", choices=("laya", "von", "mock"), required=True)
    run_parser.add_argument("--quick", action="store_true")
    run_parser.add_argument("--root", type=Path, default=RESULTS)
    run_parser.add_argument("--probe-root", type=Path, default=Path("/"), help="test fixture root; normal phone runs use /")
    for action in ("audit", "verify"):
        p = sub.add_parser(action)
        p.add_argument("--root", type=Path, default=RESULTS)
        if action == "audit":
            p.add_argument("--remote", action="store_true", help="query origin/main now; never trust a cached tracking ref")
    args = parser.parse_args()
    if args.action == "run":
        return run(args)
    reports = audit(args.root, remote=getattr(args, "remote", False))
    unresolved = any(r["local_integrity"] == "corrupt" or r["local_git"] != "clean"
                     or (getattr(args, "remote", False) and r["origin_main"] != "confirmed")
                     for r in reports)
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
