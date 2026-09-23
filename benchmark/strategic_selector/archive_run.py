#!/usr/bin/env python3
"""Run the archived selector on the Pixel, or verify archived file hashes.

Activate /termux-home/laya-test/venv and set HF_HOME before `run`.
The `mock --quick` mode is only an offline archive smoke check.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "robot_selector_benchmark.py"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    count = 0
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise ValueError(f"No manifests in {root}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        source = manifest["source"]
        if digest(BASE / source["archived_path"]) != source["archived_sha256"]:
            raise ValueError(f"Source hash mismatch: {manifest_path}")
        for entry in manifest["files"].values():
            archived = manifest_path.parent / entry["archived_name"]
            if digest(archived) != entry["archived_sha256"]:
                raise ValueError(f"Hash mismatch: {archived}")
            count += 1
    print(f"Verified {count} archived files in {root}")


def run(args):
    if args.model != "mock" and not os.environ.get("HF_HOME"):
        raise SystemExit("Set HF_HOME to the existing offline model cache first")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ") + f"-{args.model}-{uuid.uuid4().hex[:8]}"
    folder = root / run_id
    folder.mkdir()  # never reuse a run directory
    output = folder / "results.json"
    stdout = folder / "stdout.txt"
    stderr = folder / "stderr.txt"
    command = [sys.executable, str(SOURCE), "--model", args.model, "--output", str(output)]
    if args.quick:
        command.append("--quick")
    if args.model != "mock":
        command = ["taskset", "-c", "4-7", *command]
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0", OMP_NUM_THREADS="4")
    started = datetime.now(timezone.utc).isoformat()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        result = subprocess.run(command, stdout=out, stderr=err, env=env, check=False)
    files = {}
    for path in (output, stdout, stderr):
        if path.exists():
            files[path.name] = {"archived_name": path.name, "archived_sha256": digest(path), "bytes": path.stat().st_size}
    packages = {}
    for name in ("torch", "laya", "von-sdk"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "run_id": run_id, "kind": "original", "model": args.model,
        "benchmark_version": 2, "framing": "five development frames; development-selected heldout frame; reversed order",
        "fixture_split": "variants 0,1 development; variant 2 heldout; model-specific generated cases",
        "started_utc": started, "exit_code": result.returncode,
        "command": command, "python": sys.version.split()[0],
        "packages": packages,
        "environment": {key: env.get(key) for key in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS")},
        "source": {"archived_path": str(SOURCE.relative_to(BASE)), "archived_sha256": digest(SOURCE)},
        "files": files,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"RUN={folder} EXIT={result.returncode}")
    verify(root)
    for git_args in (["check-ignore", "-v", str(folder)], ["status", "--short", "--untracked-files=all", "--", str(folder)]):
        check = subprocess.run(["git", *git_args], cwd=BASE, text=True, capture_output=True, check=False)
        print(f"git {' '.join(git_args)}: {check.stdout.strip() or '(no output)'}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", choices=("laya", "von", "mock"), required=True)
    run_parser.add_argument("--quick", action="store_true")
    run_parser.add_argument("--root", type=Path, default=BASE / "results")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, default=BASE / "results")
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.root)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
