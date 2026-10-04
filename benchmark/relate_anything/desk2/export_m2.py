"""Debian venv only: invoke the author's exporter with the M1 deployment settings."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path('/termux-home')
REPO = ROOT / 'ext/RelateAnything'
MODEL = ROOT / 'models/relate_anything/relsgg-vits16'
meta = json.loads((ROOT / 'models/relate_anything/relsgg-vits16plus/relateanything.json').read_text())
# Preserve the published M2 sidecar before the author's exporter replaces it.
original = MODEL / 'published_relateanything.json'
if not original.exists():
    original.write_bytes((MODEL / 'relateanything.json').read_bytes())
command = [sys.executable, str(REPO / 'deploy/export_onnx.py'),
           '--checkpoint', str(MODEL / 'model.pth'), '--out', str(MODEL / 'relateanything.onnx'),
           '--vocab-npz', str(MODEL / 'predicate_bank.npz'), '--vocab-mode', meta['vocab_mode'],
           '--opset', str(meta['opset']), '--img-size', str(meta['img_size']),
           '--max-boxes', str(meta['max_boxes']), '--weights', 'ema', '--check']
# --vocab-npz exports all 243 bank rows; vocabulary is a dynamic input.
# At inference and horse parity use the same default 35 rows as M1.
print('EXACT COMMAND:', shlex.join(command), flush=True)
result = subprocess.run(command, cwd=REPO, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                             'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
raise SystemExit(result.returncode)
