"""Variant configuration and immutable quality gates for speed runs."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'desk2'))
import desk_check as dc
import speed_block as sb

VARIANTS = ('fp32', 'int8', 'xnnpack', 'img384', 'img336')
CLUSTERS = {'MID': ({4, 5}, 2), 'BIG': ({6, 7}, 2), 'LITTLE': ({0, 1, 2, 3}, 4)}
NAME = 'relsgg-vits16'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def directory(variant):
    return dc.MODELS / (NAME if variant in ('fp32', 'xnnpack') else f'{NAME}-{variant}')


def identity(variant):
    d = directory(variant)
    return {n: sha(d / n) for n in ('relateanything.onnx', 'relateanything.json', 'predicate_bank.npz')}


def make_head(variant, threads=2):
    providers = ['CPUExecutionProvider']
    if variant == 'xnnpack':
        if 'XnnpackExecutionProvider' not in dc.ort.get_available_providers():
            raise RuntimeError('XnnpackExecutionProvider NOT AVAILABLE')
        # One XNNPACK thread avoids another private worker pool; the pinned ORT
        # caller runs its kernels. CPU fallback retains the cluster thread count.
        providers = [('XnnpackExecutionProvider', {'intra_op_num_threads': '1'}), 'CPUExecutionProvider']
    head = dc.Head(NAME, directory=directory(variant), size={'img384': 384, 'img336': 336}.get(variant, 448),
                   threads=threads, providers=providers)
    if variant == 'xnnpack' and head.session.get_providers()[0] != 'XnnpackExecutionProvider':
        raise RuntimeError('XNNPACK initialization fell back')
    head.artifact_identity = identity(variant)
    return head


def require_quality(variant):
    dc.require_parity(NAME)
    quality = json.loads((HERE / 'quality.json').read_text())
    entry = quality['variants'][variant]
    if entry.get('passed') is not True or entry.get('identity') != identity(variant):
        raise RuntimeError(f'{variant}: quality missing/failed or artifact SHA-256 changed')
    refs = reference_hashes()
    if quality.get('reference_hashes') != refs or quality.get('reference_identity') != identity('fp32'):
        raise RuntimeError('quality reference SHA-256 changed')


def reference_hashes():
    paths = [dc.HERE / 'selected_photos.json', dc.HERE / 'parity_valid_relsgg-vits16.json',
             dc.HERE / 'parity_desk3/feed_components.json']
    paths += sorted((dc.HERE / 'results' / NAME).glob('*.json'))
    paths += sorted((dc.HERE / 'parity_desk3').glob('*_relsgg-vits16_onnx.npz'))
    return {str(p.relative_to(dc.HERE)): sha(p) for p in paths}


def build(variant, cluster):
    cpus, threads = CLUSTERS[cluster]
    return sb.build(NAME, cpus, threads, lambda name, threads: make_head(variant, threads))


def planned_blocks(quality):
    passing = [v for v in VARIANTS[1:] if quality['variants'].get(v, {}).get('passed') is True]
    blocks = [('fp32', 'MID')] + [(v, 'MID') for v in passing] + [('fp32', 'BIG'), ('fp32', 'LITTLE')]
    candidates = ['fp32'] + passing
    fastest = min(candidates, key=lambda v: quality['variants'][v]['informal_mid_median_ms'])
    if fastest != 'fp32':
        blocks.append((fastest, 'BIG'))
    return blocks + [('fp32', 'MID')]
