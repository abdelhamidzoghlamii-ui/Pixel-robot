"""Build external variants; run only with the existing Debian export venv."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HOME = Path('/termux-home')
MODELS = HOME / 'models/relate_anything'
SOURCE = MODELS / 'relsgg-vits16'
EXTERNAL = HOME / 'ext/RelateAnything'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if Path(sys.prefix).resolve() != HOME / 'ext/venv-relate-export':
        raise SystemExit('existing Debian export venv required')
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    reference = SOURCE / 'relateanything.onnx'
    assert sha(reference) == '24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5'
    versions = {n: md.version(n) for n in ('torch', 'onnx', 'onnxruntime', 'numpy', 'transformers')}
    exporter = EXTERNAL / 'deploy/export_onnx.py'
    report = {'versions': versions, 'python': sys.version, 'exporter_sha256': sha(exporter), 'variants': {}}
    for variant, size in [('int8', 448), ('img384', 384), ('img336', 336)]:
        directory = MODELS / f'relsgg-vits16-{variant}'
        directory.mkdir(exist_ok=False)
        output = directory / 'relateanything.onnx'
        # Banks remain outside the repository; reuse the exact reference bytes.
        shutil.copyfile(SOURCE / 'predicate_bank.npz', directory / 'predicate_bank.npz')
        item = {'directory': str(directory), 'img_size': size}
        try:
            if variant == 'int8':
                item['command'] = (f'{sys.executable} {HERE / "build_variants.py"}: '
                    f'quantize_dynamic({str(reference)!r}, {str(output)!r}, weight_type=QuantType.QInt8)')
                print(item['command'], flush=True)
                quantize_dynamic(str(reference), str(output), weight_type=QuantType.QInt8)
                shutil.copyfile(SOURCE / 'relateanything.json', directory / 'relateanything.json')
            else:
                command = [sys.executable, str(exporter), '--checkpoint', str(SOURCE / 'model.pth'),
                    '--out', str(output), '--vocab-npz', str(SOURCE / 'predicate_bank.npz'),
                    '--vocab-mode', 'input', '--opset', '17', '--img-size', str(size),
                    '--max-boxes', '32', '--weights', 'ema', '--check']
                item['command'] = shlex.join(command)
                print(item['command'], flush=True)
                with (HERE / f'build_{variant}.stdout').open('w') as stdout, (HERE / f'build_{variant}.stderr').open('w') as stderr:
                    subprocess.run(command, cwd=EXTERNAL, check=True, stdout=stdout, stderr=stderr,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
            graph = onnx.load(str(output), load_external_data=False)
            external = [t.name for t in graph.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
            if external:
                raise RuntimeError('external-data tensors require additional hash coverage: ' + repr(external))
            item.update(status='AVAILABLE', bytes=output.stat().st_size, sha256=sha(output), external_data=external,
                        bank_sha256=sha(directory / 'predicate_bank.npz'), sidecar_sha256=sha(directory / 'relateanything.json'))
        except Exception as error:
            item.update(status='NOT SUPPORTED' if variant.startswith('img') else 'BUILD FAILED', error=repr(error))
        report['variants'][variant] = item
        (HERE / 'builds.json').write_text(json.dumps(report, indent=2) + '\n')
        print(variant, item['status'], flush=True)
        if variant == 'img384' and item['status'] != 'AVAILABLE':
            report['variants']['img336'] = {'status': 'NOT SUPPORTED', 'reason': 'stopped V3/V4 after img384 failure'}
            break
    report['exporter_unchanged'] = sha(exporter) == report['exporter_sha256']
    # Explicitly resolve FIX1 review finding 3 for the reference as well.
    graph = onnx.load(str(reference), load_external_data=False)
    report['reference_external_data'] = [t.name for t in graph.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    (HERE / 'builds.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
