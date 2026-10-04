"""Retry the recorded ORT dynamic-quantization failure using its suggested option."""
from build_variants import HERE, SOURCE, MODELS, sha
import json
import sys


def main():
    assert sys.prefix == '/termux-home/ext/venv-relate-export'
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    report = json.loads((HERE / 'builds.json').read_text())
    entry = report['variants']['int8']
    entry['attempts'] = [{k: entry[k] for k in ('command', 'status', 'error')}]
    directory = MODELS / 'relsgg-vits16-int8'
    output = directory / 'relateanything.onnx'
    assert not output.exists(), 'do not overwrite an existing graph'
    entry['command'] = (f'{sys.executable} {HERE / "quantize_int8.py"}; quantize_dynamic('
        f'{str(SOURCE / "relateanything.onnx")!r}, {str(output)!r}, weight_type=QuantType.QInt8, '
        "extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT})")
    print(entry['command'], flush=True)
    try:
        quantize_dynamic(str(SOURCE / 'relateanything.onnx'), str(output), weight_type=QuantType.QInt8,
                         extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT})
        (directory / 'relateanything.json').write_bytes((SOURCE / 'relateanything.json').read_bytes())
        graph = onnx.load(str(output), load_external_data=False)
        external = [t.name for t in graph.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
        assert not external
        entry.update(status='AVAILABLE', bytes=output.stat().st_size, sha256=sha(output), external_data=external,
                     bank_sha256=sha(directory / 'predicate_bank.npz'), sidecar_sha256=sha(directory / 'relateanything.json'))
        entry.pop('error', None)
    except Exception as error:
        entry.update(status='BUILD FAILED', error=repr(error))
    (HERE / 'builds.json').write_text(json.dumps(report, indent=2) + '\n')
    print(entry['status'], flush=True)


if __name__ == '__main__':
    main()
