"""Standard ORT constant-MatMul weight quantization after the default graph failed."""
from build_variants import HERE, SOURCE, MODELS, sha
import json
import sys

assert sys.prefix == '/termux-home/ext/venv-relate-export'
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

report = json.loads((HERE / 'builds.json').read_text())
entry = report['variants']['int8']
entry['attempts'].append({k: entry[k] for k in ('command', 'status', 'bytes', 'sha256')})
entry['attempts'][-1]['native_load_error'] = 'MatMulInteger ShapeInferenceError: Incompatible dimensions for matrix multiplication (quality_initial.stderr)'
directory = MODELS / 'relsgg-vits16-int8'
output = directory / 'relateanything.onnx'
failed = directory / 'failed_default_tensor_type.onnx'
assert not failed.exists()
output.rename(failed)
entry['command'] = (f'{sys.executable} {HERE / "quantize_matmul.py"}; quantize_dynamic('
    f'{str(SOURCE / "relateanything.onnx")!r}, {str(output)!r}, weight_type=QuantType.QInt8, '
    "op_types_to_quantize=['MatMul'], extra_options={'MatMulConstBOnly': True})")
print(entry['command'], flush=True)
try:
    quantize_dynamic(str(SOURCE / 'relateanything.onnx'), str(output), weight_type=QuantType.QInt8,
                     op_types_to_quantize=['MatMul'], extra_options={'MatMulConstBOnly': True})
    graph = onnx.load(str(output), load_external_data=False)
    external = [t.name for t in graph.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    assert not external
    weights = [t.name for t in graph.graph.initializer if t.data_type == onnx.TensorProto.INT8]
    assert weights
    entry.update(status='AVAILABLE', bytes=output.stat().st_size, sha256=sha(output), external_data=external,
                 int8_initializers=len(weights), quantization_scope='constant MatMul weights int8; Conv/Gemm remain fp32')
except Exception as error:
    entry.update(status='NOT AVAILABLE', build_status='BUILD FAILED', error=repr(error),
                 artifact_path=str(failed), bytes=failed.stat().st_size, sha256=sha(failed))
(HERE / 'builds.json').write_text(json.dumps(report, indent=2) + '\n')
print(entry['status'], flush=True)
