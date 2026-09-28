# Export stock Ultralytics yolo11/yolo26 n/s/m to ONNX at 320 and 640 with the deployed file's
# metadata args (batch 1, half False, dynamic False, simplify True, nms False; opset 20 as in the
# deployed graph), plus an int8 dynamic-quantized copy of the deployed yolo11m.onnx. Run in the
# yolo_bench venv from /termux-home/yolo_bench/weights.
import os, shutil
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

OUT = '/termux-home/yolo_bench/models'
for fam in ('yolo11', 'yolo26'):
    for size in 'nsm':
        for imgsz in (320, 640):
            dst = f'{OUT}/{fam}{size}_{imgsz}.onnx'
            if os.path.exists(dst):
                continue
            kw = dict(format='onnx', imgsz=imgsz, batch=1, half=False, dynamic=False,
                      simplify=True, opset=20, nms=False, device='cpu')
            if fam == 'yolo26':
                kw['end2end'] = False  # deployed file is end2end False: same [1,84,N] output and post-processing
            f = YOLO(f'{fam}{size}.pt').export(**kw)
            shutil.move(f, dst)

dst = f'{OUT}/yolo11m_deployed_int8dyn.onnx'
if not os.path.exists(dst):
    quantize_dynamic('/termux-home/robot/yolo11m.onnx', dst, weight_type=QuantType.QInt8)
