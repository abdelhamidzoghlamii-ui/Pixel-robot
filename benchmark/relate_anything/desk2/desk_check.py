"""Motors-off RelateAnything desk check. Native Termux Python; no torch/cv2."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[2]
HOME = Path('/data/data/com.termux/files/home')
MODELS = HOME / 'models/relate_anything'
EXTERNAL = HOME / 'ext/RelateAnything'
LABEL = 'INFORMAL — agents resident, NOT VALID TIMING'
NAMES = ('relsgg-vits16plus', 'relsgg-vits16')
OUTPUTS = ['pred_logits', 'pair_logits', 'sub_idx', 'obj_idx', 'valid_mask']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def open_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def preprocess(image, boxes, size=448, maximum=32):
    boxes = np.asarray(boxes, np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not 2 <= len(boxes) <= maximum:
        raise ValueError('need 2..32 xyxy boxes; never silently truncate detections')
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError('invalid xyxy boxes')
    width, height = image.size
    b = boxes / np.array([width, height, width, height], np.float32)
    padded = np.zeros((1, maximum, 4), np.float32)
    padded[0, :len(b), :2] = (b[:, :2] + b[:, 2:]) / 2
    padded[0, :len(b), 2:] = b[:, 2:] - b[:, :2]
    x = np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), np.float32)
    return {'image': np.ascontiguousarray((x / 255).transpose(2, 0, 1)[None]),
            'boxes': padded, 'box_counts': np.array([len(b)], np.int64)}


def require_parity(name):
    if name == 'relsgg-vits16':
        parity = json.loads((HERE / 'parity_valid_relsgg-vits16.json').read_text())
        if parity.get('passed') is not True:
            raise RuntimeError('STOP M2: required two-photo real-pair parity failed')
        graph = MODELS / name / 'relateanything.onnx'
        if parity.get('graph_sha256') != hashlib.sha256(graph.read_bytes()).hexdigest():
            raise RuntimeError('STOP M2: graph SHA-256 differs from tested parity graph')


def mapped_thresholds(raw_thresholds, a, b):
    raw = np.asarray(raw_thresholds, np.float64)
    clipped = np.clip(np.where(np.isfinite(raw), raw, .5), 1e-6, 1 - 1e-6)
    return np.where(np.isfinite(raw), sigmoid(a * np.log(clipped / (1 - clipped)) + b), 0.4)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -np.asarray(x)))


class Head:
    def __init__(self, name, *, directory=None, size=448, threads=2, providers=None):
        if directory is None:
            require_parity(name)
        directory = directory or MODELS / name
        if size not in (448, 384, 336) or threads not in (2, 4):
            raise ValueError('unsupported size/thread count')
        self.size = size
        self.meta = json.loads((directory / 'relateanything.json').read_text())
        if (self.meta['img_size'], self.meta['max_boxes'], self.meta['vocab_mode'], self.meta['opset']) != (size, 32, 'input', 17):
            raise ValueError('unsupported sidecar contract')
        with np.load(directory / 'predicate_bank.npz', allow_pickle=True) as bank:
            names = list(map(str, bank['names']))
            self.predicates = list(map(str, bank['default']))
            assert len(self.predicates) == 35 and len(set(self.predicates)) == 35
            indices = [names.index(p) for p in self.predicates]
            self.W = np.ascontiguousarray(bank['W'][indices], np.float32)
            self.alpha = np.ascontiguousarray(bank['alpha'][indices], np.float32)
            raw_thresholds = np.asarray(bank['thr'][indices], np.float32)
        cal = self.meta['calibration']
        self.a, self.b = float(cal['a']), float(cal['b'])
        if not np.isfinite([self.a, self.b]).all() or self.a <= 0:
            raise ValueError('invalid calibration')
        self.thresholds = mapped_thresholds(raw_thresholds, self.a, self.b)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        start = time.monotonic()
        self.session = ort.InferenceSession(str(directory / 'relateanything.onnx'), options,
                                            providers=providers or ['CPUExecutionProvider'])
        self.load_ms = (time.monotonic() - start) * 1000
        self.metadata = {kind: [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in values]
                         for kind, values in [('inputs', self.session.get_inputs()), ('outputs', self.session.get_outputs())]}
        inputs = {v.name: v for v in self.session.get_inputs()}
        expected = {'image': ('tensor(float)', [1, 3, size, size]),
                    'boxes': ('tensor(float)', [1, 32, 4]), 'box_counts': ('tensor(int64)', [1]),
                    'W': ('tensor(float)', [35, 512]), 'alpha': ('tensor(float)', [35])}
        if set(inputs) != set(expected) or [v.name for v in self.session.get_outputs()] != OUTPUTS:
            raise ValueError('unexpected real graph names')
        for name, (dtype, shape) in expected.items():
            actual = inputs[name]
            if actual.type != dtype or len(actual.shape) != len(shape) or any(isinstance(x, int) and x != y for x, y in zip(actual.shape, shape)):
                raise ValueError(f'graph input mismatch: {name} {actual}')
        assert self.W.shape == (35, 512) and self.alpha.shape == (35,)

    def infer(self, image, detections):
        feed = preprocess(image, [d['box_xyxy'] for d in detections], size=getattr(self, 'size', 448))
        feed.update(W=self.W, alpha=self.alpha)
        start = time.monotonic()
        outputs = self.session.run(OUTPUTS, feed)
        milliseconds = (time.monotonic() - start) * 1000
        pred, pair, subject, obj, valid = [v[0] for v in outputs]
        if pred.shape != (128, 35) or any(v.shape != (128,) for v in (pair, subject, obj, valid)):
            raise ValueError('unexpected real output shapes')
        if not np.isfinite(pred).all() or not np.isfinite(pair).all():
            raise ValueError('nonfinite logits')
        scores = sigmoid(self.a * (pred + pair[:, None]) + self.b)
        best = scores.argmax(axis=1)
        keep = valid & (subject >= 0) & (obj >= 0) & (subject < len(detections)) & (obj < len(detections)) & (subject != obj)
        slots = np.flatnonzero(keep)
        slots = sorted(slots, key=lambda k: -float(scores[k, best[k]]))[:10]
        rows = []
        for k in slots:
            si, oi, pi = int(subject[k]), int(obj[k]), int(best[k])
            score, threshold = float(scores[k, pi]), float(self.thresholds[pi])
            rows.append({'subject_idx': si, 'subject': detections[si]['class_name'],
                         'predicate': self.predicates[pi], 'object_idx': oi, 'object': detections[oi]['class_name'],
                         'score': score, 'threshold': threshold, 'pass': score >= threshold})
        return rows, milliseconds


def peak_rss_kib():
    return int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmHWM:')))


def select_photos():
    sys.path.insert(0, str(ROBOT))
    import detect_person
    detector = detect_person.get_detector()
    selected, inventory = [], []
    paths = [p for directory in ('test_photos', 'bench_photos') for p in sorted((ROBOT / directory).glob('*.jpg'))]
    extras = [ROBOT / 'bench_photos' / n for n in ('blocked_1.jpg', 'roomsig_couch_1.jpg')]
    for p in paths:
        detections = detector.detect(open_image(p), 640)
        inventory.append({'photo': str(p.relative_to(ROBOT)), 'count': len(detections)})
        if len(detections) >= 2:
            selected.append({'photo': str(p.relative_to(ROBOT)), 'detections': detections})
        if len(selected) == 5:
            break
    if len(selected) < 5:
        raise RuntimeError('fewer than 5 qualifying photos before extras')
    for p in extras:
        if any(row['photo'] == str(p.relative_to(ROBOT)) for row in selected):
            continue
        detections = detector.detect(open_image(p), 640)
        inventory.append({'photo': str(p.relative_to(ROBOT)), 'count': len(detections)})
        if len(detections) >= 2:
            selected.append({'photo': str(p.relative_to(ROBOT)), 'detections': detections})
    write_json(HERE / 'selected_photos.json', {'selection_inventory': inventory, 'photos': selected})
    image = open_image(ROBOT / selected[0]['photo'])
    image.thumbnail((640, 640), Image.Resampling.BILINEAR)
    image.save(HERE / 'speed_photo.jpg', quality=95)
    image = open_image(HERE / 'speed_photo.jpg')  # boxes belong to the saved, decoded JPEG
    write_json(HERE / 'speed_input.json', {'source_photo': selected[0]['photo'], 'image': 'speed_photo.jpg',
                                         'size': list(image.size), 'detections': detector.detect(image, 640)})
    print(json.dumps({'selected': [row['photo'] for row in selected], 'speed_size': image.size}))


def run(name):
    output = HERE / 'results' / name
    output.mkdir(parents=True, exist_ok=True)
    head = Head(name)
    print(json.dumps(head.metadata, indent=2), flush=True)
    horse = open_image(EXTERNAL / 'assets/reel/images/horse.jpg')
    detections = [{'class_name': label, 'box_xyxy': box} for label, box in
                  [('person', [470, 130, 650, 630]), ('horse', [90, 310, 1010, 875])]]
    triplets, milliseconds = head.infer(horse, detections)
    sanity = any(r['subject_idx'] == 0 and r['object_idx'] == 1 and r['predicate'] == 'riding' for r in triplets[:5])
    write_json(output / 'sanity.json', {'passed': sanity, 'detections': detections, 'triplets': triplets,
                                       'ort_ms': milliseconds, 'timing_label': LABEL, 'graph': head.metadata})
    print('SANITY', name, sanity, json.dumps(triplets), flush=True)
    if not sanity:
        raise SystemExit('STOP model: horse sanity failed')
    timings, results = [], []
    for row in json.loads((HERE / 'selected_photos.json').read_text())['photos']:
        image = open_image(ROBOT / row['photo'])
        triplets, milliseconds = head.infer(image, row['detections'])
        timings.append(milliseconds)
        result = {**row, 'triplets': triplets, 'ort_ms': milliseconds, 'timing_label': LABEL}
        results.append(result)
        stem = row['photo'].replace('/', '__').removesuffix('.jpg')
        write_json(output / (stem + '.json'), result)
        draw = ImageDraw.Draw(image)
        for i, detection in enumerate(row['detections']):
            box = detection['box_xyxy']
            draw.rectangle(box, outline='red', width=max(2, image.width // 400))
            draw.text((box[0], box[1]), f"{i}: {detection['class_name']}", fill='red',
                      stroke_width=1, stroke_fill='white', font_size=max(14, image.width // 80))
        annotations = HERE / 'annotations'
        annotations.mkdir(exist_ok=True)
        image.save(annotations / (stem + '.jpg'), quality=90)
    summary = {'model': name, 'timing_label': LABEL, 'load_ms': head.load_ms,
               'median_ort_ms': float(np.median(timings)), 'VmHWM_KiB': peak_rss_kib(),
               'graph': head.metadata, 'sanity_pass': sanity, 'photos': results,
               'timing_scope': 'session construction load; ORT session.run only per image; excludes decode/preprocess/postprocess'}
    write_json(output / 'summary.json', summary)
    assert 'torch' not in sys.modules and 'cv2' not in sys.modules
    print(json.dumps({k: v for k, v in summary.items() if k not in ('graph', 'photos')}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--select', action='store_true')
    ap.add_argument('--model', choices=NAMES)
    args = ap.parse_args()
    if sys.platform != 'android':
        ap.error('native Termux Python required')
    if args.select:
        select_photos()
    elif args.model:
        run(args.model)
    else:
        ap.error('choose --select or --model')
