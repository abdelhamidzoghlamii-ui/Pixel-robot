"""RelateAnything M2 for the picker (DECISIONS #130; layout L2 from #131: BIG cpus 6-7, 2 threads, run once right
before each picker call on the latest 640 frame and its boxes).

M2 is our fp32 ONNX export of relsgg-vits16. The weights, graph, vocabulary bank and the AGPL upstream source stay
outside this repository (#130); only their paths and SHA-256s are here. The weight licence for shipped use is not
evaluated. Code marked "copied from" is the campaign code, copied and trimmed (benchmark/ is not deployed).
"""
import hashlib
import json
import os
import threading
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from detect_person import PinnedDetector, pin, thread_ids

DIRECTORY = '/data/data/com.termux/files/home/models/relate_anything/relsgg-vits16'
# The files the campaign ran (benchmark/campaign/expected_hashes.json). The graph hash is the one that passed the
# real-pair parity check against PyTorch (benchmark/relate_anything/desk2/parity_valid_relsgg-vits16.json, #130):
# a different graph is refused, as desk_check.require_parity() does.
HASHES = {'relateanything.onnx': '24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5',
          'relateanything.json': 'e70152faa7875bf9ae865d14b3539b7aac358f33bfb10db265365441db26bfaa',
          'predicate_bank.npz': '6043ff8fb765db34d89019617979d566b79967b7740c72bb4007786179839f48'}
CPUS = {6, 7}     # BIG (#131 L2)
THREADS = 2       # intra-op; ORT counts the calling thread as one of them
MAX_BOXES = 32    # the head takes 2..32 boxes
OUTPUTS = ['pred_logits', 'pair_logits', 'sub_idx', 'obj_idx', 'valid_mask']
TOO_FEW = 'M2 relations:\n- none (fewer than 2 objects).'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# Copied from benchmark/relate_anything/desk2/desk_check.py: preprocess().
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


# Copied from benchmark/relate_anything/desk2/desk_check.py: mapped_thresholds(), sigmoid().
def mapped_thresholds(raw_thresholds, a, b):
    raw = np.asarray(raw_thresholds, np.float64)
    clipped = np.clip(np.where(np.isfinite(raw), raw, .5), 1e-6, 1 - 1e-6)
    return np.where(np.isfinite(raw), sigmoid(a * np.log(clipped / (1 - clipped)) + b), 0.4)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -np.asarray(x)))


class Head:
    """Copied from benchmark/relate_anything/desk2/desk_check.py: Head, trimmed to fp32 / 448 px / 2 threads / CPU.
    Changed: the SHA-256 of all three files is checked first (the parity gate, and before np.load(allow_pickle=True)
    reads the bank); infer() returns the rows only (Relate.relations() times the whole call)."""

    def __init__(self, directory=DIRECTORY):
        for name, digest in HASHES.items():
            if sha256(os.path.join(directory, name)) != digest:
                raise RuntimeError(f'STOP M2: {name} SHA-256 differs from the parity-checked files')
        self.size = 448
        with open(os.path.join(directory, 'relateanything.json')) as f:
            self.meta = json.load(f)
        if (self.meta['img_size'], self.meta['max_boxes'], self.meta['vocab_mode'], self.meta['opset']) != (448, 32, 'input', 17):
            raise ValueError('unsupported sidecar contract')
        # allow_pickle as in desk_check; safe only because the bank's SHA-256 was checked just above.
        with np.load(os.path.join(directory, 'predicate_bank.npz'), allow_pickle=True) as bank:
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
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(os.path.join(directory, 'relateanything.onnx'), options,
                                            providers=['CPUExecutionProvider'])
        inputs = {v.name: v for v in self.session.get_inputs()}
        expected = {'image': ('tensor(float)', [1, 3, 448, 448]),
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
        feed = preprocess(image, [d['box_xyxy'] for d in detections], size=self.size)
        feed.update(W=self.W, alpha=self.alpha)
        outputs = self.session.run(OUTPUTS, feed)
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
        return rows


# Copied from benchmark/campaign/phase1.py: context() (the picker text of the campaign's L2 layout).
def chosen(rows):
    return sorted((r for r in rows if r.get('pass') is True and r['score'] >= r['threshold']),
                  key=lambda r: (-r['score'], r['subject_idx'], r['object_idx'], r['predicate']))[:5]


def context(rows):
    def clean(s):
        return ' '.join(str(s).replace('_', ' ').split())
    return 'M2 relations:\n' + ('\n'.join(f'- {clean(r["subject"])} [{r["subject_idx"]}] {clean(r["predicate"])} {clean(r["object"])} [{r["object_idx"]}].'
                                   for r in chosen(rows)) or '- none above threshold.')


class Relate:
    """The M2 head with its ORT worker pinned to CPUS; the calling thread (the picker worker) pins itself with
    pin_caller(). Only these 2 threads are ever pinned: never the process, pid 0 or the main thread.
    Copied from benchmark/relate_anything/desk2/speed_block.py: build() (worker diff and pinning); raises on any
    failure. affinity()/check_affinity() are detect_person.PinnedDetector's (same caller_tid/worker_tids/cpus)."""
    affinity = PinnedDetector.affinity
    check_affinity = PinnedDetector.check_affinity

    def __init__(self, cpus=CPUS, head=Head):
        self.cpus = set(cpus)
        before = thread_ids()  # the diff sees only the ORT worker: no other code starts threads meanwhile
        self.head = head()
        self.worker_tids = sorted(thread_ids() - before)
        if len(self.worker_tids) != THREADS - 1:
            raise RuntimeError(f'cannot identify the M2 ORT worker: expected {THREADS - 1}, got {self.worker_tids}')
        for tid in self.worker_tids:
            pin(tid, self.cpus)
        self.caller_tid = None

    def pin_caller(self):
        """Call from the thread that will run infer() (never the main thread: pin() refuses it)."""
        tid = threading.get_native_id()
        pin(tid, self.cpus)
        self.caller_tid = tid

    def relations(self, image, detections):
        """(picker text, boxes used, relations in the text, inference ms or None). Fewer than 2 boxes: no M2 run.
        More than 32: the 32 highest-confidence boxes, in detector order (copied from phase1.run_cycle's m2())."""
        if len(detections) < 2:
            return TOO_FEW, len(detections), 0, None
        if len(detections) > MAX_BOXES:
            keep = sorted(sorted(range(len(detections)), key=lambda i: -detections[i]['conf'])[:MAX_BOXES])
            detections = [detections[i] for i in keep]
        started = time.perf_counter()
        rows = self.head.infer(image, detections)
        ms = (time.perf_counter() - started) * 1000
        return context(rows), len(detections), len(chosen(rows)), ms
