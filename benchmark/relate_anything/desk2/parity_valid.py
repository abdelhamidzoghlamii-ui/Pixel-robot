"""DESK3 fixed real-pair parity. prepare/compare: native Termux; torch: Debian venv."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'parity_desk3'
NAMES = ('relsgg-vits16plus', 'relsgg-vits16')
OUTPUTS = ('pred_logits', 'pair_logits', 'sub_idx', 'obj_idx', 'valid_mask')
INPUTS = ('image', 'boxes', 'box_counts', 'W', 'alpha')
LABEL = 'INFORMAL — agents resident, NOT VALID TIMING'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {n: z[n] for n in z.files}


def load_feed(key, name):
    parts = json.loads((DATA / 'feed_components.json').read_text())[f'{key}_{name}']
    feed = {}
    for kind in ('base', 'vocabulary'):
        part = parts[kind]
        path = HERE / part['path']
        if sha(path) != part['sha256']:
            raise RuntimeError('parity feed component SHA-256 mismatch: ' + str(path))
        feed.update(load(path))
    return feed


def prepare():
    import desk_check as dc
    assert sys.platform == 'android'
    DATA.mkdir(exist_ok=True)
    components = {}
    vocab_directory = dc.MODELS / 'parity_fix1'
    vocab_directory.mkdir(exist_ok=True)
    photos = [{'photo': str(dc.EXTERNAL / 'assets/reel/images/horse.jpg'), 'key': 'horse',
               'detections': [{'class_name': n, 'box_xyxy': b} for n, b in
                              [('person', [470,130,650,630]), ('horse', [90,310,1010,875])]]},
              next(r for r in json.loads((HERE / 'selected_photos.json').read_text())['photos']
                   if r['photo'] == 'bench_photos/blocked_1.jpg') | {'key': 'blocked_1'}]
    for row in photos:
        p = dc.ROBOT / row['photo']
        feed = dc.preprocess(dc.open_image(p), [d['box_xyxy'] for d in row['detections']])
        path = DATA / (row['key'] + '_image_boxes.npz')
        np.savez(path, **feed)
        row.update(photo_sha256=sha(p), base_feed_sha256=sha(path))
        for name in NAMES:
            directory = dc.MODELS / name
            with np.load(directory / 'predicate_bank.npz', allow_pickle=True) as z:
                names = list(map(str, z['names']))
                default = list(map(str, z['default']))
                idx = [names.index(n) for n in default]
                vocab = {n: np.ascontiguousarray(z[n][idx], np.float32) for n in ('W','alpha')}
            vocab_path = vocab_directory / f'{name}_vocab.npz'
            if not vocab_path.exists():
                np.savez(vocab_path, **vocab)
            saved = load(vocab_path)
            if any(not np.array_equal(saved[n], vocab[n]) for n in vocab):
                raise RuntimeError('existing external parity vocabulary differs from bank')
            components[f'{row["key"]}_{name}'] = {
                'base': {'path': str(path.relative_to(HERE)), 'sha256': sha(path)},
                'vocabulary': {'path': str(vocab_path), 'sha256': sha(vocab_path)}}
    (DATA / 'photos.json').write_text(json.dumps(photos, indent=2)+'\n')
    (DATA / 'feed_components.json').write_text(json.dumps(components, indent=2)+'\n')
    print('Saved one image/boxes feed per photo and one external W/alpha archive per model.')


def torch_run(name):
    assert sys.platform != 'android', 'torch only in Debian venv'
    import torch
    sys.path.insert(0, '/termux-home/ext/RelateAnything')
    from deploy.export_onnx import RelSGGExport
    from relsgg.api import RelateAnything
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    directory = Path('/termux-home/models/relate_anything') / name
    with np.load(directory / 'predicate_bank.npz', allow_pickle=True) as z:
        names = list(map(str, z['default']))
    feed = load_feed('horse', name)
    ra = RelateAnything.from_checkpoint(str(directory / 'model.pth'), names, device='cpu',
                                       weights='ema', img_size=448, strict=True, embeddings=feed['W'])
    wrapper = RelSGGExport(ra.model.eval(), vocab_as_input=True).eval()
    for row in json.loads((DATA / 'photos.json').read_text()):
        feed = load_feed(row['key'], name)
        # The wrapper assigns these exact W/alpha tensors to the vocabulary head.
        with torch.no_grad():
            result = wrapper(*(torch.from_numpy(feed[n]) for n in INPUTS))
        np.savez(DATA / f'{row["key"]}_{name}_torch.npz',
                 **dict(zip(OUTPUTS, [v.numpy() for v in result])))
        print(name, row['key'], 'PyTorch saved; hashed feed components', LABEL, flush=True)
    (DATA / f'{name}_torch_versions.json').write_text(json.dumps({'torch':torch.__version__,
                       'python':sys.version, 'weights':'ema', 'vocabulary':names}, indent=2)+'\n')


def real_slots(out, count):
    s, o, v = [out[n][0] for n in OUTPUTS[2:]]
    slots = [k for k in range(len(v)) if v[k] and 0 <= s[k] < count and 0 <= o[k] < count and s[k] != o[k]]
    pairs = {(int(s[k]), int(o[k])): int(k) for k in slots}
    assert len(pairs) == len(slots), 'duplicate real pair slots need explicit comparison'
    return pairs


def compare(name):
    import desk_check as dc
    assert sys.platform == 'android'
    import onnxruntime as ort
    assert ort.__version__ == '1.25.1'
    directory = dc.MODELS / name
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / 'relateanything.onnx'), options,
                                  providers=['CPUExecutionProvider'])
    head = dc.Head.__new__(dc.Head)
    meta = json.loads((directory / 'relateanything.json').read_text())
    head.a, head.b = meta['calibration']['a'], meta['calibration']['b']
    with np.load(directory / 'predicate_bank.npz', allow_pickle=True) as z:
        names = list(map(str,z['names'])); head.predicates = list(map(str,z['default']))
        raw = np.asarray(z['thr'][[names.index(n) for n in head.predicates]], np.float32)
    clipped = np.clip(raw,1e-6,1-1e-6)
    # Historical desk2 decoding, including float32 threshold mapping, for diagnostics.
    head.thresholds = np.where(np.isfinite(raw), dc.sigmoid(head.a*np.log(clipped/(1-clipped))+head.b), .4)
    photos = []
    for row in json.loads((DATA / 'photos.json').read_text()):
        key = row['key']; feed = load_feed(key, name)
        head.W, head.alpha = feed['W'], feed['alpha']
        ref = load(DATA / f'{key}_{name}_torch.npz')
        got = dict(zip(OUTPUTS, session.run(list(OUTPUTS),feed)))
        np.savez(DATA / f'{key}_{name}_onnx.npz', **got)
        count = int(feed['box_counts'][0]); a = real_slots(ref,count); b = real_slots(got,count)
        matched = []
        for pair in sorted(a.keys() & b.keys()):
            i,j = a[pair],b[pair]
            pred = float(np.max(np.abs(ref['pred_logits'][0,i]-got['pred_logits'][0,j])))
            rel = float(abs(ref['pair_logits'][0,i]-got['pair_logits'][0,j]))
            matched.append({'pair':pair,'torch_slot':i,'onnx_slot':j,'pred_max_abs':pred,'pair_abs':rel})
        mismatches = []
        for k in range(128):
            fields = [n for n in OUTPUTS if not np.array_equal(ref[n][0,k],got[n][0,k])]
            if fields:
                mismatches.append({'slot':k,'fields':fields,'torch_pair':[int(ref[n][0,k]) for n in OUTPUTS[2:4]],
                    'onnx_pair':[int(got[n][0,k]) for n in OUTPUTS[2:4]],
                    'torch_real':k in a.values(),'onnx_real':k in b.values(),
                    'pred_max_abs':float(np.max(np.abs(ref['pred_logits'][0,k]-got['pred_logits'][0,k]))),
                    'pair_abs':float(abs(ref['pair_logits'][0,k]-got['pair_logits'][0,k]))})
        decoded=[]
        for out in (ref,got):
            class Saved:
                def run(self,names,inputs): return [out[n] for n in names]
            head.session=Saved()
            decoded.append(head.infer(dc.open_image(dc.ROBOT/row['photo']), row['detections'])[0])
        semantic = lambda rows: [{k:v for k,v in r.items() if k not in ('score','threshold')} for r in rows]
        same = set(a)==set(b)
        passed = same and bool(matched) and all(np.isfinite(r['pred_max_abs']) and np.isfinite(r['pair_abs'])
                        and r['pred_max_abs']<=1e-3 and r['pair_abs']<=1e-3 for r in matched)
        photos.append({**row,'feed_components':json.loads((DATA/'feed_components.json').read_text())[f'{key}_{name}'],
             'real_pair_sets_identical':same,
             'torch_real_pairs':sorted(a),'onnx_real_pairs':sorted(b),'matched_real_pairs':matched,
             'max_real_pred_delta':max((r['pred_max_abs'] for r in matched),default=None),
             'max_real_pair_delta':max((r['pair_abs'] for r in matched),default=None),
             'all_slot_deltas':{n:float(np.max(np.abs(ref[n]-got[n]))) for n in OUTPUTS[:2]},
             'all_slot_exact':{n:bool(np.array_equal(ref[n],got[n])) for n in OUTPUTS[2:]},
             'mismatched_slots':mismatches,'top10_identical':semantic(decoded[0])==semantic(decoded[1]),
             'top10_full_records_bitwise_identical':decoded[0]==decoded[1],
             'top10_torch':decoded[0],'top10_onnx':decoded[1],'passed':passed})
        print(name,key,'PASS' if passed else 'FAIL','real max',photos[-1]['max_real_pred_delta'],
              photos[-1]['max_real_pair_delta'],'top10',photos[-1]['top10_identical'],LABEL,flush=True)
    result={'model':name,'criterion':'same real (sub,obj) set; every matched real pair pred max abs and pair abs <= 1e-3',
        'real_definition':'valid_mask True, sub != obj, both indices in range','threshold_abs_logits':1e-3,
        'top10_identity_definition':'ordered decoded triplets and threshold flags identical; scores reported separately',
        'passed':all(r['passed'] for r in photos),'ort_version':ort.__version__,'providers':session.get_providers(),
        'graph_sha256':sha(directory/'relateanything.onnx'),'checkpoint_sha256':sha(directory/'model.pth'),
        'vocabulary_delivery':'bank default names in order; same indexed W/alpha float32 rows in saved feed; embeddings=W then RelSGGExport(vocab_as_input=True) assigns exact W/alpha',
        'timing_label':LABEL,'photos':photos}
    (HERE/f'parity_valid_{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('prepare','torch','compare')); ap.add_argument('--model',choices=NAMES)
    args=ap.parse_args()
    if args.mode=='prepare': prepare()
    elif args.model: (torch_run if args.mode=='torch' else compare)(args.model)
    else: ap.error('--model required')
