"""Fixed owner quality criterion; native Termux, motors off, informal speeds only."""
import json
import sys
import traceback
from variants import HERE, VARIANTS, NAME, dc, sb, sha, identity, make_head, build, reference_hashes
import parity_valid as pv


def triplet_map(rows, detections):
    def key(r):
        return (tuple(detections[r['subject_idx']]['box_xyxy']), r['predicate'],
                tuple(detections[r['object_idx']]['box_xyxy']))
    return {key(r): r for r in rows}


def compare_triplets(reference, got, detections):
    a, b = triplet_map(reference, detections), triplet_map(got, detections)
    common = a.keys() & b.keys()
    delta = max((abs(a[k]['score'] - b[k]['score']) for k in common), default=None)
    flips = sum(a[k]['pass'] != b[k]['pass'] for k in common)
    return {'overlap': len(common), 'reference_count': len(a), 'variant_count': len(b),
            'max_score_delta': delta, 'pass_fail_flips': flips,
            'passed': len(common) >= 9 and delta is not None and delta <= .05}


def parity(head):
    rows = []
    for photo in json.loads((pv.DATA / 'photos.json').read_text()):
        key = photo['key']
        feed = pv.load_feed(key, NAME)
        ref = pv.load(pv.DATA / f'{key}_{NAME}_onnx.npz')
        try:
            got = dict(zip(dc.OUTPUTS, head.session.run(dc.OUTPUTS, feed)))
        except Exception as error:
            rows.append({'photo': key, 'passed': False, 'error': repr(error)})
            continue
        a = pv.real_slots(ref, int(feed['box_counts'][0]))
        b = pv.real_slots(got, int(feed['box_counts'][0]))
        deltas = [(float(dc.np.max(dc.np.abs(ref['pred_logits'][0, a[k]] - got['pred_logits'][0, b[k]]))),
                   float(abs(ref['pair_logits'][0, a[k]] - got['pair_logits'][0, b[k]]))) for k in a.keys() & b.keys()]
        pred = max((d[0] for d in deltas), default=None)
        pair = max((d[1] for d in deltas), default=None)
        passed = bool(deltas) and set(a) == set(b) and pred <= 1e-3 and pair <= 1e-3
        rows.append({'photo': key, 'same_real_pair_set': set(a) == set(b),
                     'max_pred_delta': pred, 'max_pair_delta': pair, 'passed': passed})
    return rows


def main():
    assert sys.platform == 'android'
    print(dc.LABEL, flush=True)
    dc.require_parity(NAME)
    builds = json.loads((HERE / 'builds.json').read_text())
    selected = json.loads((dc.HERE / 'selected_photos.json').read_text())['photos']
    assert len(selected) == 7
    result = {'criterion': 'horse top1 person-riding-horse; EACH 7 photos overlap >=9/10 by identical subject box, predicate, object box; every common score delta <=0.05. XNNPACK: same real pair set and pred/pair logits deltas <=1e-3 on horse and blocked_1.',
              'timing_label': dc.LABEL, 'providers_available': dc.ort.get_available_providers(),
              'ort_version': dc.ort.__version__, 'optimization_level': str(dc.ort.SessionOptions().graph_optimization_level),
              'reference_identity': identity('fp32'), 'reference_hashes': reference_hashes(), 'variants': {}}
    for variant in VARIANTS:
        entry = {'passed': False, 'photos': []}
        result['variants'][variant] = entry
        if variant not in ('fp32', 'xnnpack') and builds['variants'][variant]['status'] != 'AVAILABLE':
            entry['status'] = builds['variants'][variant]['status']
            continue
        if variant == 'xnnpack' and 'XnnpackExecutionProvider' not in dc.ort.get_available_providers():
            entry['status'] = 'NOT AVAILABLE'
            continue
        caller = head = None
        try:
            entry.update(identity=identity(variant), status='AVAILABLE')
            head, caller, tids = build(variant, 'MID')
            entry.update(providers=head.session.get_providers(), worker_caller_tids=tids,
                         cpus=[4, 5], threads=2, optimization_level=str(head.session.get_session_options().graph_optimization_level))
            if variant == 'xnnpack':
                entry['parity'] = parity(head)
            horse_dets = [{'class_name': n, 'box_xyxy': b} for n, b in
                          [('person', [470, 130, 650, 630]), ('horse', [90, 310, 1010, 875])]]
            horse, _ = caller.detect(dc.open_image(dc.EXTERNAL / 'assets/reel/images/horse.jpg'), horse_dets)
            entry['horse_top1'] = horse[:1]
            entry['horse_pass'] = bool(horse) and (horse[0]['subject'], horse[0]['predicate'], horse[0]['object']) == ('person', 'riding', 'horse')
            for row in selected:
                sb.check_pinning(tids)
                got, ms = caller.detect(dc.open_image(dc.ROBOT / row['photo']), row['detections'])
                sb.check_pinning(tids)
                filename = row['photo'].replace('/', '__').removesuffix('.jpg') + '.json'
                ref = json.loads((dc.HERE / 'results' / NAME / filename).read_text())
                assert ref['detections'] == row['detections']
                compared = compare_triplets(ref['triplets'], got, row['detections'])
                entry['photos'].append({'photo': row['photo'], **compared, 'triplets': got, 'ort_ms': ms})
                print(variant, row['photo'], {k: v for k, v in compared.items()}, dc.LABEL, flush=True)
            if variant == 'xnnpack':
                entry['passed'] = all(r['passed'] for r in entry['parity'])
            elif variant == 'fp32':
                # The reference is not subject to the changed-numbers 9/10 rule.
                entry['reference_reproduced'] = all(r['overlap'] == r['reference_count'] and
                    r['max_score_delta'] == 0 for r in entry['photos'])
                entry['passed'] = entry['reference_reproduced'] and entry['horse_pass']
            else:
                entry['passed'] = entry['horse_pass'] and all(r['passed'] for r in entry['photos'])
            # Same fixed speed photo, warmup + 3 calls, MID verified, only eligible variants.
            if entry['passed'] or variant == 'fp32':
                datum = json.loads((dc.HERE / 'speed_input.json').read_text())
                image = dc.open_image(dc.HERE / datum['image'])
                samples = []
                for i in range(4):
                    sb.check_pinning(tids)
                    _, ms = caller.detect(image, datum['detections'])
                    sb.check_pinning(tids)
                    if i: samples.append(ms)
                entry['informal_mid_ms'] = samples
                entry['informal_mid_median_ms'] = float(dc.np.median(samples))
        except Exception as error:
            entry.update(passed=False, error=repr(error))
            traceback.print_exc()
        finally:
            if caller: caller.close()
            head = caller = None
        dc.write_json(HERE / 'quality.json', result)
        print(variant, 'PASS' if entry['passed'] else 'FAIL', dc.LABEL, flush=True)
    dc.write_json(HERE / 'quality.json', result)


if __name__ == '__main__':
    main()
