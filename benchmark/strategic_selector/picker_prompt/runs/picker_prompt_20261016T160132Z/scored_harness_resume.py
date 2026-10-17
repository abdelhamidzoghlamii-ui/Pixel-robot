"""PICKPROMPT-1 desk harness; no hardware imports, no runtime changes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import picker

SEED = 20261016
MISSIONS = ('explore and map the rooms', 'find a person and say hello',
            'map the rooms quietly and do not disturb people')
MODES = tuple(picker.MODES)
RULES = '''Robot rules (first matching rule wins):
If the mission is to map quietly and not disturb people and a person is visible, keep exploring.
If a person is visible, you have not greeted them in the last 300 seconds, and they are close or very close, stop and greet them.
If a person is visible, you have not greeted them in the last 300 seconds, and they are far or medium distance, go toward them.
If a person is visible and you have greeted them in the last 300 seconds, keep exploring.
If no person is visible, no new room has been found for at least 60 seconds (30 seconds for the mission to find a person and say hello), and no look-around has happened in the last 60 seconds, look around in place.
Otherwise, keep exploring.'''
DESCRIPTIONS = {'EXPLORE': 'keep exploring (default)',
                'APPROACH': 'go toward the person (seen, not greeted yet, still far)',
                'GREET': 'stop and greet the person (close, not greeted yet)',
                'LOOK': 'look around in place (nothing new for a while)'}


def label(s):
    """Return (mode, first matching rule); independent of any model output."""
    people = [p for p in s['scene'] if p[0] == 'person']
    recent = s['person_greeted_s_ago'] is not None and s['person_greeted_s_ago'] < 300
    if s['mission'] == MISSIONS[2] and people:
        return 'EXPLORE', 'R1'
    if people and not recent and any(p[2] in ('close', 'very close') for p in people):
        return 'GREET', 'R2'
    if people and not recent and any(p[2] in ('far', 'medium') for p in people):
        return 'APPROACH', 'R3'
    if people and recent:
        return 'EXPLORE', 'R4'
    threshold = 30 if s['mission'] == MISSIONS[1] else 60
    if not people and s['new_room_s_ago'] >= threshold and (s['last_look_s_ago'] is None or s['last_look_s_ago'] >= 60):
        return 'LOOK', 'R5'
    return 'EXPLORE', 'R6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def scene_lines(console):
    """Skip timing metadata; preserve the distinct owner's scene strings."""
    found = {}
    for line in Path(console).read_text().splitlines():
        if '[YOLO] ' not in line:
            continue
        text = line.split('[YOLO] ', 1)[1].strip()
        if text.startswith('size='):
            continue
        objects = []
        for part in text.split(', '):
            match = re.fullmatch(r'(.+?) (left|center|right)(?: \((very close|close|medium|far)\))?', part)
            if match:
                objects.append(list(match.groups()))
        if objects or text in ('nothing detected', 'none'):
            found[text] = objects
    if not found:
        raise ValueError('no usable scene lines in owner console')
    return sorted(found.items())


def generate(console, out):
    rng = random.Random(SEED)
    scenes = scene_lines(console)
    items = []
    # 30 per mode; EXPLORE is distributed over its three rules.
    for rule, count in [('R1', 10), ('R2', 30), ('R3', 30), ('R4', 10), ('R5', 30), ('R6', 10)]:
        for j in range(count):
            source, original = rng.choice(scenes)
            scene = [[name, pos, bucket or rng.choice(('far', 'medium', 'close', 'very close'))]
                     for name, pos, bucket in original if name != 'person']
            person = rule in ('R1', 'R2', 'R3', 'R4')
            if person:
                bucket = rng.choice(('close', 'very close') if rule == 'R2' else
                                    ('far', 'medium') if rule == 'R3' else ('far', 'medium', 'close', 'very close'))
                scene.append(['person', rng.choice(('left', 'center', 'right')), bucket])
            mission = MISSIONS[2] if rule == 'R1' else MISSIONS[j % (3 if rule in ('R5', 'R6') else 2)]
            greeted = rng.choice((0, 1, 299)) if rule == 'R4' else rng.choice((None, 300, 301, 900))
            new_room = rng.choice((60, 61, 120, 600)) if rule == 'R5' else rng.choice((0, 10, 29))
            last_look = rng.choice((None, 60, 61, 180))
            if len(scene) < 2:
                m2 = 'M2 relations:\n- none (fewer than 2 objects).'
            elif j % 3 == 0:
                m2 = f'M2 relations:\n- {scene[0][0]} [0] near {scene[1][0]} [1].'
            elif j % 3 == 1:
                m2 = f'M2 relations:\n- {scene[1][0]} [1] near {scene[0][0]} [0].'
            else:
                m2 = 'M2 relations:\n- none above threshold.'
            s = dict(id=f'S{len(items):03}', mission=mission, scene=scene,
                     ultrasonic_cm=rng.choice((15, 24, 25, 80, 150, 999)),
                     current_mode=rng.choice(MODES), mode_s=rng.choice((0, 5, 20, 65)),
                     last_moves=[rng.choice(('FORWARD', 'LEFT', 'RIGHT', 'HOLD')) for _ in range(3)],
                     rooms_seen=rng.choice(([], ['bedroom'], ['bedroom', 'living room'])), m2=m2,
                     person_greeted_s_ago=greeted, new_room_s_ago=new_room, last_look_s_ago=last_look,
                     provenance=dict(yolo_line=source, original_objects=original,
                                     buckets_synthetic=[i for i, p in enumerate(original) if p[2] is None],
                                     person_replaced_or_removed=True, m2_synthetic=True,
                                     other_fields_synthetic=True))
            s['label'], s['rule'] = label(s)
            assert s['rule'] == rule
            items.append(s)
    for mode in MODES:
        group = [s for s in items if s['label'] == mode]
        rng.shuffle(group)
        for j, s in enumerate(group):
            s['split'] = 'DEV' if j < 18 else 'TEST'
    rng.shuffle(items)
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'set.jsonl'
    path.write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in items))
    (out / 'set_sha256.txt').write_text(sha(path) + '  set.jsonl\n')
    write_json(out / 'freeze.json', dict(utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
               set_sha256=sha(path), console_sha256=sha(console), distinct_scene_lines=len(scenes),
               scene_lines=[x[0] for x in scenes], harness_sha256=sha(__file__),
               counts={key: dict(Counter(str(s[key]) for s in items)) for key in ('label', 'rule', 'mission', 'split')}))
    return items


def options(s, variant, reverse=False):
    keys = picker.eligible(any(p[0] == 'person' for p in s['scene']))
    if reverse:
        keys.reverse()
    phrases = DESCRIPTIONS if variant == 'V3' else picker.MODES
    return keys, [phrases[k] for k in keys]


def state(s, variant, content_free=False):
    text = picker.situation(s['mission'], [(n, 1.0, p, b) for n, p, b in s['scene']],
                            s['ultrasonic_cm'], s['current_mode'], s['mode_s'], s['last_moves'], s['rooms_seen'])
    text += '\n' + s['m2']
    if variant != 'V0':
        ago = lambda v: 'never' if v is None else f'{v} s ago'
        text += (f"\nPerson greeted: {ago(s['person_greeted_s_ago'])}"
                 f"\nNew room found: {ago(s['new_room_s_ago'])}"
                 f"\nLast look-around: {ago(s['last_look_s_ago'])}")
    if content_free:
        # Erase all situation values, including every M2 line; retain only field names.
        text = '\n'.join(line.split(':', 1)[0] + ': N/A' for line in text.splitlines() if ':' in line)
    if variant in ('V2', 'V3'):
        text += '\n' + RULES
    return text


def suffix(phrases):
    return ''.join([f'\nQuestion (choice): {picker.INSTRUCTION.strip()}', '\nOptions:',
                    *(f'\n({picker.LETTERS[j]}) {p.strip()}' for j, p in enumerate(phrases)), '\nAnswer: ('])


class StaticGemma(picker.Gemma):
    """Reuse production scoring; tokenize the fixed pieces only once at startup."""
    def __init__(self, post=picker.http):
        super().__init__(post)
        self.prefix = ([self.bos_token_id] if self.bos_token_id is not None else []) + self.encode('<state>\n')
        self.end = self.encode('\n</state>\n')
        self.suffixes = {}
        for descriptions in (picker.MODES, DESCRIPTIONS):
            for person in (False, True):
                for reverse in (False, True):
                    keys = picker.eligible(person)[:: -1 if reverse else 1]
                    phrases = tuple(descriptions[k] for k in keys)
                    self.suffixes[phrases] = self.encode(suffix(phrases))

    def prompt_ids(self, text, phrases):
        ids = self.encode(text)
        assert len(ids) <= picker.MAX_STATE_TOKENS, 'desk states must not truncate'
        return self.prefix + ids + self.end + self.suffixes[tuple(phrases)]


def metrics(rows):
    confusion = {a: {b: 0 for b in MODES} for a in MODES}
    for r in rows:
        confusion[r['label']][r['choice']] += 1
    recall, precision = {}, {}
    for m in MODES:
        total = sum(confusion[m].values())
        predicted = sum(confusion[a][m] for a in MODES)
        recall[m] = confusion[m][m] / total if total else 0
        precision[m] = confusion[m][m] / predicted if predicted else 0
    tokens = sorted(r['prompt_tokens'] for r in rows)
    return dict(n=len(rows), accuracy=sum(r['choice'] == r['label'] for r in rows) / len(rows),
                macro_recall=statistics.mean(recall.values()), recall=recall, precision=precision,
                confusion=confusion, r1_violations=sum(r['rule'] == 'R1' and r['choice'] in ('APPROACH', 'GREET') for r in rows),
                median_tokens=statistics.median(tokens), p90_tokens=tokens[math.ceil(.9 * len(tokens)) - 1],
                median_ms=statistics.median(r['ms'] for r in rows))


def best(names, scores):
    return min(names, key=lambda v: (-scores[v]['macro_recall'], scores[v]['median_tokens'], v))


def select(scores):
    qualified = [v for v, m in scores.items() if v != 'V5' and m['r1_violations'] == 0 and min(m['recall'].values()) >= .6]
    return best(qualified, scores) if qualified else None


def run(out, resume=False):
    path = out / 'set.jsonl'
    frozen = json.loads((out / 'freeze.json').read_text())
    assert sha(path) == frozen['set_sha256'] == (out / 'set_sha256.txt').read_text().split()[0]
    if resume:
        assert sha(out / 'scored_harness_initial.py') == frozen['harness_sha256']
        assert not (out / 'calibration.json').exists(), 'resume is limited to pre-calibration DEV'
        assert not list(out.glob('*_TEST.jsonl')), 'TEST cannot be resumed or repeated'
        previous = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
        assert not any(e['event'] == 'selection_frozen' for e in previous)
        (out / 'scored_harness_resume.py').write_bytes(Path(__file__).read_bytes())
    else:
        assert sha(__file__) == frozen['harness_sha256'], 'harness changed since set freeze'
        if (out / 'events.jsonl').exists():
            raise ValueError('run already started; use explicit pre-calibration DEV recovery')
        (out / 'scored_harness_initial.py').write_bytes(Path(__file__).read_bytes())
    items = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(items) == 120 and Counter(s['split'] for s in items) == {'DEV': 72, 'TEST': 48}
    assert all((s['label'], s['rule']) == label(s) for s in items)
    events = (out / 'events.jsonl').open('a' if resume else 'x')

    def event(kind, **fields):
        events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=kind, **fields)) + '\n')
        events.flush()

    # The desk prompts can exceed production's 10 s timeout; scoring parameters stay unchanged.
    picker.TIMEOUT_S = 120
    event('resume_before_gemma_call' if resume else 'before_first_gemma_call',
          set_sha256=sha(path), harness_sha256=sha(__file__), http_timeout_s=picker.TIMEOUT_S)
    gemma = StaticGemma()
    scores, rows_by_variant, calibrations = {}, {}, {}
    speed = json.loads((out / 'speed.json').read_text()) if resume and (out / 'speed.json').exists() else []
    base = None

    def score(variant, split, prompt_variant, reverse=False):
        result_path = out / f'{variant}_{split}.jsonl'
        rows = [json.loads(line) for line in result_path.read_text().splitlines()] if resume and result_path.exists() else []
        saved = {r['id']: r for r in rows}
        assert len(saved) == len(rows)
        with result_path.open('a' if resume else 'x') as output:
            for s in items:
                if s['split'] != split:
                    continue
                if s['id'] in saved and (variant != 'V0' or any(r['id'] == s['id'] for r in speed)):
                    continue
                keys, phrases = options(s, prompt_variant, reverse)
                text = state(s, prompt_variant)
                # Four tokenize requests versus one: same assembled ids on every DEV item.
                if split == 'DEV':
                    assert picker.Gemma.prompt_ids(gemma, text, phrases) == gemma.prompt_ids(text, phrases)
                if variant == 'V0':
                    # Use production decide with production prompt assembly for the baseline.
                    original = gemma.prompt_ids
                    gemma.prompt_ids = lambda st, ph: picker.Gemma.prompt_ids(gemma, st, ph)
                    try:
                        # Start after the parity assertion; avoid counting an extra tokenization.
                        t = time.perf_counter()
                        dist = gemma.decide(text, phrases)
                        ms = (time.perf_counter() - t) * 1000
                    finally:
                        gemma.prompt_ids = original
                    t = time.perf_counter()
                    fast = gemma.decide(text, phrases)
                    fast_ms = (time.perf_counter() - t) * 1000
                    assert max(abs(dist[p] - fast[p]) for p in phrases) < 1e-6
                    speed.append(dict(id=s['id'], standard_ms=ms, static_ms=fast_ms,
                                      standard_first=True, token_parity=True))
                    write_json(out / 'speed.json', speed)
                else:
                    t = time.perf_counter()
                    dist = gemma.decide(text, phrases)
                    ms = (time.perf_counter() - t) * 1000
                raw = dict(zip(keys, (dist[p] for p in phrases)))
                probs = raw
                if variant in ('V4', 'V5'):
                    baseline = calibrations[variant][','.join(keys)]['probabilities']
                    adjusted = {k: raw[k] / baseline[k] for k in keys}
                    mass = sum(adjusted.values())
                    probs = {k: adjusted[k] / mass for k in keys}
                row = dict(id=s['id'], variant=variant, prompt_variant=prompt_variant, split=split,
                           label=s['label'], rule=s['rule'], choice=max(keys, key=probs.get),
                           eligible=keys, raw_probabilities=raw, probabilities=probs,
                           prompt_tokens=gemma.prompt_tokens, ms=ms)
                if s['id'] in saved:  # Recover lost speed timings without replacing any saved DEV decision.
                    assert max(abs(raw[k] - saved[s['id']]['raw_probabilities'][k]) for k in keys) < 1e-6
                else:
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    rows.append(row)
        event('scored', variant=variant, split=split, n=len(rows), set_sha256=sha(path))
        print(variant, split, json.dumps(metrics(rows)), flush=True)
        return rows

    for v in ('V0', 'V1', 'V2', 'V3'):
        rows_by_variant[v] = score(v, 'DEV', v)
        scores[v] = metrics(rows_by_variant[v])
    base = best(('V1', 'V2', 'V3'), scores)
    event('calibration_base_selected', base=base, criterion='macro recall, median tokens, variant id')
    for variant in ('V4', 'V5'):
        calibrations[variant] = {}
        for person in (False, True):
            s = next(s for s in items if s['split'] == 'DEV' and any(p[0] == 'person' for p in s['scene']) == person)
            keys, phrases = options(s, base, variant == 'V5')
            text = state(s, base, content_free=True)
            dist = gemma.decide(text, phrases)
            probabilities = dict(zip(keys, (dist[p] for p in phrases)))
            assert all(p > 0 for p in probabilities.values())
            calibrations[variant][','.join(keys)] = dict(state=text, phrases=phrases, probabilities=probabilities,
                                                       prompt_tokens=gemma.prompt_tokens)
        write_json(out / 'calibration.json', calibrations)
        rows_by_variant[variant] = score(variant, 'DEV', base, variant == 'V5')
        scores[variant] = metrics(rows_by_variant[variant])
    selected = select(scores)
    changed = sum(a['choice'] != b['choice'] for a, b in zip(rows_by_variant['V4'], rows_by_variant['V5']))
    summary = dict(dev=scores, calibration_base=base, selected=selected, v5_changed=changed,
                   timing_validity='NOT A MEASUREMENT: agent resident; standard always precedes static',
                   speed=dict(standard_median_ms=statistics.median(r['standard_ms'] for r in speed),
                              static_median_ms=statistics.median(r['static_ms'] for r in speed),
                              parity_dev_items=len(speed)), test=None)
    write_json(out / 'speed.json', speed)
    event('selection_frozen', selected=selected, dev=scores)
    if selected:
        prompt_variant = base if selected == 'V4' else selected
        summary['test'] = metrics(score(selected, 'TEST', prompt_variant))
        examples = []
        for person in (True, False):
            s = next(s for s in items if s['split'] == 'DEV' and any(p[0] == 'person' for p in s['scene']) == person)
            keys, phrases = options(s, prompt_variant)
            examples.append(dict(id=s['id'], person=person, mode_order=keys,
                                 text='<bos><state>\n' + state(s, prompt_variant) + '\n</state>\n' + suffix(phrases)))
        write_json(out / 'selected_prompts.json', examples)
    else:
        summary['outcome'] = 'NO VARIANT QUALIFIES'
        summary['best_macro_recall'] = max(m['macro_recall'] for v, m in scores.items() if v != 'V5')
    write_json(out / 'summary.json', summary)
    (out / 'summary.md').write_text('# PICKPROMPT-1\n\nTimings are NOT measurements (agent resident).\n\n'
                                   + ('Selected: ' + selected if selected else 'NO VARIANT QUALIFIES')
                                   + '\n\n```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    assert sha(path) == frozen['set_sha256']
    event('finished', selected=selected, set_sha256=sha(path))
    events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'run'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume-dev', action='store_true', help='recover an interrupted pre-calibration DEV run; never TEST')
    parser.add_argument('--console', type=Path, default=Path('/termux-home/robot_runs/picker1_dry_console.log'))
    args = parser.parse_args()
    if args.command == 'generate':
        generate(args.console, args.output)
    else:
        run(args.output, args.resume_dev)


if __name__ == '__main__':
    main()
