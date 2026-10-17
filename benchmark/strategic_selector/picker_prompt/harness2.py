"""PICKPROMPT-2: frozen desk accuracy test; no hardware, root or production changes."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
import urllib.request

import harness as h
import server_manager as sm

SOURCE = Path('/termux-home/robot_runs/picker_prompt_20261016T160132Z')
SET_SHA = '19773ea00d376d67e78641844241b6a8a310c0c8a7b1a048113f9c435a76ae44'
FIELDS = ('person_visible', 'person_close', 'greeted_in_last_5_min', 'mission_allows_people',
          'nothing_new_for_a_while', 'action')
CANDIDATES = ('A0', 'A2', 'A4', 'B1-E2B', 'B2-E2B', 'B1-E4B', 'B2-E4B')
VARIANTS = CANDIDATES + ('A5', 'B3-E2B', 'B3-E4B')
INSTRUCTION = 'First fill in the facts, then choose the action. Answer only with JSON.'


def truth(s):
    people = [p for p in s['scene'] if p[0] == 'person']
    return dict(person_visible=bool(people), person_close=any(p[2] in ('close', 'very close') for p in people),
                greeted_in_last_5_min=s['person_greeted_s_ago'] is not None and s['person_greeted_s_ago'] < 300,
                mission_allows_people=s['mission'] != h.MISSIONS[2],
                nothing_new_for_a_while=h.label(s)[1] == 'R5')


def action_from_facts(facts):
    """B3: translate only Gemma's facts into label()'s inputs, without the original item's truths."""
    return h.label(dict(mission=h.MISSIONS[0] if facts['mission_allows_people'] else h.MISSIONS[2],
                        scene=[['person', 'center', 'close' if facts['person_close'] else 'far']]
                              if facts['person_visible'] else [],
                        person_greeted_s_ago=0 if facts['greeted_in_last_5_min'] else None,
                        new_room_s_ago=60 if facts['nothing_new_for_a_while'] else 0,
                        last_look_s_ago=None))[0]


def grammar(person):
    """GBNF fixes all six keys, their order and types, and the eligible action enum."""
    parts = [json.dumps('{'), 'ws']
    for i, key in enumerate(FIELDS):
        if i:
            parts += [json.dumps(','), 'ws']
        parts += [json.dumps(json.dumps(key)), 'ws', json.dumps(':'), 'ws',
                  'action' if key == 'action' else 'boolean', 'ws']
    parts += [json.dumps('}'), 'ws']
    actions = ' | '.join(json.dumps(json.dumps(k)) for k in h.picker.eligible(person))
    return ('root ::= ' + ' '.join(parts) + '\nboolean ::= "true" | "false"\n'
            'ws ::= [ \\t\\n\\r]*\naction ::= ' + actions + '\n')


def json_payload(s, variant):
    state = h.state(s, 'V2' if variant == 'B2' else 'V1')
    prompt = '<start_of_turn>user\n' + state + '\n' + INSTRUCTION + '<end_of_turn>\n<start_of_turn>model\n'
    return dict(prompt=prompt, grammar=grammar(truth(s)['person_visible']), temperature=0,
                n_predict=96, cache_prompt=False, stream=False, stop=['<end_of_turn>'])


def parse_json(raw, person):
    def object_pairs(pairs):
        if len({k for k, v in pairs}) != len(pairs):
            raise ValueError('duplicate key')
        return dict(pairs)
    try:
        parsed = json.loads(raw, object_pairs_hook=object_pairs)
        valid = (isinstance(parsed, dict) and tuple(parsed) == FIELDS
                 and all(type(parsed[k]) is bool for k in FIELDS[:-1])
                 and parsed['action'] in h.picker.eligible(person))
        return parsed, valid
    except (ValueError, TypeError):
        return None, False


class DeskHTTP:
    def __init__(self):
        self.last = None

    def __call__(self, path, body=None):
        request = urllib.request.Request(h.picker.URL + path,
                  None if body is None else json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        if path == '/completion':
            self.last = result
        return result


def base_row(s, variant, model):
    label, rule = h.label(s)
    return dict(id=s['id'], variant=variant, model=model, split=s['split'], label=label, rule=rule)


def letter_rows(items, variant, gemma, transport, calibration):
    prompt_variant = 'V0' if variant == 'A0' else 'V2'
    for s in items:
        keys, phrases = h.options(s, prompt_variant, reverse=variant == 'A5')
        started = time.perf_counter()
        distribution = gemma.decide(h.state(s, prompt_variant), phrases)
        ms = (time.perf_counter() - started) * 1000
        raw = dict(zip(keys, (distribution[p] for p in phrases)))
        probabilities = raw
        if variant in ('A4', 'A5'):
            baseline = calibration[variant][','.join(keys)]['probabilities']
            adjusted = {k: raw[k] / baseline[k] for k in keys}
            probabilities = {k: v / sum(adjusted.values()) for k, v in adjusted.items()}
        data = transport.last
        yield dict(**base_row(s, variant, 'E4B'), choice=max(keys, key=probabilities.get), valid=True,
                   eligible=keys, raw_probabilities=raw, probabilities=probabilities,
                   prompt_tokens=gemma.prompt_tokens, generated_tokens=data['timings']['predicted_n'],
                   raw_text=data.get('content'), parsed_json=None, ms=ms)


def json_rows(items, variant, model, transport):
    for s in items:
        body = json_payload(s, variant)
        started = time.perf_counter()
        data = transport('/completion', body)
        ms = (time.perf_counter() - started) * 1000
        raw = data.get('content')
        parsed, valid = parse_json(raw, truth(s)['person_visible'])
        yield dict(**base_row(s, variant + '-' + model, model), choice=parsed['action'] if valid else None,
                   valid=valid, raw_text=raw, parsed_json=parsed, prompt=body['prompt'],
                   prompt_tokens=data['timings']['prompt_n'], generated_tokens=data['timings']['predicted_n'], ms=ms)


def diagnostic_rows(rows, model):
    return [dict(r, variant='B3-' + model,
                 choice=action_from_facts(r['parsed_json']) if r['valid'] else None) for r in rows]


def write_rows(path, rows):
    saved = []
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
            f.flush()
            saved.append(row)
    print(path.name, len(saved), flush=True)
    return saved


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metrics(rows):
    confusion = {m: dict.fromkeys((*h.MODES, 'INVALID'), 0) for m in h.MODES}
    for r in rows:
        confusion[r['label']][r['choice'] if r['valid'] else 'INVALID'] += 1
    exact = [Fraction(confusion[m][m], sum(confusion[m].values())) if sum(confusion[m].values()) else Fraction(0)
             for m in h.MODES]
    recall = dict(zip(h.MODES, map(float, exact)))
    return dict(n=len(rows), accuracy=sum(r['valid'] and r['choice'] == r['label'] for r in rows) / len(rows),
                macro_recall=float(statistics.mean(exact)), recall=recall, confusion=confusion,
                r1_violations=sum(r['rule'] == 'R1' and r['choice'] in ('APPROACH', 'GREET') for r in rows),
                json_validity=sum(r['valid'] for r in rows) / len(rows),
                median_prompt_tokens=statistics.median(r['prompt_tokens'] for r in rows),
                median_generated_tokens=statistics.median(r['generated_tokens'] for r in rows),
                median_total_tokens=statistics.median(r['prompt_tokens'] + r['generated_tokens'] for r in rows),
                median_ms=statistics.median(r['ms'] for r in rows))


def select(scores):
    eligible = [v for v in CANDIDATES if scores[v]['r1_violations'] == 0 and min(scores[v]['recall'].values()) >= .6
                and (v.startswith('A') or scores[v]['json_validity'] >= .95)]
    return min(eligible, key=lambda v: (-scores[v]['macro_recall'], scores[v]['median_total_tokens'], v)) if eligible else None


def summarize(out):
    """Recompute every statistic from per-item files; invalid JSON stays in every denominator."""
    items = {s['id']: s for s in read_rows(out / 'set.jsonl')}
    rows = {v: read_rows(out / (v + '_DEV.jsonl')) for v in VARIANTS}
    expected = {i for i, s in items.items() if s['split'] == 'DEV'}
    calibration = json.loads((out / 'calibration.json').read_text())
    for variant, values in rows.items():
        assert len(values) == len(expected) and {r['id'] for r in values} == expected
        for r in values:
            assert (r['label'], r['rule']) == h.label(items[r['id']])
            assert r['split'] == 'DEV' and r['variant'] == variant
            if variant.startswith('A'):
                keys, phrases = h.options(items[r['id']], 'V0' if variant == 'A0' else 'V2', variant == 'A5')
                assert keys == r['eligible']
                adjusted = r['raw_probabilities']
                if variant in ('A4', 'A5'):
                    baseline = calibration[variant][','.join(keys)]['probabilities']
                    adjusted = {k: adjusted[k] / baseline[k] for k in keys}
                assert all(abs(r['probabilities'][k] - adjusted[k] / sum(adjusted.values())) < 1e-9 for k in keys)
                assert r['valid'] and r['choice'] == max(keys, key=r['probabilities'].get)
            else:
                parsed, valid = parse_json(r['raw_text'], truth(items[r['id']])['person_visible'])
                assert parsed == r['parsed_json'] and valid == r['valid']
                choice = (action_from_facts(parsed) if variant.startswith('B3') else parsed['action']) if valid else None
                assert r['choice'] == choice
    scores = {v: metrics(values) for v, values in rows.items()}
    fact_accuracy = {v: {k: sum(r['valid'] and r['parsed_json'][k] == truth(items[r['id']])[k] for r in rows[v]) / len(rows[v])
                        for k in FIELDS[:-1]} for v in CANDIDATES if v.startswith('B')}
    a4 = {r['id']: r['choice'] for r in rows['A4']}
    return dict(dev=scores, fact_accuracy=fact_accuracy, selected=select(scores),
                a5_changed=sum(r['choice'] != a4[r['id']] for r in rows['A5']),
                timing_validity='NOT A MEASUREMENT: agent resident', test=None)


def server_command(model):
    file = 'gemma-4-E4B-it-Q4_0.gguf' if model == 'E4B' else 'gemma-4-E2B-it-Q4_0.gguf'
    return [sm.LLAMA_SERVER, '-m', sm.HOME + '/models/' + file, '--port', '8080', '--ctx-size', '2048',
            '--threads', sm.THREADS, '--threads-batch', sm.BATCH_THREADS, '--parallel', '1',
            '--swa-full', '--cache-ram', '0', '--host', '127.0.0.1']


def mem_available():
    return next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))


def servers_left():
    result = subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


@contextmanager
def server(out, model, phase):
    if servers_left():
        raise RuntimeError('existing llama-server: refusing to stop someone else\'s server')
    command = server_command(model)
    record = dict(model=model, phase=phase, setup='setup_e4b_q4' if model == 'E4B' else 'setup_q4',
                  command=command, mem_before=mem_available())
    transport = DeskHTTP()
    with (out / (model + '_' + phase + '.server.log')).open('x') as log:
        started = time.monotonic()
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = started + 90
            while True:
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('server failed to become healthy; see server log')
                try:
                    if transport('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                time.sleep(.5)
            record.update(mem_after=mem_available(), load_seconds=time.monotonic() - started)
            print('SERVER_READY', model, phase, record['mem_after'], flush=True)
            yield transport
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                except ProcessLookupError:
                    proc.wait()
            record.update(exit=proc.returncode, servers_left=servers_left())
            with (out / 'servers.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print('SERVER_STOPPED', model, phase, record['servers_left'] or 'NONE', flush=True)
            assert not record['servers_left']


def freeze(out, source=None):
    source = SOURCE if source is None else source
    assert h.sha(source / 'set.jsonl') == SET_SHA
    assert not list(source.glob('*_TEST.jsonl')), 'PICKPROMPT-1 TEST must remain unscored'
    out.mkdir(parents=True, exist_ok=False)
    for name in ('set.jsonl', 'set_sha256.txt', 'freeze.json'):
        shutil.copyfile(source / name, out / name)
        assert h.sha(source / name) == h.sha(out / name)
    assert json.loads((out / 'freeze.json').read_text())['set_sha256'] == SET_SHA
    assert (out / 'set_sha256.txt').read_text().split()[0] == SET_SHA
    items = read_rows(out / 'set.jsonl')
    assert len({s['id'] for s in items}) == 120 and Counter(s['split'] for s in items) == {'DEV': 72, 'TEST': 48}
    assert all((s['label'], s['rule']) == h.label(s) for s in items)
    h.write_json(out / 'freeze2.json', dict(utc=datetime.now(timezone.utc).isoformat(), set_sha256=SET_SHA,
                 source=str(source), copied_sha256={n: h.sha(out / n) for n in ('set.jsonl', 'set_sha256.txt', 'freeze.json')},
                 code_sha256={n: h.sha(Path(__file__).parent / n) for n in ('harness2.py', 'test_harness2.py', 'README2.md',
                                                                          'harness.py', 'test_harness.py', 'README.md')},
                 json_constraint_field='grammar', candidates=CANDIDATES))
    return items


def run(out):
    items = freeze(out)
    frozen = json.loads((out / 'freeze2.json').read_text())
    dev = [s for s in items if s['split'] == 'DEV']
    events = (out / 'events.jsonl').open('x')
    def event(kind, **fields):
        events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=kind, **fields)) + '\n')
        events.flush()
    try:
        event('before_first_gemma_call', set_sha256=SET_SHA, code_sha256=frozen['code_sha256'], timeout_s=120)
        for model in ('E4B', 'E2B'):
            with server(out, model, 'DEV') as transport:
                if model == 'E4B':
                    gemma = h.StaticGemma(transport)
                    calibrations = {}
                    for variant in ('A0', 'A2', 'A4', 'A5'):
                        if variant in ('A4', 'A5'):
                            calibrations[variant] = {}
                            for person in (False, True):
                                sample = next(s for s in dev if truth(s)['person_visible'] == person)
                                keys, phrases = h.options(sample, 'V2', reverse=variant == 'A5')
                                text = h.state(sample, 'V2', content_free=True)
                                dist = gemma.decide(text, phrases)
                                probs = dict(zip(keys, (dist[p] for p in phrases)))
                                assert all(p > 0 for p in probs.values())
                                calibrations[variant][','.join(keys)] = dict(state=text, phrases=phrases, probabilities=probs)
                            h.write_json(out / 'calibration.json', calibrations)
                        write_rows(out / (variant + '_DEV.jsonl'), letter_rows(dev, variant, gemma, transport, calibrations))
                for variant in ('B1', 'B2'):
                    rows = write_rows(out / (variant + '-' + model + '_DEV.jsonl'), json_rows(dev, variant, model, transport))
                    if variant == 'B1':
                        write_rows(out / ('B3-' + model + '_DEV.jsonl'), diagnostic_rows(rows, model))
        summary = summarize(out)
        selected = summary['selected']
        event('selection_frozen', selected=selected, dev=summary['dev'])
        h.write_json(out / 'summary.json', summary)
        if selected:
            test_items = [s for s in items if s['split'] == 'TEST']
            model = 'E4B' if selected.startswith('A') else selected.split('-')[1]
            event('test_started', variant=selected)
            with server(out, model, 'TEST') as transport:
                if selected.startswith('A'):
                    gemma = h.StaticGemma(transport)
                    calibration = json.loads((out / 'calibration.json').read_text())
                    rows = letter_rows(test_items, selected, gemma, transport, calibration)
                else:
                    rows = json_rows(test_items, selected.split('-')[0], model, transport)
                summary['test'] = metrics(write_rows(out / (selected + '_TEST.jsonl'), rows))
        else:
            summary['outcome'] = 'NO VARIANT QUALIFIES'
        h.write_json(out / 'summary.json', summary)
        for name, digest in frozen['copied_sha256'].items():
            assert h.sha(out / name) == digest == h.sha(SOURCE / name)
        for name, digest in frozen['code_sha256'].items():
            assert h.sha(Path(__file__).parent / name) == digest, 'code changed after first Gemma call'
        event('finished', selected=selected, set_sha256=h.sha(out / 'set.jsonl'))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    def exit_on_signal(signum, frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, exit_on_signal)
    run(args.output)


if __name__ == '__main__':
    main()
