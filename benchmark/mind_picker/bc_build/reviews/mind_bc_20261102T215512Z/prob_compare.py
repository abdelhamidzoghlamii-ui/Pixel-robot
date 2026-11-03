"""Coder's extra check after the run: probabilities and rows vs the b1609 records (not part of bc.py)."""
import json
M = '/termux-home/robot/benchmark/mind_picker/'
rows = [json.loads(l) for l in open('/termux-home/robot_runs/mind_bc/20261102T213956Z/requests.jsonl')]
L = lambda p: [json.loads(l) for l in open(M + p)]
b0 = {x['id']: x for x in L('b0_cache/runs/20261102T110006Z/requests.jsonl')}
l1 = {x['id']: x for x in L('b0b_slots/runs/20261102T123830Z_L1/requests.jsonl')}
b2 = {(x['order'], x['id']): x for x in L('b2_rules/runs/20261102T143251Z/requests.jsonl') if x.get('set') == 'TEST'}
dp = lambda a, b: max(abs(a[k] - b[k]) for k in a)
ph = {p: [x for x in rows if x['phase'] == p] for p in ('K1', 'K2', 'K3')}
print('K1 vs B0 record: max |dp|', max(dp(x['probs'], b0[x['id']]['probs']) for x in ph['K1'] if x.get('probs')),
      '; generated ids equal', all(x['generated_ids'] == b0[x['id']]['generated_ids'] for x in ph['K1']))
F = ('prompt_tokens', 'prompt_ids_sha256', 'prompt_n', 'cache_n', 'top', 'generated_ids')
print('K2 vs B0b L1 record, differing (id, field, record, now):',
      [(x['id'], f, l1[x['id']].get(f), x.get(f)) for x in ph['K2'] for f in F if x.get(f) != l1[x['id']].get(f)])
for x in ph['K2']:
    if x.get('probs'):
        print(f'  K2 {x["id"]}: |dp| vs B0b L1 {dp(x["probs"], l1[x["id"]]["probs"]):.6f}; top {x["top"]} / {l1[x["id"]]["top"]}')
print('K3 vs B2 record: max |dp|', max(dp(x['probs'], b2[(x['order'], x['id'])]['probs']) for x in ph['K3']),
      '; reasons equal', sum(x['reason'] == b2[(x['order'], x['id'])]['reason'] for x in ph['K3']), '/ 80',
      '; cache_n equal', sum(x['cache_n'] == b2[(x['order'], x['id'])]['cache_n'] for x in ph['K3']), '/ 80')
