import json, sys
t = json.load(open('/termux-home/mind_b2_test/b2_test.json'))
M = {'M1': "Explore and map the flat.", 'M2': "Find people and keep them company.",
     'M3': "Map the flat quietly and do not disturb anyone.", 'M4': "Help the owner find his keys."}
print('missions equal task:', t['missions'] == M, '| top-level keys', sorted(t), '| rows', len(t['items']))
N = ['explore', 'approach', 'greet', 'look', 'wait']; bad = 0
for line, it in zip(open(sys.argv[1]), t['items']):
    i, rest = line.rstrip('\n').split(' ', 1); m, cur, ev, mem, al, lab = rest.split('|')
    s, _, ok = lab.partition(' ('); ok = [ok.rstrip(')')] if ok else []
    want = dict(id=int(i), mission=m, current=cur, event=ev, memory=None if mem == '-' else mem, allowed=N, strict=s, also_ok=ok)
    if it != want: bad += 1; print('DIFF', i, it, want)
print('items differing from the task list:', bad)
