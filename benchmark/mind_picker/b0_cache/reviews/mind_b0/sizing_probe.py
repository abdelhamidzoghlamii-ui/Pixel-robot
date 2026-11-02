import json, subprocess, sys, time, os, signal
sys.dont_write_bytecode = True
sys.path.insert(0, '/data/data/com.termux/files/home/robot/benchmark/mind_picker/b0_cache')
import b0
s = json.loads(b0.SET.read_text())
assert b0.repro.port_free()
log = open('/tmp/claude-0/-data-data-com-termux-files-home-robot/11b52d99-1107-4a41-881f-33ee7e82dff3/scratchpad/probe_server.log', 'w')
p = subprocess.Popen(b0.mem1.command(2048), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    t0 = time.monotonic()
    while not b0.cr.healthy():
        assert p.poll() is None and time.monotonic() - t0 < 120
        time.sleep(0.2)
    post = lambda path, body: b0.cr.post(path, body, 30)
    p0, ids0, bad = b0.render(post, b0.messages(s, 'picker', 0))
    print('template problem:', bad)
    print('names:', b0.name_ids(post, p0, ids0, s['names']))
    for k in range(5):
        print('T', k, len(b0.render(post, b0.messages(s, 'picker', k))[1]))
    print('T4 with 32-word reply', len(b0.render(post, b0.messages(s, 'picker', 4, 'word ' * 32))[1]))
    print('t1 prefix', len(b0.tokenize(post, '<|turn>system\n' + b0.transcript(s, 1))))
    a = b0.render(post, b0.messages(s, 'shared', 1))[1]; b = b0.render(post, b0.messages(s, 'picker', 2, 'Hi.'))[1]
    print('LCP shared R4 vs T2', b0.mem1.lcp(a, b))
finally:
    os.killpg(p.pid, signal.SIGTERM); print('exit', p.wait(30))
