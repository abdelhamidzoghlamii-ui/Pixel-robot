import json, subprocess, sys, time, os, signal, urllib.request
sys.dont_write_bytecode = True
sys.path.insert(0, '/data/data/com.termux/files/home/robot/benchmark/mind_picker/b0b_slots')
import b0b
b0 = b0b.b0; s = json.loads(b0.SET.read_text())
assert b0.repro.port_free()
log = open('/tmp/claude-0/-data-data-com-termux-files-home-robot/11b52d99-1107-4a41-881f-33ee7e82dff3/scratchpad/probe_b0b_server.log', 'w')
p = subprocess.Popen(b0b.command('L2'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    t0 = time.monotonic()
    while not b0.cr.healthy():
        assert p.poll() is None and time.monotonic() - t0 < 120; time.sleep(0.2)
    post = lambda path, body: b0.cr.post(path, body, 30)
    print('slots', [(x['id'], x['n_ctx']) for x in json.loads(urllib.request.urlopen('http://127.0.0.1:8080/slots').read())])
    msgs = [{'role': 'system', 'content': s['companion_sys']}, {'role': 'user', 'content': s['speech']},
            {'role': 'assistant', 'content': 'I am exploring the flat.'}, {'role': 'user', 'content': b0b.CHAT2}]
    t = post('/apply-template', {'messages': msgs, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    print('R9 template equal:', t == b0.chat.build_prompt('gemma', msgs)); print(repr(t)); print(repr(b0.chat.build_prompt('gemma', msgs)))
finally:
    os.killpg(p.pid, signal.SIGTERM); print('exit', p.wait(30))
