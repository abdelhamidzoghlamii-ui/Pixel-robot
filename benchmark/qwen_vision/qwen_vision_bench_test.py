"""Offline checks: tiny fake HTTP child only; never loads llama or model weights."""
import base64
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


def fake_server(mode, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    if mode == 'crash':
        raise SystemExit(7)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = {'status': 'ok'} if self.path == '/health' else {
                'modalities': {'vision': mode != 'text-only'}, 'media_marker': '[img-1]'}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/completion'
            assert base64.b64decode(value['prompt']['multimodal_data'][0]) == b'fake JPEG'
            assert value['prompt']['prompt_string'].count('[img-1]') == 1
            assert 'image_data' not in value
            body = json.dumps({'content': 'PERSON=L, TV=L'}).encode()
            self.send_response(500 if mode == 'http-error' else 200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if mode == 'slow-body':
                # Headers and some bytes arrive promptly, remainder exceeds deadline.
                self.wfile.write(body[:1])
                self.wfile.flush()
                time.sleep(1)
                body = body[1:]
            self.wfile.write(body)
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    import qwen_vision_bench as b
    assert b.parse_answer('person = l, TV=L') == {'PERSON': 'L', 'TV': 'L'}
    assert b.parse_answer('NONE') == {}
    for bad in ('', 'PERSON=L,PERSON=R', 'PERSON=L because', 'PERSON=LEFT', 'CAT=L'):
        assert b.parse_answer(bad) is None, bad
    p = b.payload(b'fake JPEG')
    assert p['prompt']['prompt_string'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    assert '<start_of_turn>' not in p['prompt']['prompt_string']
    assert p['cache_prompt'] is False and p['stop'] == ['<|im_end|>', '<|endoftext|>']
    assert b.mem_available() > 0
    original_popen = subprocess.Popen
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for mode in ('ok', 'http-error', 'slow-body', 'crash', 'text-only'):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            children = []
            def launch(command, **kwargs):
                assert command[command.index('--image-min-tokens') + 1] == '128'
                assert command[command.index('--image-max-tokens') + 1] == '128'
                assert kwargs['env']['LLAMA_MEDIA_MARKER'] == '[img-1]'
                child = original_popen([sys.executable, __file__, '--fake-server', mode, str(port)], **kwargs)
                children.append(child)
                return child
            with patch.object(b, 'PORT', port), patch.object(b.subprocess, 'Popen', launch):
                row = b.measure(Path('unused.gguf'), Path('unused-mmproj.gguf'), 128,
                                json.dumps(p).encode(), 0.1 if mode == 'slow-body' else None,
                                tmp / (mode + '.log'))
            assert children[0].poll() is not None, 'child leaked'
            assert row['mem_available_before_launch_kib'] > 0
            assert row['mem_available_under_load_min_kib'] > 0
            if mode == 'ok':
                assert row['http_success'] and row['http_completed'] and row['projector_loaded']
                assert row['actual'] == {'PERSON': 'L', 'TV': 'L'}
                assert row['process_alive_after'] and row['mem_available_request_min_kib'] > 0
            elif mode == 'http-error':
                assert row['http_completed'] and not row['http_success'] and row['http_status'] == 500
                assert row['process_alive_after'] and not row['parse_valid']
            elif mode == 'slow-body':
                assert row['timeout'] and not row['http_success'] and not row['http_completed']
                assert 0.08 <= row['latency_s'] < 0.8 and row['process_alive_after']
            elif mode == 'crash':
                assert row['exit_before_cleanup'] == 7 and not row['process_alive_after']
                assert not row['http_success'] and 'setup_error' in row
            else:
                assert not row['projector_loaded'] and not row['http_success']
            print('PASS', mode)
        # Missing and zero-byte models become skips, never launches.
        (tmp / 'Qwen3.5-zero.gguf').touch()
        with patch.object(b, 'MODEL_DIR', tmp), patch.object(b, 'MODELS', [
                ('missing', 'unused', '2B'), ('zero', 'unused', '2B')]), \
                patch.object(b.subprocess, 'check_output', return_value=b.BUILD):
            assert all(m.get('skip') for m in b.check_inputs()['models'])
        result = {'manifest': {'models': [{'name': 'test'}]}, 'rows': [
            dict(model='test', budget=128, **{'pass': 'unbounded'}, http_success=False,
                 process_alive_after=True, projector_loaded=True, correct=False)]}
        assert 'test | 128 | UNPROVEN' in b.summary(result)
        b.save(tmp, result)
        assert json.loads((tmp / 'results.json').read_text()) == result
    print('PASS: transport, strict labels, full-body deadline, memory, child cleanup, skips, summary')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fake-server':
        fake_server(sys.argv[2], int(sys.argv[3]))
    else:
        main()
