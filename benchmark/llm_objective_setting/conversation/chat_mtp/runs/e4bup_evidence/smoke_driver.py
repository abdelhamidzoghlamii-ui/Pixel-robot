import json,os,shlex,subprocess,sys,time,urllib.request
from pathlib import Path
from unittest import mock
sys.path.insert(0,'/termux-home/robot')
import chat
import server_manager as sm
out=Path('/termux-home/review/e4bup')

def mem():
 return next(l.strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))

def servers():
 return subprocess.run(['pgrep','-x','llama-server'],capture_output=True,text=True).stdout.strip()

assert not servers(), 'another llama-server exists; do not stop it'
records=[]
for kind in ('setup_e4b_q4_mtp','chat_e4b_q4_mtp'):
 print('SMOKE',kind,'NOT A MEASUREMENT',flush=True)
 record={'kind':kind,'mem_before':mem()}
 proc=None
 with (out/(kind+'.server.log')).open('w') as log:
  def launch(cmd):
   global proc
   record['command']=cmd
   proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   t0=time.monotonic()
   if kind.startswith('setup'):
    def system(command):
     if command.startswith('pkill '):
      assert not servers()
     else:
      assert command.endswith(' 2>/dev/null &')
      launch(shlex.split(command[:-len(' 2>/dev/null &')]))
     return 0
    with mock.patch.object(sm.os,'system',side_effect=system):
     assert sm.start_setup(kind), 'setup not healthy'
   else:
    key=next(k for k,m in chat.MODELS.items() if m['file']==chat.E4B_MTP_MODEL)
    cmd=chat.server_cmd(key,True)
    if chat.shutil.which('taskset') and {4,5,6,7}.issubset(os.sched_getaffinity(0)):
     cmd=['taskset','-c','4-7']+cmd
    launch(cmd)
    deadline=time.monotonic()+90
    while True:
     assert proc.poll() is None, 'server exited while loading'
     try:
      with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2) as r:
       if r.status==200: break
     except OSError: pass
     assert time.monotonic()<deadline, 'health timeout'
     time.sleep(0.5)
   record.update(load_seconds=time.monotonic()-t0,mem_after=mem())
   body=chat.chat_payload(chat.build_prompt('gemma',[{'role':'user','content':'Explain in three short sentences how a robot can explore a room safely.'}]),'gemma')
   record['payload']=body
   req=urllib.request.Request(chat.URL,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=120) as r: result=json.load(r)
   record['response']=result
   print(json.dumps(record),flush=True)
   assert result.get('content'), 'empty reply'
   assert result['timings'].get('draft_n',0)>0, 'no drafts recorded'
   assert result['timings'].get('draft_n_accepted',0)>0, 'no drafts accepted'
  finally:
   if proc is not None:
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
     proc.kill();proc.wait()
   record['server_exit']=None if proc is None else proc.returncode
   record['servers_left']=servers()
   records.append(record)
   (out/'smoke.json').write_text(json.dumps(records,indent=2)+'\n')
   print('STOPPED',kind,'servers_left',record['servers_left'] or 'NONE',flush=True)
   assert not record['servers_left']
