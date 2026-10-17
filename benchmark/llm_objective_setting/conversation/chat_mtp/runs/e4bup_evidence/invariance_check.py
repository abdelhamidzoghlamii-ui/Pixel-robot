import ast,hashlib,json,subprocess,sys,types
from pathlib import Path
root=Path('/termux-home/robot');out=Path('/termux-home/review/e4bup')
sys.path.insert(0,str(root))
import chat,server_manager as sm
runner=root/'benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py'
sys.path.insert(0,str(runner.parent))
import chat_mtp as cm
base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip()
def original(path): return subprocess.check_output(['git','show',f'{base}:{path}'],cwd=root).decode()
oldsm=ast.parse(original('server_manager.py'));newsm=ast.parse((root/'server_manager.py').read_text())
def branches(tree):
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='start_setup').body[-1]
 result={}
 while isinstance(node,ast.If):
  if isinstance(node.test,ast.Compare):
   result[node.test.comparators[0].value]=[ast.dump(n) for n in node.body]
  node=node.orelse[0] if node.orelse else None
 return result
ob,nb=branches(oldsm),branches(newsm)
for name in ('setup_q4','setup_q4_mtp','setup_e4b'): assert ob[name]==nb[name],name
for tree in (oldsm,newsm):
 funcs={n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
 if tree is oldsm: oldfuncs=funcs
 else:
  for name in ('start_server','kill_servers','wait_for_server','mtp_draft_ok'): assert funcs[name]==oldfuncs[name]
basechat=types.ModuleType('base_chat');basechat.__file__=str(root/'chat.py')
exec(compile(original('chat.py'),basechat.__file__,'exec'),basechat.__dict__)
assert chat.MODEL_FILES[:-1]==basechat.MODEL_FILES
for n,f,t,c in basechat.MODEL_FILES:
 oldkey=next(k for k,m in basechat.MODELS.items() if m['file']==f)
 newkey=next(k for k,m in chat.MODELS.items() if m['file']==f)
 assert newkey==oldkey
 for draft in (False,True): assert chat.server_cmd(newkey,draft)==basechat.server_cmd(oldkey,draft)
for family in ('gemma','qwen','mistral'):
 prompt=chat.build_prompt(family,[{'role':'user','content':'Hello'}])
 assert chat.chat_payload(prompt,family)==basechat.chat_payload(prompt,family)
basecm=types.ModuleType('base_runner');basecm.__file__=str(runner)
exec(compile(original(str(runner.relative_to(root))),str(runner),'exec'),basecm.__dict__)
assert cm.SHA256==basecm.SHA256
for config in ('OFF','ON'): assert cm.command(config)==basecm.command(config)
assert cm.payload('hello')==basecm.payload('hello')
assert cm.prompt_texts()==basecm.prompt_texts()
assert cm.report([],[],cm.IDS)==basecm.report([],[],basecm.IDS)
changed=subprocess.check_output(['git','diff','--name-only'],cwd=root).decode().splitlines()
assert set(changed)=={'chat.py','server_manager.py','test_chat_mtp.py',str(runner.relative_to(root)),str(runner.parent.relative_to(root)/'test_chat_mtp_runner.py')}
assert subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root).decode().splitlines()==['test_e4b_mtp.py']
print('PASS: old setup branches/helper functions AST-identical; old menu tuples/indices, commands, payloads unchanged; E2B runner commands, hashes, prompts, payload and report byte-identical; scope exactly six files.')
