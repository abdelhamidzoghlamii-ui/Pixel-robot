from pathlib import Path
import hashlib,json,shutil,subprocess
out=Path('/termux-home/review/e4bup')
root=Path('/termux-home/models')
assert shutil.disk_usage(root).free >= 9680000000, 'disk gate'
commit=json.loads((out/'hub_model.json').read_text())['sha']
tree=json.loads((out/'hub_tree.json').read_text())
records=[]
for name in ('gemma-4-E4B-it-Q4_0.gguf','MTP/mtp-gemma-4-E4B-it-Q8_0.gguf'):
 e=next(e for e in tree if e['path']==name)
 dest=root / name.replace('MTP/','mtp/')
 dest.parent.mkdir(exist_ok=True)
 url=f'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/{commit}/{name}'
 print('DOWNLOAD',url,flush=True)
 subprocess.run(['curl','-f','-L','-C','-','--retry','3','--connect-timeout','30','-o',str(dest),url],check=True)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 got=h.hexdigest()
 if dest.stat().st_size != e['lfs']['size'] or got != e['lfs']['oid']:
  dest.unlink()
  raise SystemExit('download byte/hash mismatch; partial deleted; STOP')
 r={'url':url,'commit':commit,'path':str(dest),'bytes':dest.stat().st_size,'sha256':got}
 records.append(r)
 (out/'downloads.json').write_text(json.dumps(records,indent=2)+'\n')
 print('VERIFIED',json.dumps(r),flush=True)
