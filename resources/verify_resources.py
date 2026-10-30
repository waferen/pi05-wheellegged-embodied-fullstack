import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'资源校验清单.json').read_text(encoding='utf-8'))
failed=[]
for item in manifest['bundles']:
 path=root/item['archive'];h=hashlib.sha256()
 if not path.exists():failed.append(path.name);print('MISSING',path.name);continue
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 ok=path.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256']
 print('PASS' if ok else 'FAIL',path.name)
 if not ok:failed.append(path.name)
sys.exit(bool(failed))
