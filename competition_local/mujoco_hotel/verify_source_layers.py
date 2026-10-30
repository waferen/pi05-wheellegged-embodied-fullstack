"""Verify every source USD layer against the original migration receipts."""
import hashlib,json,time
from pathlib import Path
from build_hotel import HERE,ROOT,readable_path
mapping=json.loads((HERE/'layer_map.json').read_text(encoding='utf-8'))
receipts={}
for line in (ROOT/'migration/transfer_receipts_assets.jsonl').read_text(encoding='utf-8').splitlines():
    record=json.loads(line)
    if record.get('status')=='ok':receipts[record['path']]=record
results=[];started=time.time()
for source in mapping:
    path=Path(source);relative=path.relative_to(ROOT/'cloud_snapshot').as_posix()
    with readable_path(path).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    results.append({'path':relative,'matches_migration_sha256':digest==receipts[relative]['sha256']})
out={'checked_at':time.time(),'checked_usd_layers':len(results),'all_original_layers_unchanged':all(r['matches_migration_sha256'] for r in results),'elapsed_seconds':time.time()-started,'files':results}
(HERE/'source_integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print({k:v for k,v in out.items() if k!='files'})
assert out['all_original_layers_unchanged']
