"""Keep a portable full pilot checkpoint alongside the Windows experiment."""
import hashlib,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=Path(json.loads((HERE/'serve_checkpoint.json').read_text())['checkpoint'])
target=HERE/'portable_checkpoint2999';pending=HERE/'portable_checkpoint2999.partial'
if not target.exists():
    manifest=[]
    for path in sorted(source.rglob('*')):
        dst=pending/path.relative_to(source)
        if path.is_dir():dst.mkdir(parents=True,exist_ok=True);continue
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        equal=False
        if dst.exists() and dst.stat().st_size==path.stat().st_size:
            with dst.open('rb') as f:equal=hashlib.file_digest(f,'sha256').hexdigest()==digest
        if not equal:
            dst.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as src,dst.open('wb') as dest:shutil.copyfileobj(src,dest,length=1024*1024)
            with dst.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
        manifest.append({'path':str(path.relative_to(source)),'bytes':path.stat().st_size,'sha256':digest})
    (HERE/'portable_manifest.json').write_text(json.dumps({'source':str(source),'completed_updates':3000,'files':manifest},indent=2))
    pending.rename(target)
print('Portable checkpoint:',target)
