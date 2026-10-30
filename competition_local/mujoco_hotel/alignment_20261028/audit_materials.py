"""Read source material bindings and compare derived material colors."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_hotel import Builder
from pxr import Usd, UsdGeom, UsdShade

b = Builder()
tree = ET.parse(HERE.parent / 'assets/hotel_b2w.xml')
mats = {x.get('name'): dict(x.attrib) for x in tree.findall('asset/material')}
records = {}
for prim in b.prims:
    if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
        continue
    try:
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    except Exception:
        continue
    if not mat:
        continue
    path = str(mat.GetPath())
    record = records.setdefault(path, {'path': path, 'users': [], 'shaders': [],
        'derived': mats.get('mat_' + hashlib.sha256(path.encode()).hexdigest()[:14])})
    record['users'].append(str(prim.GetPath()))
    if record['shaders']:
        continue
    for shader in Usd.PrimRange(mat.GetPrim()):
        if not shader.IsA(UsdShade.Shader):
            continue
        data = {'path': str(shader.GetPath()), 'attrs': {}}
        for attr in shader.GetAttributes():
            key = attr.GetName()
            if key.startswith(('inputs:', 'info:')):
                value = attr.Get()
                if value is not None:
                    data['attrs'][key] = str(value)
        record['shaders'].append(data)
out = list(records.values())
(HERE / 'source_material_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
selected = [r for r in out if any('trash' in u.lower() or 'wall' in u.lower() for u in r['users'])]
for r in selected:
    print(json.dumps({'path': r['path'], 'users': r['users'][:4], 'derived': r['derived'], 'shaders': r['shaders']}, ensure_ascii=False))
print('TOTAL', len(out), 'SELECTED', len(selected))
