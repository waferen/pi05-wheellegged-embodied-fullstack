"""Refresh only facade mirror/frame visuals with exact source corner UVs."""
import copy,json,xml.etree.ElementTree as ET
from pxr import UsdGeom
from build_hotel import Builder,HERE,OUT

def refresh():
    b=Builder()
    for prim in b.prims:
        path=str(prim.GetPath())
        if path.startswith('/World/Lounge/AntiqueMirror') and path.endswith(('_Mirror','_Frame')) and prim.IsA(UsdGeom.Mesh):
            b.mesh(prim,mode='visual')
    path=OUT/'hotel_b2w.xml';tree=ET.parse(path);root=tree.getroot();assets=root.find('asset')
    names={x.get('name') for x in assets}
    for a in b.assets:
        if a.get('name') not in names:assets.append(copy.deepcopy(a));names.add(a.get('name'))
    updates={g.get('name'):g for g in b.world.findall('geom')}
    for g in root.findall('worldbody/geom'):
        if g.get('name') in updates:g.set('mesh',updates[g.get('name')].get('mesh'))
    tree.write(path,encoding='utf-8',xml_declaration=True)
    (HERE/'runs/scene_repair_20261028/facade_uv.json').write_text(json.dumps(b.mesh_records,indent=2))
    print('Exact facade UV meshes:',len(updates))

if __name__=='__main__':refresh()
