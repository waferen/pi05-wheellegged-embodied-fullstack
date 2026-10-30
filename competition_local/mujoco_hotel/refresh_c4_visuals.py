"""Restore source C4 face-corner UVs without changing any physics mapping."""
import copy,json,xml.etree.ElementTree as ET
from pxr import UsdGeom
from build_hotel import Builder,HERE,OUT


def refresh():
    b=Builder()
    for prim in b.prims:
        if not str(prim.GetPath()).startswith('/World/C4/') or not prim.IsA(UsdGeom.Mesh):continue
        subsets=[UsdGeom.Subset(p) for p in prim.GetChildren()
                 if p.IsA(UsdGeom.Subset) and str(UsdGeom.Subset(p).GetFamilyNameAttr().Get())=='materialBind']
        if subsets:
            covered=set()
            for subset in subsets:
                indices=list(subset.GetIndicesAttr().Get());covered.update(indices)
                b.mesh(prim,indices,subset.GetPrim(),'_'+subset.GetPrim().GetName(),mode='visual')
            remaining=sorted(set(range(len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())))-covered)
            if remaining:b.mesh(prim,remaining,tag='_unassigned',mode='visual')
        else:b.mesh(prim,mode='visual')
    path=OUT/'hotel_b2w.xml';tree=ET.parse(path);root=tree.getroot();before=copy.deepcopy(root)
    assets=root.find('asset');names={a.get('name') for a in assets}
    updates={g.get('name'):g for g in b.world.findall('geom')}
    changed=[]
    for g in root.findall('.//geom'):
        if g.get('name') not in updates:continue
        source=updates[g.get('name')]
        assert g.get('group')=='0' and g.get('contype')=='0'
        assert g.get('material')==source.get('material'), 'Material binding mismatch'
        changed.append({'geom':g.get('name'),'before':g.get('mesh'),'after':source.get('mesh')})
        g.set('mesh',source.get('mesh'))
    assert len(changed)==len(updates)==9
    for a in b.assets:
        if a.tag=='mesh' and a.get('name') not in names:
            assets.append(copy.deepcopy(a));names.add(a.get('name'))
    # Removing mesh assets and normalizing only the nine visual references must
    # leave the entire rest of the scene identical, including collision assets.
    check=copy.deepcopy(root)
    for item in changed:
        check.find(f".//geom[@name='{item['geom']}']").set('mesh',item['before'])
    for doc in (before,check):
        for a in list(doc.find('asset')):
            if a.tag=='mesh':doc.find('asset').remove(a)
    assert ET.tostring(before)==ET.tostring(check)
    assert all(r['exact_corner_uv'] and r['source_triangles']==r['lod_triangles'] for r in b.mesh_records)
    tree.write(path,encoding='utf-8',xml_declaration=True)
    out=HERE/'runs/c4_uv_20261028';out.mkdir(parents=True,exist_ok=True)
    report={'passed':True,'visual_updates':changed,'mesh_records':b.mesh_records,
            'physics_materials_lights_cameras_unchanged':True,'source_assets_unchanged':True}
    (out/'refresh.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Restored exact C4 UVs:',len(changed),'visuals;',sum(r['source_triangles'] for r in b.mesh_records),'triangles')


if __name__=='__main__':refresh()
