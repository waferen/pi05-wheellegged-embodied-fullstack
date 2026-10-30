"""Replace convex-filled vessel colliders with thin source-surface prisms.

Source USD and visual meshes remain unchanged. Only the bowl's collision hulls
and net support are replaced; the pedestal keeps its original collider.
"""
import copy,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent

def repair_container(path=None):
    path=Path(path or HERE/'assets/hotel_b2w.xml');tree=ET.parse(path);root=tree.getroot()
    world=root.find('worldbody');assets=root.find('asset')
    report_path=HERE/'runs/scene_repair_20261028/container_repair.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    if any(g.get('name','').startswith('vessel_shell_') for g in world.findall('geom')):
        return json.loads(report_path.read_text()) if report_path.exists() else {'already_repaired':True}
    for parent in (world,assets):
        for child in list(parent):
            if child.get('name','').startswith('vessel_shell_'):parent.remove(child)
    meshes={a.get('name'):a for a in assets.findall('mesh')}
    removed=[];count=0
    for geom in list(world.findall('geom')):
        name=geom.get('name','')
        if not (name.startswith('World_BALL_') and name.endswith('_collision')):continue
        is_shell=any(s in name for s in ('6b8530f_','d3eafeb_'))
        is_net=name.startswith('World_BALL_Net_')
        if not (is_shell or is_net):continue
        mesh=trimesh.load(path.parent/meshes[geom.get('mesh')].get('file'),force='mesh',process=False)
        world.remove(geom);removed.append(name)
        attrs={k:v for k,v in geom.attrib.items() if k not in ('name','type','mesh','pos','quat','size')}
        if is_net:
            low,high=mesh.bounds;center=(low+high)/2;size=high-low;t=.004
            # Thin basket walls/floor preserve a receptacle instead of its hull.
            pieces=[(center*np.array([1,1,0])+[0,0,low[2]+.025],[(size[0])/2,(size[1])/2,t])]
            for axis in (0,1):
                for sign in (-1,1):
                    pos=center.copy();pos[axis]+=sign*(size[axis]/2-t)
                    half=size/2;half[axis]=t
                    pieces.append((pos,half))
            for i,(pos,half) in enumerate(pieces):
                ET.SubElement(world,'geom',name=f'vessel_shell_net_{i}',type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,half)),**attrs)
            continue
        for tri in mesh.triangles:
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);area=np.linalg.norm(normal)
            if area<1e-7:continue
            normal/=area
            # 2mm total thickness, not a solid hull spanning the interior.
            vertices=np.vstack([tri-normal*.001,tri+normal*.001])
            faces=[[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]]
            key=f'vessel_shell_{count:04d}'
            ET.SubElement(assets,'mesh',name=key,vertex=' '.join(f'{x:.10g}' for x in vertices.ravel()),
                face=' '.join(str(x) for f in faces for x in f),inertia='shell')
            ET.SubElement(world,'geom',name=key,type='mesh',mesh=key,**attrs);count+=1
    if not removed:raise RuntimeError('Vessel colliders already replaced; use saved pre-repair XML for regeneration')
    tree.write(path,encoding='utf-8',xml_declaration=True)
    report={'removed_convex_hulls':removed,'shell_prisms':count,'basket_boxes':5,'wall_thickness_m':.002,
        'visual_meshes_unchanged':True,'original_usd_unchanged':True,
        'approximation':'Source surface triangles extruded ±1mm; net approximated by five thin support panels.'}
    report_path.write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':print(json.dumps(repair_container(),indent=2))
