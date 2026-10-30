"""Precise, decomposed source collision surfaces for Piper fingers and C4."""
import copy,hashlib,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import UsdGeom

DEFAULT_PAIR={'condim':'6','friction':'1 1 .005 .0001 .0001',
    'solref':'.005 1','solimp':'.99 .999 .001 .5 2'}

def apply_grasp_pairs(root,settings=None):
    finger_geoms=[];c4_geoms=[]
    for body in root.iter('body'):
        body_name=body.get('name','')
        names=[g.get('name') for g in body.findall('geom') if g.get('group')=='3']
        if body_name in ('World_b2w_piper_arm_link7','World_b2w_piper_arm_link8'):finger_geoms.extend(names)
        elif body_name.startswith('World_C4_'):c4_geoms.extend(names)
    assert finger_geoms and c4_geoms
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    for entry in list(contact):
        if entry.get('name','').startswith('grasp_pair_'):contact.remove(entry)
    for i,finger in enumerate(finger_geoms):
        for j,c4 in enumerate(c4_geoms):
            ET.SubElement(contact,'pair',name=f'grasp_pair_{i}_{j}',geom1=finger,geom2=c4,**(settings or DEFAULT_PAIR))
    return len(finger_geoms)*len(c4_geoms)

def parts_for(vertices,faces,cache,decompose):
    digest=hashlib.sha256(vertices.tobytes()+faces.tobytes()+b'coacd1.0.14-v2' + str(decompose).encode()).hexdigest()[:20]
    dest=cache/(digest+'.npz')
    if dest.exists():
        with np.load(dest) as data:return [(data[f'v{i}'],data[f'f{i}']) for i in range(int(data['count']))]
    mesh=trimesh.Trimesh(vertices,faces,process=True)
    if decompose:
        import coacd
        coacd.set_log_level('warn')
        parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),threshold=.03,
            max_convex_hull=12,preprocess_mode='auto',preprocess_resolution=80,
            resolution=1000,mcts_nodes=5,mcts_iterations=10,mcts_max_depth=2,
            pca=True,merge=True,decimate=True,max_ch_vertex=96,seed=0)
    else:
        hull=mesh.convex_hull;parts=[(np.asarray(hull.vertices),np.asarray(hull.faces))]
    arrays={'count':len(parts)}
    for i,(v,f) in enumerate(parts):arrays[f'v{i}']=v;arrays[f'f{i}']=f
    np.savez_compressed(dest,**arrays)
    return parts

def refine_grasp_contacts(path,builder=None,pair_settings=None,decomposition=True,refine_fingers=True):
    from build_hotel import Builder,name,fmt,quat,transform_points,triangulate,HERE,OUT
    b=builder or Builder();tree=ET.parse(path);root=tree.getroot();assets=root.find('asset')
    cache=HERE/'collision_parts';cache.mkdir(exist_ok=True)
    rows=[];finger_geoms=[];c4_geoms=[]
    for prim in b.prims:
        source=str(prim.GetPath())
        finger=source in ('/World/b2w_piper/arm_link7/collisions/link7/mesh','/World/b2w_piper/arm_link8/collisions/link8/mesh')
        if finger and not refine_fingers:continue
        c4=source.startswith('/World/C4/') and prim.IsA(UsdGeom.Mesh)
        if not (finger or c4):continue
        geom_name=name(source)+'_collision'
        body=next((body for body in root.iter('body') if any(g.get('name')==geom_name for g in body.findall('geom'))),None)
        if body is None:continue
        old=next(g for g in body.findall('geom') if g.get('name')==geom_name)
        template=copy.deepcopy(old)
        for geom in list(body.findall('geom')):
            if geom.get('name')==geom_name or geom.get('name','').startswith(geom_name+'_part_'):body.remove(geom)
        mesh=UsdGeom.Mesh(prim);vertices=np.array(mesh.GetPointsAttr().Get(),dtype=float)
        faces=triangulate(mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get())[0]
        if mesh.GetOrientationAttr().Get()=='leftHanded':faces=faces[:,::-1]
        approximation=prim.GetAttribute('physics:approximation')
        decompose=(finger and bool(decomposition)) or (decomposition is True and approximation and str(approximation.Get())=='sdf')
        print('processing',source,'decomposition',bool(decompose),flush=True)
        canonical_parts=parts_for(vertices,faces,cache,bool(decompose))
        matrix=b.xforms.GetLocalToWorldTransform(prim)
        if finger:matrix=matrix*b.body_world[b.body_for(prim)].GetInverse()
        parts=[]
        for v,f in canonical_parts:
            points=transform_points(v,matrix)
            if c4:points-=np.fromstring(body.get('pos'),sep=' ')
            parts.append(trimesh.Trimesh(points,f,process=True).convex_hull)
        for i,piece in enumerate(parts):
            digest=hashlib.sha256(source.encode()+piece.vertices.tobytes()+piece.faces.tobytes()).hexdigest()[:20]
            mesh_key='grasp_collision_'+digest
            file=OUT/(mesh_key+'.obj')
            if not file.exists():piece.export(file)
            if not any(a.get('name')==mesh_key for a in assets.findall('mesh')):
                ET.SubElement(assets,'mesh',name=mesh_key,file=file.name,maxhullvert='-1',inertia='convex')
            geom=copy.deepcopy(template)
            geom.set('name',geom_name if i==0 else geom_name+f'_part_{i:02d}')
            geom.set('mesh',mesh_key);geom.set('pos','0 0 0');geom.set('quat','1 0 0 0')
            body.append(geom)
            (finger_geoms if finger else c4_geoms).append(geom.get('name'))
        if c4 and decompose:
            # Source SDF bodies have no explicit inertia. Use the decomposed shape.
            volumes=np.array([p.volume for p in parts]);total=volumes.sum()
            com=np.average([p.center_mass for p in parts],axis=0,weights=volumes)
            inertia=np.zeros((3,3))
            for piece,volume in zip(parts,volumes):
                delta=piece.center_mass-com
                inertia+=piece.moment_inertia+volume*((delta@delta)*np.eye(3)-np.outer(delta,delta))
            inertial=body.find('inertial');inertia*=float(inertial.get('mass'))/total
            diagonal,axes=np.linalg.eigh(inertia)
            if np.linalg.det(axes)<0:axes[:,0]*=-1
            inertial.set('pos',fmt(com));inertial.set('diaginertia',fmt(diagonal))
            inertial.set('quat',fmt(Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]))
        rows.append({'source':source,'body':body.get('name'),'parts':len(parts),'decomposed':bool(decompose),
            'bounds_m':np.ptp(np.concatenate([p.vertices for p in parts]),axis=0).tolist()})
        print('refined',source,len(parts),flush=True)
    settings=pair_settings or DEFAULT_PAIR
    pair_count=apply_grasp_pairs(root,settings)
    ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)
    report={'source_assets_unchanged':True,'geometry':rows,'pair_settings':settings,
        'pair_count':pair_count,'coacd_version':'1.0.14',
        'note':'Original world scale, mass and actuator forces unchanged. Convex decomposition is an approximation to the source surfaces, not an exact SDF.'}
    report_file='grasp_refinement.json' if Path(path).name=='hotel_b2w.xml' else 'grasp_refinement_candidate.json'
    (HERE/report_file).write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    import shutil
    from hotel_runtime import HERE,MODEL
    path=HERE/'assets/hotel_b2w_grasp_candidate.xml';shutil.copyfile(MODEL,path)
    refine_grasp_contacts(path)
