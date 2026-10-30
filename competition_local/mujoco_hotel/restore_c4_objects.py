"""Restore source C4 rigid bodies from the preview's world-baked geometries."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics

def restore_c4_objects(path,builder=None):
    from build_hotel import Builder,name,fmt,transform_points,HERE
    b=builder or Builder()
    tree=ET.parse(path);root=tree.getroot();world=root.find('worldbody');records=[]
    for prim in b.prims:
        source=str(prim.GetPath())
        if not source.startswith('/World/C4/') or not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        api=UsdPhysics.RigidBodyAPI(prim)
        if not api.GetRigidBodyEnabledAttr().Get() or api.GetKinematicEnabledAttr().Get():continue
        body_name=name(source)
        existing=world.find(f"body[@name='{body_name}']")
        if existing is not None:continue
        prefix=body_name+'_'
        geoms=[g for g in world.findall('geom') if g.get('name','').startswith(prefix)]
        if not geoms:raise RuntimeError(f'No converted C4 geometry for {source}')
        points=[];masses=[];approximations=[]
        for child in b.prims:
            child_path=str(child.GetPath())
            if not child_path.startswith(source+'/') or not child.IsA(UsdGeom.Mesh):continue
            points.append(transform_points(np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get()),b.xforms.GetLocalToWorldTransform(child)))
            mass=child.GetAttribute('physics:mass')
            if mass and mass.HasAuthoredValueOpinion() and float(mass.Get())>0:masses.append(float(mass.Get()))
            approx=child.GetAttribute('physics:approximation')
            if approx:approximations.append(str(approx.Get()))
        if not masses:raise RuntimeError(f'No authored mass for {source}; refusing to invent one')
        mass=sum(masses)
        # Source has no explicit inertia. Use its existing convex collision hull.
        origin=np.array(b.xforms.GetLocalToWorldTransform(prim))[3,:3]
        hull=trimesh.convex.convex_hull(np.concatenate(points)-origin)
        props=hull.mass_properties
        inertia=props.inertia*(mass/props.mass)
        diagonal,axes=np.linalg.eigh(inertia)
        if np.linalg.det(axes)<0:axes[:,0]*=-1
        q=Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
        body=ET.SubElement(world,'body',name=body_name,pos=fmt(origin))
        ET.SubElement(body,'freejoint',name=body_name+'_free')
        ET.SubElement(body,'inertial',mass=f'{mass:.12g}',pos=fmt(props.center_mass),diaginertia=fmt(diagonal),quat=fmt(q))
        for geom in geoms:
            world.remove(geom)
            geom.set('pos',fmt(np.fromstring(geom.get('pos','0 0 0'),sep=' ')-origin))
            if geom.get('group')=='3':
                # C4 must collide with the robot AND the hotel/table/other C4s.
                geom.set('contype','3');geom.set('conaffinity','3')
                geom.set('condim','4')
            body.append(geom)
        records.append({'source':source,'body':body_name,'mass_kg':mass,'source_position_m':origin.tolist(),
            'geoms':len(geoms),'source_collision_approximations':approximations,
            'inertia':'computed from convex hull; original USD has no explicit inertia',
            'limitation':'Source SDF collision remains a convex approximation in this adapter.' if 'sdf' in approximations else None})
    ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)
    report_path=HERE/'c4_restoration.json'
    if records:report_path.write_text(json.dumps({'restored':records,'source_assets_unchanged':True},indent=2),encoding='utf-8')
    return records

if __name__=='__main__':
    from hotel_runtime import MODEL
    print(json.dumps(restore_c4_objects(MODEL),indent=2))
