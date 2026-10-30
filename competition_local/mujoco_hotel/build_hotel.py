"""Convert the migrated hotel USD and its actual B2W/Piper to MuJoCo.

Source files are read only. Visual LODs and MuJoCo collision/drive mappings are
derived assets; static furniture and convex collision approximations are explicit.
"""
import hashlib,json,math,time,os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import trimesh
import fast_simplification
from PIL import Image
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf
from camera_calibration import is_robot_sensor,intrinsic_attributes,persist_physical_near
from material_resolver import resolve as resolve_material
from material_resolver import reflective_facade_albedo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ROBOT='/World/b2w_piper'
OUT=HERE/'assets'

def fmt(value):return ' '.join(f'{float(x):.9g}' for x in np.asarray(value).ravel())
def name(path):return str(path).replace('/','_').strip('_')
def quat(matrix):
    rotation=np.array(matrix,dtype=float)[:3,:3].T
    rotation=rotation/np.linalg.norm(rotation,axis=0)
    q=Rotation.from_matrix(rotation).as_quat()
    return fmt(q[[3,0,1,2]])
def usdquat(q):return np.array([float(q.GetReal()),*map(float,q.GetImaginary())])
def readable_path(path):
    text=str(path)
    if os.name=='nt' and not text.startswith('\\\\?\\'):return Path('\\\\?\\'+str(Path(path).resolve()))
    return Path(path)
def collision_enabled(prim):
    # The competition authoring applies some collider APIs to a visuals Xform
    # (notably wheels), whose descendant mesh provides the collision shape.
    current=prim
    while current and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.CollisionAPI):return UsdPhysics.CollisionAPI(current).GetCollisionEnabledAttr().Get()!=False
        if current!=prim and current.HasAPI(UsdPhysics.RigidBodyAPI):break
        current=current.GetParent()
    return False
def transform_points(points,matrix):
    m=np.array(matrix)
    return np.asarray(points,dtype=np.float64)@m[:3,:3]+m[3,:3]
def triangulate(counts,indices):
    counts=np.asarray(counts);indices=np.asarray(indices,dtype=np.int32)
    if np.all(counts==3):return indices.reshape(-1,3),np.arange(len(counts)),np.arange(len(indices)).reshape(-1,3)
    triangles=[];owners=[];corners=[];offset=0
    for i,count in enumerate(counts):
        polygon=indices[offset:offset+count]
        for j in range(1,count-1):
            triangles.append([polygon[0],polygon[j],polygon[j+1]]);owners.append(i);corners.append([offset,offset+j,offset+j+1])
        offset+=count
    return np.asarray(triangles,dtype=np.int32),np.asarray(owners,dtype=np.int32),np.asarray(corners,dtype=np.int32)

class Builder:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True)
        mapping=json.loads((HERE/'layer_map.json').read_text(encoding='utf-8'))
        self.source=next(k for k in mapping if k.replace('\\','/').endswith('/Collected_task03/hotel/task03.usd'))
        self.stage=Usd.Stage.Open(mapping[self.source]);self.xforms=UsdGeom.XformCache()
        assert UsdGeom.GetStageMetersPerUnit(self.stage)==1 and UsdGeom.GetStageUpAxis(self.stage)=='Z'
        self.root=ET.Element('mujoco',model='PPSUC competition hotel - original B2W and Piper')
        ET.SubElement(self.root,'compiler',angle='radian',inertiafromgeom='false',balanceinertia='true',autolimits='true')
        ET.SubElement(self.root,'option',timestep='0.002',integrator='implicitfast',gravity='0 0 -9.81',iterations='50',cone='elliptic',noslip_iterations='3')
        ET.SubElement(self.root,'size',njmax='10000',nconmax='2000')
        visual=ET.SubElement(self.root,'visual')
        ET.SubElement(visual,'global',offwidth='1280',offheight='800',azimuth='130',elevation='-24')
        ET.SubElement(visual,'headlight',ambient='.25 .25 .25',diffuse='.45 .45 .45',specular='.1 .1 .1')
        ET.SubElement(visual,'map',znear='.01',zfar='80',fogstart='40',fogend='80')
        self.assets=ET.SubElement(self.root,'asset');self.world=ET.SubElement(self.root,'worldbody')
        ET.SubElement(self.assets,'texture',name='preview_sky',type='skybox',builtin='gradient',rgb1='.28 .36 .47',rgb2='.65 .71 .78',width='256',height='1536')
        self.actuators=ET.SubElement(self.root,'actuator')
        ET.SubElement(self.world,'light',pos='20 7 8',dir='0 0 -1',directional='true',castshadow='true',diffuse='.4 .4 .4')
        from lighting import apply_lighting_xml
        apply_lighting_xml(self.root)
        self.prims=list(Usd.PrimRange(self.stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()))
        self.rigid={str(p.GetPath()):p for p in self.prims if str(p.GetPath()).startswith(ROBOT+'/') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
        self.bodies={};self.body_world={k:self.xforms.GetLocalToWorldTransform(v) for k,v in self.rigid.items()}
        self.mesh_assets={};self.materials={};self.mesh_records=[];self.errors=[];self.joint_records=[]
        library=HERE/'material_defaults.json'
        self.material_defaults=json.loads(library.read_text(encoding='utf-8'))['materials'] if library.exists() else {}
        overrides=HERE/'material_overrides.json'
        self.material_overrides=json.loads(overrides.read_text(encoding='utf-8'))['materials'] if overrides.exists() else {}

    def body_for(self,prim):
        p=prim
        while p and not p.IsPseudoRoot():
            path=str(p.GetPath())
            if path in self.rigid:return path
            p=p.GetParent()
        return None

    def robot(self):
        parents={};joints={}
        for p in self.prims:
            if str(p.GetPath()).startswith(ROBOT+'/') and p.IsA(UsdPhysics.Joint):
                joint=UsdPhysics.Joint(p);a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
                if a and b and str(b[0]) in self.rigid:parents[str(b[0])]=str(a[0]);joints[str(b[0])]=p
        def add(path):
            if path in self.bodies:return self.bodies[path]
            parent=parents.get(path)
            target=add(parent) if parent in self.rigid else self.world
            mat=self.body_world[path]
            rel=mat*self.body_world[parent].GetInverse() if parent in self.rigid else mat
            body=ET.SubElement(target,'body',name=name(path),pos=fmt(np.array(rel)[3,:3]),quat=quat(rel))
            self.bodies[path]=body
            if parent not in self.rigid:ET.SubElement(body,'freejoint',name='robot_root')
            mass=UsdPhysics.MassAPI(self.rigid[path])
            ET.SubElement(body,'inertial',mass=f'{mass.GetMassAttr().Get():.9g}',pos=fmt(mass.GetCenterOfMassAttr().Get()),
                quat=fmt(usdquat(mass.GetPrincipalAxesAttr().Get())),diaginertia=fmt(mass.GetDiagonalInertiaAttr().Get()))
            if path in joints:
                prim=joints[path]
                if not prim.IsA(UsdPhysics.FixedJoint):
                    joint=UsdPhysics.Joint(prim);revolute=prim.IsA(UsdPhysics.RevoluteJoint)
                    typed=UsdPhysics.RevoluteJoint(prim) if revolute else UsdPhysics.PrismaticJoint(prim)
                    axis={'X':Gf.Vec3d(1,0,0),'Y':Gf.Vec3d(0,1,0),'Z':Gf.Vec3d(0,0,1)}[str(typed.GetAxisAttr().Get())]
                    rotation=Gf.Rotation(Gf.Quatd(joint.GetLocalRot1Attr().Get()))
                    axis=rotation.TransformDir(axis)
                    lower=float(typed.GetLowerLimitAttr().Get());upper=float(typed.GetUpperLimitAttr().Get())
                    if revolute:lower,upper=np.deg2rad([lower,upper])
                    joint_name=prim.GetName();wheel=joint_name.endswith('_foot_joint')
                    attrs={'name':joint_name,'type':'hinge' if revolute else 'slide','pos':fmt(joint.GetLocalPos1Attr().Get()),
                           'axis':fmt(axis),'damping':'1' if wheel else ('25' if joint_name in ('arm_joint7','arm_joint8') else '5'),
                           'armature':'.01','frictionloss':'.01'}
                    if np.isfinite([lower,upper]).all() and abs(lower)<1e10 and abs(upper)<1e10 and lower<upper:
                        attrs.update(limited='true',range=fmt([lower,upper]))
                    else:attrs['limited']='false'
                    ET.SubElement(body,'joint',**attrs)
                    if wheel:
                        ET.SubElement(self.actuators,'velocity',name=joint_name+'_drive',joint=joint_name,kv='8',forcelimited='true',forcerange='-20 20',ctrlrange='-10 10')
                    else:
                        gripper=joint_name in ('arm_joint7','arm_joint8');arm=joint_name.startswith('arm_')
                        kp=2000 if gripper else 80 if arm else 160
                        limit=500 if gripper else 100 if arm else 320 if '_calf_' in joint_name else 200
                        ET.SubElement(self.actuators,'position',name=joint_name+'_drive',joint=joint_name,kp=str(kp),forcelimited='true',forcerange=fmt([-limit,limit]))
                    self.joint_records.append({'name':joint_name,'type':attrs['type'],'body':path,'range':[lower,upper],'source':str(prim.GetPath())})
            return body
        for path in self.rigid:add(path)

    def material(self,prim):
        try:material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        except Exception:material=None
        key=str(material.GetPath()) if material else 'unbound'
        if key in self.materials:return self.materials[key]
        color=np.array([.65,.67,.69]);texture=None;roughness=.7;metallic=0.
        if material:
            for child in Usd.PrimRange(material.GetPrim()):
                if not child.IsA(UsdShade.Shader):continue
                shader=UsdShade.Shader(child)
                source=child.GetAttribute('info:mdl:sourceAsset').Get()
                defaults={}
                if source and hasattr(source,'path'):
                    defaults=self.material_defaults.get(Path(source.path).name,{})
                variant=child.GetAttribute('info:mdl:sourceAsset:subIdentifier').Get()
                color,texture,values=resolve_material(shader,defaults,variant,readable_path,OUT)
                if texture is not None and key.startswith('/World/Lounge/AntiqueMirror') and key.endswith('/Looks/Mirror'):
                    texture=reflective_facade_albedo(texture,shader,OUT,readable_path)
                roughness=values.get('roughness',roughness)
                value=shader.GetInput('reflection_roughness_constant').Get()
                if value is not None:roughness=float(value)
                value=shader.GetInput('metallic_constant').Get()
                if value is not None:metallic=float(value)
        try:
            display=UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if not material and display:color=np.array(display[0])
        except Exception:pass
        override=self.material_overrides.get(key)
        if override:
            color=np.array(override['color']);texture=None
            roughness=override.get('roughness',roughness);metallic=override.get('metallic',metallic)
        key_name='mat_'+hashlib.sha256(key.encode()).hexdigest()[:14]
        attrs={'name':key_name,'rgba':fmt([*color,override.get('alpha',1) if override else 1]),'specular':f'{max(.1,metallic):.4g}','shininess':f'{max(0,1-roughness):.4g}'}
        if texture:
            texture_name='tex_'+hashlib.sha256(str(texture).encode()).hexdigest()[:16]
            target=OUT/(texture_name+'.png')
            try:
                if not target.exists():
                    with Image.open(texture) as image:
                        image=image.convert('RGB');image.thumbnail((1024,1024));image.save(target)
                if not any(x.get('name')==texture_name for x in self.assets.findall('texture')):
                    ET.SubElement(self.assets,'texture',name=texture_name,type='2d',file=target.name)
                attrs['texture']=texture_name
            except Exception as exc:self.errors.append({'texture':str(texture),'error':str(exc)})
        ET.SubElement(self.assets,'material',**attrs)
        self.materials[key]=key_name
        return key_name

    def mesh(self,prim,selected_faces=None,material_prim=None,tag='',mode='all'):
        mesh=UsdGeom.Mesh(prim);path=str(prim.GetPath());body_path=self.body_for(prim)
        robot=body_path is not None
        purpose=str(mesh.GetPurposeAttr().Get());visible=mesh.ComputeVisibility()!='invisible' and purpose not in ('guide','proxy') and '/collisions/' not in path
        collision=collision_enabled(prim)
        if mode=='visual':collision=False
        if mode=='collision':visible=False
        if not visible and not collision:return
        points=np.array(mesh.GetPointsAttr().Get(),dtype=np.float64)
        if len(points)<3:return
        faces,owners,corners=triangulate(mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get())
        if selected_faces is not None:
            chosen=np.isin(owners,np.asarray(selected_faces,dtype=np.int32))
            faces,owners,corners=faces[chosen],owners[chosen],corners[chosen]
        if len(faces)==0:return
        source_faces=len(faces)
        if mesh.GetOrientationAttr().Get()=='leftHanded':faces=faces[:,::-1];corners=corners[:,::-1]
        uv=None;corner_uv=None
        try:
            st=UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
            if st:
                flat=np.array(st.ComputeFlattened(),dtype=float)
                if str(st.GetInterpolation())=='faceVarying' and len(flat)==sum(mesh.GetFaceVertexCountsAttr().Get()):
                    corner_uv=flat[corners]
                    # Average seams for the LOD; high frequency texture is approximate.
                    indices=np.array(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int)
                    uv=np.zeros((len(points),2));weight=np.zeros(len(points))
                    np.add.at(uv,indices,flat);np.add.at(weight,indices,1);uv/=np.maximum(weight[:,None],1)
                elif len(flat)==len(points):uv=flat;corner_uv=flat[faces]
        except Exception:pass
        max_faces=12000 if robot else 3000
        # Atlas seams on facade panels and task objects must retain per-corner
        # UVs. Averaging them during LOD smears straps and packaging colors.
        if (path.startswith('/World/Lounge/AntiqueMirror') and path.endswith(('_Mirror','_Frame'))) or path.startswith('/World/C4/'):
            max_faces=max(max_faces,len(faces))
        # Weld CAD triangle soup before simplification so shared edges survive.
        digest=hashlib.sha256(points.tobytes()+faces.tobytes()+str(max_faces).encode()).hexdigest()[:20]
        local_cache=OUT/(digest+'.npz')
        exact=len(faces)<=max_faces
        if exact:
            used,inverse=np.unique(faces,return_inverse=True)
            lod_points=points[used];lod_faces=inverse.reshape(-1,3)
        elif local_cache.exists():
            with np.load(local_cache) as cache:
                lod_points=cache['points'];lod_faces=cache['faces']
        else:
            welded=trimesh.Trimesh(points,faces,process=True)
            lod_points=np.asarray(welded.vertices);lod_faces=np.asarray(welded.faces)
            if len(lod_faces)>max_faces:
                lod_points,lod_faces=fast_simplification.simplify(lod_points,lod_faces,target_count=max_faces,agg=7)
            np.savez_compressed(local_cache,points=lod_points,faces=lod_faces)
        if len(lod_faces)<1:return
        if exact and corner_uv is not None:
            texture_coords=corner_uv.reshape(-1,2);texture_faces=np.arange(len(texture_coords)).reshape(-1,3)
        elif uv is not None:
            texture_coords=uv[cKDTree(points).query(lod_points)[1]];texture_faces=lod_faces.copy()
        else:texture_coords=None;texture_faces=None
        mat=self.xforms.GetLocalToWorldTransform(prim)
        if robot:mat=mat*self.body_world[body_path].GetInverse()
        transformed=transform_points(lod_points,mat)
        centered=transformed-transformed.mean(axis=0)
        eigenvalues,eigenvectors=np.linalg.eigh(centered.T@centered)
        planar=eigenvalues[0]<max(eigenvalues[-1],1e-20)*1e-12
        thickness=0.
        if planar:
            # MuJoCo requires a 3D convex hull even for visual mesh assets.
            # Give source planes a sub-millimeter thickness in the derived file.
            thickness=float(np.clip(np.ptp(transformed,axis=0).max()*1e-5,.00002,.001))
            normal=eigenvectors[:,0]
            offset=len(transformed)
            transformed=np.concatenate([transformed+normal*thickness/2,transformed-normal*thickness/2])
            lod_faces=np.concatenate([lod_faces,lod_faces[:,::-1]+offset])
            if texture_coords is not None:texture_faces=np.concatenate([texture_faces,texture_faces[:,::-1]])
        # A unique pose per USD instance is baked into the corresponding mesh.
        mesh_key='mesh_'+hashlib.sha256((path+tag+digest+'material_uv_v2').encode()).hexdigest()[:20]
        if planar:mesh_key+='_thin'
        file=OUT/(mesh_key+'.obj')
        if not file.exists():
            with file.open('w',encoding='ascii') as stream:
                np.savetxt(stream,transformed,fmt='v %.8g %.8g %.8g')
                if texture_coords is not None:np.savetxt(stream,texture_coords,fmt='vt %.8g %.8g')
                for row,tri in enumerate(lod_faces+1):
                    stream.write('f '+' '.join(f'{i}/{texture_faces[row,j]+1}' if texture_coords is not None else str(i) for j,i in enumerate(tri))+'\n')
        ET.SubElement(self.assets,'mesh',name=mesh_key,file=file.name,inertia='shell',maxhullvert='64')
        target=self.bodies[body_path] if robot else self.world
        if visible:
            ET.SubElement(target,'geom',name=name(path+tag)+'_visual',type='mesh',mesh=mesh_key,material=self.material(material_prim or prim),contype='0',conaffinity='0',group='0',density='0')
        if collision:
            # Match source per-prim convex collision. Room-wide concave objects
            # require later decomposition; no whole-hotel convex hull is created.
            ET.SubElement(target,'geom',name=name(path)+'_collision',type='mesh',mesh=mesh_key,contype='2' if robot else '1',
                conaffinity='1' if robot else '2',group='3',density='0',friction='1 .005 .0001',rgba='.2 .6 .3 .2',solref='.01 1')
        self.mesh_records.append({'source':path,'material_subset':tag,'source_triangles':source_faces,'lod_triangles':len(lod_faces),'visible':visible,'collision':bool(collision),'robot':robot,'derived_planar_thickness_m':thickness,'exact_corner_uv':bool(exact and corner_uv is not None)})

    def primitive(self,prim):
        body_path=self.body_for(prim)
        if not body_path:return
        mat=self.xforms.GetLocalToWorldTransform(prim)*self.body_world[body_path].GetInverse()
        m=np.array(mat);scale=np.linalg.norm(m[:3,:3],axis=1)
        attrs={'name':name(prim.GetPath()),'pos':fmt(m[3,:3]),'quat':quat(mat),'contype':'2','conaffinity':'1','density':'0','group':'3','friction':'1 .005 .0001'}
        if prim.IsA(UsdGeom.Cube):attrs.update(type='box',size=fmt(scale*float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2))
        elif prim.IsA(UsdGeom.Capsule):
            capsule=UsdGeom.Capsule(prim);axis={'X':0,'Y':1,'Z':2}[str(capsule.GetAxisAttr().Get())]
            vector=np.zeros(3);vector[axis]=float(capsule.GetHeightAttr().Get())/2
            ends=transform_points(np.array([-vector,vector]),mat)
            attrs.pop('pos');attrs.pop('quat');attrs.update(type='capsule',fromto=fmt(ends),size=fmt([float(capsule.GetRadiusAttr().Get())*max(np.delete(scale,axis))]))
        else:return
        ET.SubElement(self.bodies[body_path],'geom',**attrs)

    def cameras(self):
        for prim in self.prims:
            if not prim.IsA(UsdGeom.Camera):continue
            body_path=self.body_for(prim);target=self.bodies[body_path] if body_path else self.world
            matrix=self.xforms.GetLocalToWorldTransform(prim)
            if body_path:matrix=matrix*self.body_world[body_path].GetInverse()
            camera=UsdGeom.Camera(prim)
            focal=camera.GetFocalLengthAttr().Get();aperture=camera.GetVerticalApertureAttr().Get()
            fovy=min(150,max(10,math.degrees(2*math.atan(aperture/(2*focal)))))
            optics=intrinsic_attributes(camera) if is_robot_sensor(prim.GetPath()) else {'fovy':f'{fovy:.6g}'}
            ET.SubElement(target,'camera',name=name(prim.GetPath()),pos=fmt(np.array(matrix)[3,:3]),quat=quat(matrix),**optics)

    def run(self):
        self.robot()
        meshes=[p for p in self.prims if p.IsA(UsdGeom.Mesh)]
        for i,prim in enumerate(meshes):
            subsets=[UsdGeom.Subset(p) for p in prim.GetChildren() if p.IsA(UsdGeom.Subset) and str(UsdGeom.Subset(p).GetFamilyNameAttr().Get())=='materialBind']
            if subsets:
                self.mesh(prim,mode='collision')
                covered=[]
                for subset in subsets:
                    indices=list(subset.GetIndicesAttr().Get());covered.extend(indices)
                    self.mesh(prim,indices,subset.GetPrim(),'_'+subset.GetPrim().GetName(),mode='visual')
                remaining=np.setdiff1d(np.arange(len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())),covered)
                if len(remaining):self.mesh(prim,remaining,tag='_unassigned',mode='visual')
            else:self.mesh(prim)
            if i%20==0:
                print('mesh',i,'/',len(meshes),str(prim.GetPath()),flush=True)
                (HERE/'conversion_status.json').write_text(json.dumps({'phase':'exporting_meshes','processed':i,'total':len(meshes),'updated':time.time()}))
        for prim in self.prims:
            if prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Capsule):self.primitive(prim)
        self.cameras()
        output=OUT/'hotel_b2w.xml'
        tree=ET.ElementTree(self.root);ET.indent(tree);tree.write(output,encoding='utf-8',xml_declaration=True)
        from restore_c4_objects import restore_c4_objects
        restore_c4_objects(output,self)
        from refine_grasp_contacts import refine_grasp_contacts
        refine_grasp_contacts(output,self,decomposition=False,refine_fingers=False)
        from repair_container import repair_container
        repair_container(output)
        persist_physical_near(output)
        record={'source':self.source,'original_unchanged':True,'source_sha256':hashlib.sha256(Path(self.source).read_bytes()).hexdigest(),
            'output':str(output),'mesh_records':self.mesh_records,'joints':self.joint_records,'errors':self.errors,
            'rigid_robot_bodies':len(self.bodies),'source_triangles':sum(m['source_triangles'] for m in self.mesh_records),
            'lod_triangles':sum(m['lod_triangles'] for m in self.mesh_records),
            'limitations':['Hotel furniture and doors are static in this preview.','Static mesh collisions use per-prim convex approximations; concave details need further decomposition.','MDL shaders are approximated with diffuse textures/colors; UV seams and visual LOD may differ.','No navigation, stairs or grasping policy has been validated.'],
            'references':['https://mujoco.readthedocs.io/en/stable/computation.html','https://openusd.org/dev/api/usd_physics_page_front.html']}
        (HERE/'conversion_report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        (HERE/'conversion_status.json').write_text(json.dumps({'phase':'exported','finished':time.time(),'output':str(output)}))
        print('EXPORTED',record['source_triangles'],'->',record['lod_triangles'],'triangles',flush=True)

if __name__=='__main__':Builder().run()
