"""Contact-only bench using the converted real Piper fingers and real C4 mesh.

The bench prescribes gripper lift; this is not a full robot IK/pickup test.
No welds, adhesion, object position writes during simulation, or applied forces.
"""
import copy,json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image
from scipy.spatial.transform import Rotation
from hotel_runtime import HotelSimulation,HERE,MODEL

s=HotelSimulation();original=ET.parse(MODEL).getroot()
m,d=s.model,s.data
grip=m.body('World_b2w_piper_gripper_base').id
fingers=[m.body('World_b2w_piper_arm_link'+str(i)).id for i in (7,8)]
target=m.body('World_C4_easy_C4_1').id
center=np.mean([d.geom_xpos[next(i for i in range(m.ngeom) if m.geom_bodyid[i]==body and m.geom_contype[i])] for body in fingers],axis=0)
new_origin=np.array([0.,0.,.5]);approach=Rotation.from_euler('y',-90,degrees=True).as_matrix()
grip_rotation=approach@d.xmat[grip].reshape(3,3)
target_center=new_origin+approach@(center-d.xpos[grip])
root=ET.Element('mujoco',model='Piper C4 contact bench')
ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false',autolimits='true')
ET.SubElement(root,'option',timestep='.002',integrator='implicitfast',iterations='100',cone='elliptic')
root.append(copy.deepcopy(original.find('asset')))
for mesh in root.find('asset').findall('mesh'):mesh.set('file',str(HERE/'assets'/mesh.get('file')))
for tex in root.find('asset').findall('texture'):
    if tex.get('file'):tex.set('file',str(HERE/'assets'/tex.get('file')))
world=ET.SubElement(root,'worldbody')
gripper=copy.deepcopy(next(b for b in original.iter('body') if b.get('name')=='World_b2w_piper_gripper_base'))
gripper.set('pos',' '.join(map(str,new_origin)));gripper.set('quat',' '.join(map(str,Rotation.from_matrix(grip_rotation).as_quat()[[3,0,1,2]])))
axis=grip_rotation.T@np.array([0.,0.,1.])
ET.SubElement(gripper,'joint',name='bench_lift',type='slide',axis=' '.join(map(str,axis)),damping='10')
world.append(gripper)
object_body=copy.deepcopy(next(b for b in original.iter('body') if b.get('name')=='World_C4_easy_C4_1'))
object_origin=target_center-(d.xipos[target]-d.xpos[target])
object_body.set('pos',' '.join(map(str,object_origin)));world.append(object_body)
# Place supporting tabletop directly below the source C4 hull.
gid=next(i for i in range(m.ngeom) if m.geom_bodyid[i]==target and m.geom_contype[i])
rotation=d.geom_xmat[gid].reshape(3,3)
lower=(d.geom_xpos[gid]+rotation@m.geom_aabb[gid,:3])[2]-np.abs(rotation[2])@m.geom_aabb[gid,3:]
table_z=lower+object_origin[2]-d.xpos[target,2]-.001
ET.SubElement(world,'geom',name='table',type='plane',pos=f'0 0 {table_z}',size='1 1 .02',contype='1',conaffinity='3',friction='1 .005 .0001')
act=ET.SubElement(root,'actuator')
for i in (7,8):
    entry=copy.deepcopy(next(a for a in original.find('actuator') if a.get('name')==f'arm_joint{i}_drive'))
    entry.set('kp','2000');act.append(entry)
ET.SubElement(act,'position',name='lift',joint='bench_lift',kp='5000',kv='100')
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
for i in (7,8):model.dof_damping[model.joint(f'arm_joint{i}').dofadr[0]]=25
data.joint('arm_joint7').qpos[0]=.045;data.joint('arm_joint8').qpos[0]=-.045
data.ctrl[:]=[.045,-.045,0]
def run(seconds):
    for _ in range(round(seconds/model.opt.timestep)):mujoco.mj_step(model,data)
run(.4)
open_object=data.xipos[model.body('World_C4_easy_C4_1').id].copy()
data.ctrl[:2]=[.001,-.001];run(.6)
obj=model.body('World_C4_easy_C4_1').id
start=float(data.xipos[obj,2]);closed_object=data.xipos[obj].copy();contact_bodies=[]
def picture(label):
    camera=mujoco.MjvCamera();camera.lookat[:]=[-.16,0,.55];camera.distance=.42;camera.azimuth=80;camera.elevation=-25
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        renderer.update_scene(data,camera=camera)
        Image.fromarray(renderer.render()).save(HERE/f'c4_grip_{label}.png')
picture('closed')
for k in range(data.ncon):
    b1=int(model.geom_bodyid[data.contact[k].geom1]);b2=int(model.geom_bodyid[data.contact[k].geom2])
    if obj in (b1,b2):contact_bodies.append(model.body(b2 if b1==obj else b1).name)
for i in range(500):data.ctrl[2]=.12*(i+1)/500;mujoco.mj_step(model,data)
run(.5)
picture('lifted')
rise=float(data.xipos[obj,2]-start)
result={'passed':rise>.08 and all('World_b2w_piper_arm_link'+str(i) in contact_bodies for i in (7,8)),
    'object':'World_C4_easy_C4_1','object_mass_kg':float(model.body_mass[obj]),
    'closed_finger_positions':[float(data.joint('arm_joint7').qpos[0]),float(data.joint('arm_joint8').qpos[0])],
    'contact_bodies_before_lift':contact_bodies,'prescribed_gripper_lift_m':.12,'object_rise_m':rise,
    'object_open_xyz':open_object.tolist(),'object_closed_xyz':closed_object.tolist(),'object_final_xyz':data.xipos[obj].tolist(),
    'actual_lift_m':float(data.joint('bench_lift').qpos[0]),
    'scope':'Actual source finger/C4 geometry, original finger gains, frictional contacts only. Bench lift is prescribed; full hotel teleop pickup is not certified.'}
(HERE/'c4_grip_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
assert result['passed'],result
