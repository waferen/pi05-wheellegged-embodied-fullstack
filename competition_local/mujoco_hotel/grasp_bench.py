"""Prescribed six-axis fixture with original physical finger joints and C4.

The dynamic gripper follows a fixture target. C4 has no weld or adhesion.
"""
import copy,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from hotel_runtime import HotelSimulation,HERE,MODEL

def create_bench(approach_deg=-90,offset=(0,0,0),object_name='World_C4_easy_C4_1',model_path=None):
    sim=HotelSimulation(model_path);m,d=sim.model,sim.data
    source=ET.parse(model_path or MODEL).getroot()
    grip=m.body('World_b2w_piper_gripper_base').id
    fingers=[m.body('World_b2w_piper_arm_link'+str(i)).id for i in (7,8)]
    obj=m.body(object_name).id
    center=np.mean([d.geom_xpos[next(i for i in range(m.ngeom) if m.geom_bodyid[i]==b and m.geom_group[i]==0)] for b in fingers],axis=0)
    origin=np.array([0.,0.,.5]);approach=Rotation.from_euler('y',approach_deg,degrees=True).as_matrix()
    rotation=approach@d.xmat[grip].reshape(3,3)
    target_center=origin+approach@(center-d.xpos[grip])+np.array(offset)
    root=ET.Element('mujoco',model='Piper dynamic grip fixture')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false',autolimits='true')
    root.append(copy.deepcopy(source.find('option')))
    root.append(copy.deepcopy(source.find('asset')))
    for item in root.find('asset'):
        if item.get('file'):item.set('file',str(HERE/'assets'/item.get('file')))
    world=ET.SubElement(root,'worldbody')
    gripper=copy.deepcopy(next(b for b in source.iter('body') if b.get('name')=='World_b2w_piper_gripper_base'))
    gripper.set('pos',' '.join(map(str,origin)))
    gripper.set('quat',' '.join(map(str,Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]])))
    ET.SubElement(gripper,'freejoint',name='bench_gripper_free')
    world.append(gripper)
    ET.SubElement(world,'body',name='bench_fixture_target',mocap='true',pos=gripper.get('pos'),quat=gripper.get('quat'))
    equality=ET.SubElement(root,'equality')
    # Drive a dynamic gripper so its real velocity enters friction constraints.
    # Directly moving a mocap collider has zero velocity and invalidates drag tests.
    ET.SubElement(equality,'weld',body1=gripper.get('name'),body2='bench_fixture_target',solref='.004 1',solimp='.99 .999 .001')
    body=copy.deepcopy(next(b for b in source.iter('body') if b.get('name')==object_name))
    object_origin=target_center-(d.xipos[obj]-d.xpos[obj])
    body.set('pos',' '.join(map(str,object_origin)));world.append(body)
    lower=np.inf
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid]!=obj or not m.geom_contype[gid]:continue
        r=d.geom_xmat[gid].reshape(3,3)
        lower=min(lower,(d.geom_xpos[gid]+r@m.geom_aabb[gid,:3])[2]-np.abs(r[2])@m.geom_aabb[gid,3:])
    table_z=lower+object_origin[2]-d.xpos[obj,2]-.001
    ET.SubElement(world,'geom',name='table',type='plane',pos=f'0 0 {table_z}',size='1 1 .02',contype='1',conaffinity='3',friction='1 .005 .0001')
    act=ET.SubElement(root,'actuator')
    for i in (7,8):
        entry=copy.deepcopy(next(a for a in source.find('actuator') if a.get('name')==f'arm_joint{i}_drive'))
        entry.set('kp','2000');act.append(entry)
    # Preserve explicit finger/C4 contact pairs if the full model defines them.
    contact=source.find('contact')
    if contact is not None:
        names={g.get('name') for g in world.iter('geom')}
        pairs=[copy.deepcopy(p) for p in contact.findall('pair') if p.get('geom1') in names and p.get('geom2') in names]
        if pairs:ET.SubElement(root,'contact').extend(pairs)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    for i in (7,8):model.dof_damping[model.joint(f'arm_joint{i}').dofadr[0]]=25
    data.joint('arm_joint7').qpos[0]=.045;data.joint('arm_joint8').qpos[0]=-.045
    data.ctrl[:]=[.045,-.045];mujoco.mj_forward(model,data)
    return model,data
