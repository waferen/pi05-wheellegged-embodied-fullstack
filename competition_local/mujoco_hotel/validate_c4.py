"""Physical checks for converted C4 objects in the real hotel scene."""
import json
import mujoco
import numpy as np
from original_teleop import OriginalTeleop,HERE

sim=OriginalTeleop();m,d=sim.model,sim.data
(HERE/'c4_validation.json').write_text(json.dumps({'passed':False,'status':'running'}),encoding='utf-8')
def report_failure(kind,value,tb):
    (HERE/'c4_validation.json').write_text(json.dumps({'passed':False,'status':'failed','error':str(value)}),encoding='utf-8')
    sys.__excepthook__(kind,value,tb)
import sys
sys.excepthook=report_failure
bodies=[i for i in range(m.nbody) if m.body(i).name.startswith('World_C4_')]
assert len(bodies)==6
initial=d.xpos[bodies].copy()
for _ in range(250):sim.control_frame(.02)
settled=d.xpos[bodies].copy()
# One speed norm mixed m/s and rad/s and aliased 2ms contact chatter.
# Measure actual COM displacement at every physics step over 200ms instead.
poses=[]
for _ in range(100):
    mujoco.mj_step(m,d);poses.append(d.xipos[bodies].copy())
poses=np.array(poses)
rows=[]
for i,body in enumerate(bodies):
    joint=int(m.body_jntadr[body]);assert m.jnt_type[joint]==mujoco.mjtJoint.mjJNT_FREE
    assert abs(float(m.body_mass[body])-.01)<1e-7
    colliders=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and m.geom_contype[g]]
    assert colliders and all(m.geom_contype[g]==3 and m.geom_conaffinity[g]==3 for g in colliders)
    speed=float(np.linalg.norm(d.qvel[m.jnt_dofadr[joint]:m.jnt_dofadr[joint]+6]))
    excursion=float(np.linalg.norm(np.ptp(poses[:,i],axis=0)))
    net=float(np.linalg.norm(poses[-1,i]-poses[0,i]))
    contacts=sum(d.contact[k].geom1 in colliders or d.contact[k].geom2 in colliders for k in range(d.ncon))
    rows.append({'body':m.body(body).name,'mass_kg':float(m.body_mass[body]),'initial_position':initial[i].tolist(),
        'settled_position':settled[i].tolist(),'instantaneous_qvel_norm_mixed_units':speed,'contacts':contacts,
        'pose_excursion_200ms_m':excursion,'net_displacement_200ms_m':net})
    assert excursion<.0005,(m.body(body).name,'pose excursion',excursion)
    assert net<.0002,(m.body(body).name,'pose drift',net)
    assert contacts>0,(m.body(body).name,'no supporting contact')
target=m.body('World_C4_easy_C4_1').id
before=d.xpos[target].copy()
d.xfrc_applied[target,0]=.15
for _ in range(8):sim.control_frame(.02)
d.xfrc_applied[target]=0
for _ in range(50):sim.control_frame(.02)
delta=d.xpos[target]-before
assert float(np.linalg.norm(delta[:2]))>.005,delta
assert np.isfinite(d.qpos).all() and d.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number==0
result={'passed':True,'objects':rows,'push':{'target':'World_C4_easy_C4_1','force_N':[.15,0,0],
    'duration_s':.16,'displacement_m':delta.tolist()},
    'noslip_iterations':int(m.opt.noslip_iterations),
    'scope':'Real hotel C4 support, 200ms physical-step COM pose stability and push response. Robot pickup is tested separately.'}
(HERE/'c4_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
