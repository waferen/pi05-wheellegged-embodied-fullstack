"""Exercise original keys against real source actors and MuJoCo dynamics."""
import hashlib,json,time
import numpy as np
from original_teleop import OriginalTeleop,HERE,SOURCE,ARMS

def run(sim,frames):
    for _ in range(frames):sim.control_frame()
    assert np.isfinite(sim.data.qpos).all()

def main():
    sim=OriginalTeleop();run(sim,250)
    assert sim.data.xmat[sim.base].reshape(3,3)[2,2]>.95
    start=sim.data.xpos[sim.base].copy();sim.key('W');run(sim,100)
    movement=float(np.linalg.norm(sim.data.xpos[sim.base,:2]-start[:2]));assert movement>.2
    sim.key('W',False);run(sim,1);assert sim.command[0]==0
    sim.key('2');run(sim,100)
    assert sim.mode=='stair' and sim.data.xmat[sim.base].reshape(3,3)[2,2]>.9
    sim.key('5');assert sim.arm_ready and int(sim.data.eq_active.sum())==16
    sim.key('T');sim.key('C');run(sim,75)
    joint1=float(sim.data.joint('arm_joint1').qpos[0]);closed=float(sim.data.joint('arm_joint7').qpos[0])
    assert abs(joint1-.035)<.005 and abs(closed-.001)<.001
    sim.key('P');sim.key('H');run(sim,75)
    opened=float(sim.data.joint('arm_joint7').qpos[0]);assert abs(opened-.045)<.001
    assert np.max(np.abs(sim.data.qpos[sim.qadr[16:22]]))<.005
    sim.key('1');run(sim,100);assert sim.mode=='walk' and not sim.data.eq_active.any()
    before=float(sim.data.qpos[2]-sim.floor_z);sim.key('4');run(sim,100)
    lower=float(sim.data.qpos[2]-sim.floor_z)
    assert sim.mode=='low_grasp' and sim.arm_ready and lower<before-.15
    for _ in range(8):sim.key('G');sim.key('V')
    run(sim,100);point0=sim.arm_pose()[0].copy();sim.key('PAGE_DOWN');sim.key('PAGE_DOWN',False);run(sim,100)
    delta=sim.arm_pose()[0]-point0
    assert delta[2]<-.002 and np.linalg.norm(delta)<.03
    report={'checked_at':time.time(),'passed':True,'source_script':'PPSUC2026/scripts/rsl_rl/teleop_dual_cerebellum_arm.py',
        'original_modes':{'1':'walk','2':'stair','4':'low_grasp','5':'high_grasp'},'source_actor_inference':True,
        'walk_2s_displacement_m':movement,'release_stops_command':True,'stair_actor_flat_ground_passed':True,
        'T_joint1_rad':joint1,'C_gripper_m':closed,'P_gripper_m':opened,'H_home_passed':True,
        'low_height_m':lower,'height_before_low_m':before,'internal_locks':16,'PAGE_DOWN_tcp_delta_m':delta.tolist(),
        'limits':'Stair actor switch and flat stability verified; climbing the complete hotel staircase is not yet certified. Recording adapter is disabled by default, as original without record_dir.'}
    (HERE/'original_controls_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
