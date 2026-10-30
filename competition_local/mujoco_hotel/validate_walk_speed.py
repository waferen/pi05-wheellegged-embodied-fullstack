"""Validate the faster walk command on the existing upper hotel corridor."""
import json,time
import numpy as np
import mujoco
from original_teleop import OriginalTeleop,HERE

def advance(sim,frames):
    minimum=1.
    for _ in range(frames):
        sim.control_frame()
        minimum=min(minimum,float(sim.data.xmat[sim.base].reshape(3,3)[2,2]))
        assert np.isfinite(sim.data.qpos).all()
    return minimum

def main():
    sim=OriginalTeleop()
    sim.data.qpos[:3]=[14,7,.66];mujoco.mj_forward(sim.model,sim.data)
    advance(sim,250)
    initial=sim.data.xpos[sim.base].copy();sim.key('W')
    minimum=advance(sim,200);after_forward=sim.data.xpos[sim.base].copy()
    forward_distance=float(np.linalg.norm(after_forward[:2]-initial[:2]))
    sim.key('W',False);advance(sim,1);assert sim.command[0]==0.
    sim.key('S');back_minimum=advance(sim,100);after_back=sim.data.xpos[sim.base].copy()
    sim.key('S',False);sim.key('A');turn_minimum=advance(sim,100)
    sim.key('A',False);sim.key('SPACE');advance(sim,1)
    assert np.allclose(sim.command,0.)
    sim.key('2');sim.key('W');advance(sim,1);stair_command=float(sim.command[0]);assert stair_command==.25
    result={'checked_at':time.time(),'settings':sim.speed_settings,'forward_seconds':4.,'forward_distance_m':forward_distance,
        'measured_mean_forward_m_s':forward_distance/4,'forward_upright_min':minimum,'backward_upright_min':back_minimum,
        'turn_upright_min':turn_minimum,'release_and_space_stop_passed':True,'stair_command_m_s':stair_command,
        'scope':'Upper hotel corridor, forward/backward and turn in place; no change to original keyboard meanings.'}
    result['passed']=bool(forward_distance/4>.5 and min(minimum,back_minimum,turn_minimum)>.95)
    (HERE/'walk_speed_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));assert result['passed']

if __name__=='__main__':main()
