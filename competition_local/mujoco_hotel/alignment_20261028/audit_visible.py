"""Visible GLFW audit: materials, controls, sensors and real checkpoint inference.

Closes after --hold-seconds, or Esc. All physics integration has a visible window.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import glfw
import mujoco
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from pi05_teleop import Pi05Teleop
from original_teleop import OriginalTeleop,ARMS
from camera_calibration import set_physical_near


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--policy-steps',type=int,default=100)
    parser.add_argument('--hold-seconds',type=float,default=60.)
    args=parser.parse_args()
    assert glfw.init()
    window=glfw.create_window(1200,800,'MuJoCo - interface and material audit',None,None)
    if not window:raise RuntimeError('Visible audit window unavailable')
    glfw.make_context_current(window);glfw.swap_interval(1)
    sim=None;context=None
    report={'passed':False,'visible_gui':True,'started':time.time()}
    def aborted():
        return glfw.window_should_close(window) or glfw.get_key(window,glfw.KEY_ESCAPE)==glfw.PRESS
    try:
        sim=Pi05Teleop(port=args.port,run_dir=HERE/'policy_run');m,d=sim.model,sim.data;set_physical_near(m)
        context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150);sim.attach_context(context)
        scene=mujoco.MjvScene(m,maxgeom=12000);option=mujoco.MjvOption();option.geomgroup[3]=0
        camera=sim.camera(False)
        def render(label,snapshot=None):
            if aborted():raise InterruptedError('Visible audit stopped')
            width,height=glfw.get_framebuffer_size(window);rect=mujoco.MjrRect(0,0,width,height)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW,context)
            mujoco.mjv_updateScene(m,d,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            mujoco.mjr_render(rect,scene,context)
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,
                label,'Esc exits | visible audit | original USD and checkpoint unchanged',context)
            if snapshot:
                rgb=np.empty((height,width,3),dtype=np.uint8);mujoco.mjr_readPixels(rgb,None,rect,context)
                Image.fromarray(np.flipud(rgb)).save(HERE/snapshot)
            glfw.swap_buffers(window);glfw.poll_events()
        for suffix in ('trash_can03','trash_can05'):
            ids=[i for i in range(m.ngeom) if suffix in m.geom(i).name and m.geom_group[i]==0]
            assert ids
            centers=d.geom_xpos[ids];camera.lookat[:]=centers.mean(axis=0);camera.distance=1.6;camera.azimuth=100;camera.elevation=-20
            for _ in range(20):render('Black trash can: '+suffix)
            render('Black trash can: '+suffix,suffix+'.png')
        camera=sim.camera(False)
        render('Wall colors / robot baseline','walls_and_robot.png')
        wall=m.geom('World_Guest_Room_Room_05_room_Bashroom_Cube_05_visual').id
        center=d.geom_xpos[wall]+d.geom_xmat[wall].reshape(3,3)@m.geom_aabb[wall,:3]
        thin=int(np.argmin(m.geom_aabb[wall,3:]));normal=d.geom_xmat[wall].reshape(3,3)[:,thin]
        camera.lookat[:]=center;camera.distance=1.2;camera.elevation=-5
        camera.azimuth=float(np.degrees(np.arctan2(normal[1],normal[0])))
        for _ in range(20):render('Selected Plaster_Wall_Weathered variant')
        render('Selected Plaster_Wall_Weathered variant','wall_plaster.png')
        camera=sim.camera(False)
        # Capture actual RGB sensors through the exact production observation path.
        env=sim._environment(True);obs=sim.client.observation(env,sim.mode)
        report['sensors']={key:{'shape':list(obs[key].shape),'dtype':str(obs[key].dtype),'std':float(obs[key].std())}
            for key in ('observation/image','observation/wrist_image')}
        assert all(v['std']>5 for v in report['sensors'].values())
        for label,image in zip(('head','wrist'),sim.images):Image.fromarray(image).save(HERE/(label+'_rgb.png'))
        sim.set_model_control(False)
        # Save a settled initial state, restore after mechanical contract probes.
        signature=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(m,signature));mujoco.mj_getState(m,d,state,signature)
        goals=sim.goals.copy()
        sim.set_mode('high_grasp');assert sim.arm_ready
        assert sum(d.eq_active)==16
        sim.key('C');sim.key('C',False)
        for _ in range(25):OriginalTeleop.control_frame(sim,.02);render('High grasp: close gripper')
        closed=d.qpos[sim.qadr[22:24]].copy()
        sim.key('P');sim.key('P',False)
        for _ in range(30):OriginalTeleop.control_frame(sim,.02);render('High grasp: open gripper')
        opened=d.qpos[sim.qadr[22:24]].copy()
        assert opened[0]>closed[0]+.02 and opened[1]<closed[1]-.02
        sim.set_mode('walk')
        for _ in range(56):OriginalTeleop.control_frame(sim,.02,command_override=np.zeros(3));render('Return to walk: arm home')
        assert sim.home_motion is None and max(abs(sim.goals[n]) for n in ARMS[:6])<1e-8
        sim.set_mode('low_grasp')
        for _ in range(55):OriginalTeleop.control_frame(sim,.02);render('Low grasp: height / tilt lock')
        assert sim.lock_phase=='locked' and .5-1e-9<=sim.lock_elapsed<=1.02
        report['mechanical']={'closed_fingers_m':closed.tolist(),'opened_fingers_m':opened.tolist(),
            'home_targets_zero':True,'low_lock_seconds':sim.lock_elapsed,'low_lock_reason':sim.lock_reason,
            'low_lock_metrics':sim.lock_metrics,'active_internal_locks':int(sum(d.eq_active))}
        report['mechanical']['low_pose_target_reached']=sim.lock_reason=='criteria'
        report['full_task_success_evaluated']=False
        sim.release_lock();mujoco.mj_setState(m,d,state,signature);sim.goals=goals.copy()
        sim.mode='walk';sim.home_motion=None;sim.keys.clear();sim.last_action[:]=0.;sim.ik_velocity[:]=0.
        mujoco.mj_forward(m,d);sim.client.reset();sim.set_model_control(True)
        initial_steps=sim.executed_steps;started=time.monotonic()
        while sim.executed_steps-initial_steps<args.policy_steps:
            if time.monotonic()-started>120:raise TimeoutError('Policy audit exceeded 120 seconds')
            sim.control_frame(.02)
            camera.lookat[:]=d.xpos[sim.base]+[0,0,.05]
            render(f'PI0.5 checkpoint 40000 | executed {sim.executed_steps} | phase {sim.client.task_phase}')
        sim.set_model_control(False)
        report['policy']=sim.policy_status();report['policy']['wall_seconds']=time.monotonic()-started
        assert sim.executed_steps-initial_steps>=args.policy_steps and not sim.last_error
        report['finite_state']=bool(np.isfinite(d.qpos).all())
        report['passed']=report['finite_state'];report['finished']=time.time()
        (HERE/'visible_results.json').write_text(json.dumps(report,indent=2))
        render('PASS: interface / materials / real model. Physics paused. Esc exits.','policy_audit.png')
        print(json.dumps(report,indent=2),flush=True)
        until=time.monotonic()+args.hold_seconds
        while time.monotonic()<until and not aborted():render('PASS: interface / materials / real model. Physics paused. Esc exits.')
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        (HERE/'visible_results.json').write_text(json.dumps(report,indent=2));raise
    finally:
        if sim is not None:sim.close()
        if context is not None:context.free()
        glfw.destroy_window(window);glfw.terminate()


if __name__=='__main__':main()
