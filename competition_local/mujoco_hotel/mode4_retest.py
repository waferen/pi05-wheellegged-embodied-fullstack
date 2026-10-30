"""Repeat low-grasp transitions in a visible window, without the VLA policy."""
import argparse,json,time
from pathlib import Path
import glfw,mujoco,numpy as np
from PIL import Image
from original_teleop import OriginalTeleop,HERE,ALL
from camera_calibration import set_physical_near


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',default='baseline')
    p.add_argument('--height-input',choices=['relative','world'],default='relative')
    p.add_argument('--cases',nargs='+',default=['fresh','standing','after_high'])
    p.add_argument('--no-lock',action='store_true')
    p.add_argument('--source-arm-gains',action='store_true')
    p.add_argument('--hold-seconds',type=float,default=0.)
    args=p.parse_args()
    out=HERE/'runs/mode4_retest_20261028'/args.label;out.mkdir(parents=True,exist_ok=True)
    assert glfw.init()
    window=glfw.create_window(1100,750,'MuJoCo - Mode 4 retest',None,None)
    if not window:raise RuntimeError('Visible window unavailable')
    glfw.make_context_current(window);glfw.swap_interval(1)
    report={'label':args.label,'height_input':args.height_input,'visible_gui':True,'cases':[]}
    context=None
    try:
        sim=OriginalTeleop();m,d=sim.model,sim.data;set_physical_near(m)
        if args.source_arm_gains:
            for name in ('arm_joint2','arm_joint3'):
                actuator=m.actuator(name+'_drive').id
                m.actuator_gainprm[actuator,0]=800.;m.actuator_biasprm[actuator,1]=-800.
                m.dof_damping[m.joint(name).dofadr[0]]=13.
        original_observation=sim.observation
        signature=mujoco.mjtState.mjSTATE_INTEGRATION
        initial_state=np.empty(mujoco.mj_stateSize(m,signature));mujoco.mj_getState(m,d,initial_state,signature)
        initial_goals=sim.goals.copy()
        context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150)
        for case in args.cases:
            sim.release_lock();mujoco.mj_setState(m,d,initial_state,signature);sim.goals=initial_goals.copy()
            sim.mode='walk';sim.target_height=.59;sim.home_motion=None;sim.keys.clear();sim.last_action[:]=0.
            sim.ik_target=None;sim.ik_velocity[:]=0.;mujoco.mj_forward(m,d)
            scene=mujoco.MjvScene(m,maxgeom=12000);option=mujoco.MjvOption();option.geomgroup[3]=0
            camera=sim.camera(False);camera.azimuth=90;camera.elevation=-8
            if args.height_input=='world':
                def observation(command,stabilizer=False):
                    obs=original_observation(command,stabilizer)
                    if stabilizer:obs[0,-1]=float(d.qpos[2])
                    return obs
                sim.observation=observation
            if args.no_lock:
                sim.update_low_lock=lambda dt:None
            records=[];start=None
            def frame(label,snapshot=None):
                if glfw.window_should_close(window) or glfw.get_key(window,glfw.KEY_ESCAPE)==glfw.PRESS:
                    raise InterruptedError('Retest stopped')
                ground=sim.ground_height();height=float(d.xpos[sim.base,2]-(sim.floor_z if ground is None else ground))
                tilt=float(np.degrees(np.arccos(np.clip(d.xmat[sim.base].reshape(3,3)[2,2],-1,1))))
                camera.lookat[:]=d.xpos[sim.base]+[0,0,.02]
                width,hei=glfw.get_framebuffer_size(window);rect=mujoco.MjrRect(0,0,width,hei)
                mujoco.mjv_updateScene(m,d,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
                mujoco.mjr_render(rect,scene,context)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,
                    f'{args.label} | {case} | {label}\nmode {sim.mode} | {sim.lock_phase} | height {height:.4f} / {sim.target_height:.3f} m | tilt {tilt:.2f} deg',
                    'Original stabilizer checkpoint | no PI0.5 | Esc exits',context)
                if snapshot:
                    rgb=np.empty((hei,width,3),np.uint8);mujoco.mjr_readPixels(rgb,None,rect,context)
                    Image.fromarray(np.flipud(rgb)).save(out/f'{case}_{snapshot}.png')
                glfw.swap_buffers(window);glfw.poll_events()
                return height,tilt
            def advance(count,label):
                for i in range(count):
                    tick=time.perf_counter();sim.control_frame(.02,command_override=np.zeros(3))
                    height,tilt=frame(label)
                    if start is not None:
                        records.append({'t':float(d.time-start),'height':height,'tilt_deg':tilt,'phase':sim.lock_phase,
                            'root_world':d.xpos[sim.base].tolist(),'ground_z':sim.ground_height(),
                            'q':d.qpos[sim.qadr].tolist(),'dq':d.qvel[sim.vadr].tolist(),
                            'actions':sim.last_action.tolist(),'ctrl':d.ctrl[sim.act].tolist(),
                            'contacts':int(d.ncon),'min_contact_dist':float(min((c.dist for c in d.contact),default=0.))})
                    time.sleep(max(0,.02-(time.perf_counter()-tick)))
            frame('Initial state','initial')
            if case in ('standing','after_high'):advance(50,'Walk zero command warm-up')
            if case=='after_high':
                sim.set_mode('high_grasp');advance(55,'High grasp hold')
                sim.set_mode('walk');advance(56,'Walk / arm home')
            before_height,before_tilt=frame('Before mode 4','before')
            sim.key('4');sim.key('4',False);start=float(d.time)
            first_obs=sim.observation(np.zeros(3),True).numpy()[0].tolist()
            advance(50,'Mode 4 lowering')
            frame('One second after switching','lock')
            advance(100,'Mode 4 hold')
            final_height,final_tilt=frame('Mode 4 final','final')
            summary={'case':case,'before_height':before_height,'before_tilt':before_tilt,
                'final_height':final_height,'final_tilt':final_tilt,'target':sim.target_height,
                'lock_reason':sim.lock_reason,'lock_elapsed':sim.lock_elapsed,'lock_metrics':sim.lock_metrics,
                'first_obs75':first_obs,'finite':bool(np.isfinite(d.qpos).all()),
                'ground_hit':sim.ground_hit,
                'trace':records,'joint_order':ALL}
            report['cases'].append(summary)
            (out/'results.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in summary.items() if k not in ('trace','first_obs75','joint_order')},indent=2),flush=True)
        until=time.monotonic()+args.hold_seconds
        while time.monotonic()<until:frame('Final state; physics paused')
    finally:
        if context is not None:context.free()
        glfw.destroy_window(window);glfw.terminate()


if __name__=='__main__':main()
