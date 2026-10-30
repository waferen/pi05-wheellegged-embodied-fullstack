"""Visible inspection of the containment vessel and exterior facade."""
import argparse,json,time
import glfw,mujoco,numpy as np
from PIL import Image
from hotel_runtime import HotelSimulation,HERE
from camera_calibration import set_physical_near

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='before');p.add_argument('--two-sided',action='store_true');p.add_argument('--drop',action='store_true');p.add_argument('--hide-glass',action='store_true');args=p.parse_args()
    assert glfw.init();window=glfw.create_window(1100,800,'MuJoCo - vessel and exterior inspection',None,None)
    glfw.make_context_current(window);glfw.swap_interval(1)
    sim=HotelSimulation();m,d=sim.model,sim.data;set_physical_near(m)
    if args.hide_glass:
        for i in range(m.ngeom):
            if 'AntiqueMirror' in m.geom(i).name and m.geom(i).name.endswith('_Glass_visual'):m.geom_rgba[i,3]=0
    context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150)
    scene=mujoco.MjvScene(m,maxgeom=20000);scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE]=not args.two_sided
    if args.tag=='no_shadows':scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    out=HERE/'runs/scene_repair_20261028';out.mkdir(parents=True,exist_ok=True)
    try:
        for name,look,distance,az,elev in [('vessel_front',[22.84,2.8,-.1],1.5,100,-20),
                ('vessel_back',[22.84,2.8,-.1],1.5,280,-20),('vessel_top',[22.84,2.8,-.1],1.3,90,-75),
                ('exterior',[8,7,.4],28,0,-12),('exterior_front',[8,7,.4],28,180,-12),
                ('wall_close',[19.65,15,1.2],5,180,-5)]:
            camera.lookat[:]=look;camera.distance=distance;camera.azimuth=az;camera.elevation=elev
            for _ in range(10):
                if glfw.window_should_close(window):return
                rect=mujoco.MjrRect(0,0,*glfw.get_framebuffer_size(window))
                mujoco.mjv_updateScene(m,d,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene);mujoco.mjr_render(rect,scene,context)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,args.tag+' '+name,'',context)
                glfw.swap_buffers(window);glfw.poll_events()
            rgb=np.empty((rect.height,rect.width,3),np.uint8);mujoco.mjr_readPixels(rgb,None,rect,context)
            Image.fromarray(np.flipud(rgb)).save(out/(args.tag+'_'+name+'.png'))
        if args.drop:
            bodies=[i for i in range(m.nbody) if m.body(i).name.startswith('World_C4_') and m.body_jntnum[i]>0][:2]
            camera.lookat[:]=[22.84,2.8,-.12];camera.distance=1.1;camera.azimuth=280;camera.elevation=-18
            placed=[]
            insertion_checks=[]
            body=bodies[0];j=int(m.body_jntadr[body]);q=int(m.jnt_qposadr[j])
            offset=d.xipos[body]-d.xpos[body]
            for start in (2.0,3.6):
                hits=[]
                for y in np.linspace(start,2.8,41):
                    d.qpos[q:q+3]=[22.84,y,-.08]-offset;mujoco.mj_forward(m,d)
                    hits.append(sum(1 for c in d.contact if
                        (int(m.geom_bodyid[c.geom1])==body and m.geom(int(c.geom2)).name.startswith('vessel_shell_')) or
                        (int(m.geom_bodyid[c.geom2])==body and m.geom(int(c.geom1)).name.startswith('vessel_shell_'))))
                insertion_checks.append({'start_y':start,'samples':41,'max_shell_contacts':max(hits),'clear':max(hits)==0})
            for k,body in enumerate(bodies):
                j=int(m.body_jntadr[body]);q=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j])
                # Position the actual object center, accounting for world-baked mesh offsets.
                offset=d.xipos[body]-d.xpos[body]
                d.qpos[q:q+3]=[22.74+.20*k,2.80,-.08]-offset;d.qvel[v:v+6]=0
                placed.append({'body':m.body(body).name,'id':body})
            d.qpos[:3]=[999,999,5];mujoco.mj_forward(m,d)
            early_contacts=[int(c.geom1) for c in d.contact if int(m.geom_bodyid[c.geom1]) in bodies or int(m.geom_bodyid[c.geom2]) in bodies]
            for step in range(200):
                mujoco.mj_step(m,d,nstep=10)
                mujoco.mjv_updateScene(m,d,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene);mujoco.mjr_render(rect,scene,context)
                glfw.swap_buffers(window);glfw.poll_events()
                if glfw.window_should_close(window):raise InterruptedError('Drop test closed')
            for item in placed:
                point=d.xipos[item['id']].copy();item['center']=point.tolist()
                item['inside']=bool(22.59<point[0]<23.09 and 2.62<point[1]<2.99 and -.32<point[2]<.05)
            report={'placed_objects':placed,'insertion_checks':insertion_checks,'initial_object_contacts':len(early_contacts),'finite':bool(np.isfinite(d.qpos).all()),'seconds':4.,'passed':all(p['inside'] for p in placed) and any(c['clear'] for c in insertion_checks)}
            (out/'drop_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
            rgb=np.empty((rect.height,rect.width,3),np.uint8);mujoco.mjr_readPixels(rgb,None,rect,context)
            Image.fromarray(np.flipud(rgb)).save(out/'objects_inside.png')
    finally:context.free();glfw.destroy_window(window);glfw.terminate()

if __name__=='__main__':main()
