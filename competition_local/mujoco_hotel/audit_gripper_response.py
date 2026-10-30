"""Visible empty-gripper step response using the actual source finger assembly."""
import copy,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import glfw,mujoco,numpy as np
from PIL import Image
from hotel_runtime import HERE
from original_teleop import OriginalTeleop


def main():
    source=ET.parse(HERE/'assets/hotel_b2w.xml').getroot()
    body=copy.deepcopy(source.find(".//body[@name='World_b2w_piper_gripper_base']"));body.set('pos','0 0 .5');body.set('quat','1 0 0 0')
    root=ET.Element('mujoco');root.append(copy.deepcopy(source.find('compiler')))
    ET.SubElement(root,'option',timestep='.002',integrator='implicitfast',gravity='0 0 0')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');world.append(body)
    ET.SubElement(world,'light',pos='0 0 2',directional='true')
    required=set()
    for g in body.iter('geom'):
        for key in ('mesh','material'):
            if g.get(key):required.add(g.get(key))
    for a in source.find('asset'):
        if a.tag=='material' and a.get('name') in required and a.get('texture'):required.add(a.get('texture'))
    for a in source.find('asset'):
        if a.get('name') not in required:continue
        a=copy.deepcopy(a)
        if a.get('file'):a.set('file',str(HERE/'assets'/a.get('file')))
        assets.append(a)
    act=ET.SubElement(root,'actuator')
    for n in ('arm_joint7_drive','arm_joint8_drive'):act.append(copy.deepcopy(source.find(f"actuator/*[@name='{n}']")))
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
    ids=[m.joint(n).qposadr[0] for n in ('arm_joint7','arm_joint8')]
    v=[m.joint(n).dofadr[0] for n in ('arm_joint7','arm_joint8')]
    assert glfw.init();window=glfw.create_window(800,600,'MuJoCo - gripper opening and speed check',None,None);glfw.make_context_current(window)
    context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150);scene=mujoco.MjvScene(m,maxgeom=1000)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.60];cam.distance=.40;cam.azimuth=70;cam.elevation=-25
    out=HERE/'runs/gripper_response';out.mkdir(parents=True,exist_ok=True)
    report={}
    try:
        for label,slew in [('raw',False),('limited',True)]:
            mujoco.mj_resetData(m,d);d.qpos[ids]=[.045,-.045];target=d.qpos[ids].copy();trace=[]
            adapter=object.__new__(OriginalTeleop);adapter.model=m;adapter.data=d
            adapter.act=np.r_[np.zeros(22,dtype=int),[0,1]];adapter.qadr=np.r_[np.zeros(22,dtype=int),ids]
            for k in range(600):
                requested=np.array([.045,-.045]) if k<100 or k>=350 else np.array([.001,-.001])
                d.ctrl[:]=requested
                if slew:adapter.step_physics(.002,None)
                else:mujoco.mj_step(m,d)
                trace.append({'t':float(d.time),'q':d.qpos[ids].tolist(),'v':d.qvel[v].tolist(),'requested':requested.tolist()})
                if k%5==0:
                    if glfw.window_should_close(window):raise InterruptedError('Stopped gripper check')
                    rect=mujoco.MjrRect(0,0,*glfw.get_framebuffer_size(window));mujoco.mjv_updateScene(m,d,option,None,cam,mujoco.mjtCatBit.mjCAT_ALL,scene);mujoco.mjr_render(rect,scene,context)
                    glfw.swap_buffers(window);glfw.poll_events();time.sleep(.005)
                if k in (90,340):
                    pixels=np.empty((rect.height,rect.width,3),np.uint8);mujoco.mjr_readPixels(pixels,None,rect,context);Image.fromarray(np.flipud(pixels)).save(out/f'{label}_{"open" if k==90 else "closed"}.png')
            report[label]={'max_speed_m_s':float(np.abs([r['v'] for r in trace]).max()),'closed_q_m':trace[340]['q'],
                'open_q_m':trace[-1]['q'],'close_to_95pct_seconds':next(r['t']-.2 for r in trace[100:350] if r['q'][0]<=.001+.044*.05),'trace':trace}
        (out/'response.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{a:b for a,b in x.items() if a!='trace'} for k,x in report.items()},indent=2))
    finally:context.free();glfw.destroy_window(window);glfw.terminate()


if __name__=='__main__':main()
