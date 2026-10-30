"""Visible color checks of native cameras, vessel and material-only changes."""
import json,xml.etree.ElementTree as ET
import glfw,mujoco,numpy as np
from PIL import Image
from hotel_runtime import HotelSimulation,HERE
from camera_calibration import set_physical_near,configure_scene_rendering
from collection.sensors import Sensors
from original_teleop import ALL

def main():
    out=HERE/'runs/color_match_20261028';out.mkdir(exist_ok=True)
    assert glfw.init();window=glfw.create_window(1024,768,'MuJoCo - video reference color check',None,None)
    glfw.make_context_current(window);glfw.swap_interval(1)
    sim=HotelSimulation();m,d=sim.model,sim.data;set_physical_near(m)
    sim.qadr=np.array([m.joint(n).qposadr[0] for n in ALL])
    context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150)
    sensors=Sensors(sim);sensors.attach_context(context)
    scene=mujoco.MjvScene(m,maxgeom=20000);configure_scene_rendering(scene)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[22.84,2.8,-.12];camera.distance=1.1;camera.azimuth=280;camera.elevation=-18
    try:
        for _ in range(10):
            if glfw.window_should_close(window):raise InterruptedError('Color inspection stopped')
            rect=mujoco.MjrRect(0,0,*glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(m,d,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene);mujoco.mjr_render(rect,scene,context)
            glfw.swap_buffers(window);glfw.poll_events()
        rgb=np.empty((rect.height,rect.width,3),np.uint8);mujoco.mjr_readPixels(rgb,None,rect,context)
        images={'vessel':np.flipud(rgb).copy()};env=sensors.update()
        images.update({name:env.scene[key].data.output['rgb'][0].numpy() for name,key in [('head','front_rgb'),('gripper','wrist_rgb')]})
        metrics={}
        for name,rgb in images.items():
            Image.fromarray(rgb).save(out/f'after_{name}.png')
            metrics[name]={'mean':float(rgb.mean()),'white_clip_fraction':float(np.all(rgb>=250,axis=-1).mean())}
        old=ET.parse(out/'hotel_before_color.xml');new=ET.parse(HERE/'assets/hotel_b2w.xml')
        changed=[];old_mats={a.get('name'):a.attrib for a in old.findall('asset/material')}
        for mat in new.findall('asset/material'):
            if old_mats.get(mat.get('name'))!=mat.attrib:changed.append({'name':mat.get('name'),'before':old_mats.get(mat.get('name')),'after':mat.attrib})
        # No geometry, lights, cameras, contact or controller changes in this round.
        for tree in (old,new):
            for child in list(tree.find('asset')):
                if child.tag in ('material','texture'):tree.find('asset').remove(child)
        assert ET.tostring(old.getroot())==ET.tostring(new.getroot())
        report={'passed':True,'changed_materials':changed,'metrics':metrics,'geometry_lights_cameras_unchanged':True,
            'reference_video_seconds':56.05,'reference_sampled_frames':7,'note':'Viewpoint/exposure differ; these are qualitative visual checks, not per-pixel matching.'}
        (out/'verification.json').write_text(json.dumps(report,indent=2));print('Changed materials',len(changed));print(json.dumps(metrics,indent=2))
    finally:context.free();glfw.destroy_window(window);glfw.terminate()

if __name__=='__main__':main()
