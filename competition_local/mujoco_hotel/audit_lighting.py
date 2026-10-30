"""Visible before/after lighting comparison with unchanged poses and assets."""
import json
import glfw,mujoco,numpy as np
from PIL import Image
from hotel_runtime import HotelSimulation,HERE
from lighting import apply_lighting,settings
from camera_calibration import set_physical_near,configure_scene_rendering
from collection.sensors import Sensors
from original_teleop import ALL

def main():
    assert glfw.init();window=glfw.create_window(1024,768,'MuJoCo - brightness comparison',None,None)
    if not window:raise RuntimeError('Visible window unavailable')
    glfw.make_context_current(window);glfw.swap_interval(1)
    sim=HotelSimulation();m,d=sim.model,sim.data;set_physical_near(m)
    sim.qadr=np.array([m.joint(name).qposadr[0] for name in ALL])
    context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150)
    scene=mujoco.MjvScene(m,maxgeom=20000);configure_scene_rendering(scene)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[22.84,2.8,-.12];camera.distance=1.1;camera.azimuth=280;camera.elevation=-18
    sensors=Sensors(sim);sensors.attach_context(context)
    out=HERE/'runs/brightness_20261028';out.mkdir(exist_ok=True)
    before={'headlight_ambient':[.25]*3,'headlight_diffuse':[.45]*3,'headlight_specular':[.1]*3,
            'scene_ambient':[0.]*3,'scene_diffuse':[.4]*3}
    report={'new_settings':settings(),'same_pose_and_materials':True,'images':{}}
    try:
        for tag,profile in [('before',before),('after',settings())]:
            apply_lighting(m,profile)
            for _ in range(8):
                if glfw.window_should_close(window):raise InterruptedError('Lighting audit stopped')
                rect=mujoco.MjrRect(0,0,*glfw.get_framebuffer_size(window))
                mujoco.mjv_updateScene(m,d,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
                mujoco.mjr_render(rect,scene,context);glfw.swap_buffers(window);glfw.poll_events()
            rgb=np.empty((rect.height,rect.width,3),np.uint8);mujoco.mjr_readPixels(rgb,None,rect,context)
            images={'vessel':np.flipud(rgb).copy()}
            env=sensors.update()
            images.update({name:env.scene[key].data.output['rgb'][0].numpy() for name,key in [('head','front_rgb'),('gripper','wrist_rgb')]})
            for name,rgb in images.items():
                Image.fromarray(rgb).save(out/f'{tag}_{name}.png')
                region=rgb[int(len(rgb)*.2):int(len(rgb)*.85),int(rgb.shape[1]*.2):int(rgb.shape[1]*.8)]
                report['images'][tag+'_'+name]={'mean_rgb':float(region.mean()),
                    'white_clip_fraction':float(np.mean(np.all(region>=250,axis=-1))),
                    'dark_fraction':float(np.mean(region.mean(-1)<30))}
        assert report['images']['after_vessel']['mean_rgb']>report['images']['before_vessel']['mean_rgb']*1.3
        assert report['images']['after_vessel']['white_clip_fraction']<.05
        report['passed']=True;(out/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:context.free();glfw.destroy_window(window);glfw.terminate()

if __name__=='__main__':main()
