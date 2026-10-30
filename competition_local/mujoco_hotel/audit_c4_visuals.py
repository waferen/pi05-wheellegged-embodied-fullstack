"""Capture visible MuJoCo C4 material checks at deterministic native poses."""
import argparse
import glfw,mujoco,numpy as np
from PIL import Image,ImageDraw
from hotel_runtime import HotelSimulation,HERE
from camera_calibration import set_physical_near,configure_scene_rendering


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='after');args=parser.parse_args()
    out=HERE/'runs/c4_uv_20261028';out.mkdir(parents=True,exist_ok=True)
    assert glfw.init()
    window=glfw.create_window(900,700,'MuJoCo - C4 source texture inspection',None,None)
    glfw.make_context_current(window);glfw.swap_interval(1)
    sim=HotelSimulation();m,d=sim.model,sim.data;set_physical_near(m)
    context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150)
    scene=mujoco.MjvScene(m,maxgeom=20000);configure_scene_rendering(scene)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    try:
        images=[]
        for body in ['C4_1','C4_2','C4_3','C4_4','easy_C4_1','easy_C4_2']:
            bid=m.body('World_C4_'+body).id
            camera.lookat[:]=d.xipos[bid];camera.distance=.46;camera.azimuth=130;camera.elevation=-24
            for _ in range(5):
                if glfw.window_should_close(window):raise InterruptedError('Inspection stopped')
                rect=mujoco.MjrRect(0,0,*glfw.get_framebuffer_size(window))
                mujoco.mjv_updateScene(m,d,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
                mujoco.mjr_render(rect,scene,context);glfw.swap_buffers(window);glfw.poll_events()
            rgb=np.empty((rect.height,rect.width,3),np.uint8)
            mujoco.mjr_readPixels(rgb,None,rect,context)
            image=Image.fromarray(np.flipud(rgb));image.save(out/f'{args.tag}_{body}.png')
            thumb=image.resize((450,350));ImageDraw.Draw(thumb).text((10,10),body,fill='white',stroke_width=1,stroke_fill='black');images.append(thumb)
        sheet=Image.new('RGB',(1350,700))
        for i,image in enumerate(images):sheet.paste(image,((i%3)*450,(i//3)*350))
        sheet.save(out/f'{args.tag}_all.png')
        print('Saved visible C4 views:',args.tag)
    finally:context.free();glfw.destroy_window(window);glfw.terminate()


if __name__=='__main__':main()
