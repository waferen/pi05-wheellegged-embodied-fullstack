"""Original robot keys; camera controls are mouse-operated window buttons."""
import json,os,time
import glfw,mujoco,numpy as np
from PIL import Image
from original_teleop import OriginalTeleop,HERE
from camera_calibration import sensor_viewport,set_physical_near,dual_sensor_viewports,configure_scene_rendering
SPECIAL={glfw.KEY_SPACE:'SPACE',glfw.KEY_ESCAPE:'ESCAPE',glfw.KEY_UP:'UP',glfw.KEY_DOWN:'DOWN',glfw.KEY_LEFT:'LEFT',glfw.KEY_RIGHT:'RIGHT',glfw.KEY_PAGE_UP:'PAGE_UP',glfw.KEY_PAGE_DOWN:'PAGE_DOWN',glfw.KEY_MINUS:'-',glfw.KEY_EQUAL:'=',glfw.KEY_F1:'F1',glfw.KEY_F2:'F2',glfw.KEY_F3:'F3',glfw.KEY_F4:'F4',glfw.KEY_F5:'F5'}
SPECIAL[glfw.KEY_F6]='F6'
def key_name(key):
    if key in SPECIAL:return SPECIAL[key]
    if glfw.KEY_A<=key<=glfw.KEY_Z or glfw.KEY_0<=key<=glfw.KEY_9:return chr(key)
    if glfw.KEY_KP_0<=key<=glfw.KEY_KP_9:return str(key-glfw.KEY_KP_0)
    return None

def main(initial_view='Overview',sim_factory=None):
    if not glfw.init():raise RuntimeError('GLFW initialization failed')
    glfw.window_hint(glfw.SAMPLES,4)
    glfw.window_hint(glfw.MAXIMIZED,glfw.TRUE)
    window=glfw.create_window(1280,850,'MuJoCo - PPSUC Hotel | original controls',None,None)
    if not window:raise RuntimeError('Window creation failed')
    glfw.make_context_current(window);glfw.swap_interval(1)
    factory=sim_factory or OriginalTeleop
    sim=factory();set_physical_near(sim.model);scene=mujoco.MjvScene(sim.model,maxgeom=12000)
    configure_scene_rendering(scene)
    context=mujoco.MjrContext(sim.model,mujoco.mjtFontScale.mjFONTSCALE_150)
    if hasattr(sim,'attach_context'):sim.attach_context(context)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=sim.camera(True);view='Overview';reset_requested=False;fill_camera=True;show_help=False
    mouse={'x':0.,'y':0.,'left':False,'right':False};labels=['Follow','Overview','Head','Hand','Reset','Walk -','Walk +','Frame','Help','Dual']
    def select_view(label):
        nonlocal view,reset_requested,fill_camera,show_help
        if label=='Frame':fill_camera=not fill_camera;return
        if label=='Help':show_help=not show_help;return
        if label=='Reset':reset_requested=True;return
        if label in ('Walk -','Walk +'):sim.adjust_walk_speed(-.1 if label=='Walk -' else .1);return
        view=label
        if label=='Dual':return
        if label in ('Head','Hand'):
            query='HeadCamera_HeadCamera' if label=='Head' else 'GripperCamera_GripperCamera'
            ids=[i for i in range(sim.model.ncam) if query in sim.model.camera(i).name]
            if ids:camera.type=mujoco.mjtCamera.mjCAMERA_FIXED;camera.fixedcamid=ids[0]
        else:
            preset=sim.camera(label=='Overview');camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=preset.lookat;camera.distance=preset.distance;camera.azimuth=preset.azimuth;camera.elevation=preset.elevation
    def keyboard(_window,key,scan,action,mods):
        value=key_name(key)
        if value:sim.key(value,pressed=action!=glfw.RELEASE,repeat=action==glfw.REPEAT)
    def focus(_window,focused):
        if not focused:sim.keys.clear();sim.ik_target=None;sim.ik_velocity[:]=0.
    def button(_window,button,action,mods):
        x,y=glfw.get_cursor_pos(window)
        if button==glfw.MOUSE_BUTTON_LEFT and action==glfw.PRESS and 8<=y<=38:
            for i,label in enumerate(labels):
                if 10+i*104<=x<=108+i*104:select_view(label);return
        mouse['left']=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS
        mouse['right']=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS
        mouse['x'],mouse['y']=x,y
    def move(_window,x,y):
        dx,dy=x-mouse['x'],y-mouse['y'];mouse['x'],mouse['y']=x,y
        if not mouse['left'] and not mouse['right']:return
        if view=='Dual':return
        if camera.type!=mujoco.mjtCamera.mjCAMERA_FREE:select_view('Follow')
        height=max(1,glfw.get_window_size(window)[1])
        action=mujoco.mjtMouse.mjMOUSE_MOVE_V if mouse['right'] else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        mujoco.mjv_moveCamera(sim.model,action,dx/height,dy/height,camera)
    def scroll(_window,xoffset,yoffset):
        if view!='Dual' and camera.type==mujoco.mjtCamera.mjCAMERA_FREE:mujoco.mjv_moveCamera(sim.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*yoffset,camera)
    glfw.set_key_callback(window,keyboard);glfw.set_window_focus_callback(window,focus)
    glfw.set_mouse_button_callback(window,button);glfw.set_cursor_pos_callback(window,move);glfw.set_scroll_callback(window,scroll)
    select_view(initial_view)
    last=time.perf_counter();accumulator=0.;last_status=0.;frames=0
    try:
        while not glfw.window_should_close(window) and not sim.quit:
            frame_start=time.perf_counter();elapsed=min(.1,frame_start-last);last=frame_start;accumulator+=elapsed
            glfw.poll_events()
            if reset_requested:
                if hasattr(sim,'reset_scene'):sim.reset_scene()
                else:
                    if hasattr(sim,'close'):sim.close()
                    sim=factory();set_physical_near(sim.model);scene=mujoco.MjvScene(sim.model,maxgeom=12000)
                    configure_scene_rendering(scene)
                    context.free();context=mujoco.MjrContext(sim.model,mujoco.mjtFontScale.mjFONTSCALE_150)
                    if hasattr(sim,'attach_context'):sim.attach_context(context)
                select_view('Dual' if hasattr(sim,'collection_status') else 'Follow');reset_requested=False;accumulator=0.
            while accumulator>=.02:
                if sim.control_frame(.02) is False:
                    accumulator=0.;break
                accumulator-=.02
            if view=='Follow':camera.lookat[:]=sim.data.xpos[sim.base]+[0,0,.05]
            width,height=glfw.get_framebuffer_size(window)
            if width<1 or height<1:time.sleep(.02);continue
            viewport=mujoco.MjrRect(0,0,width,height)
            scale=width/max(1,glfw.get_window_size(window)[0])
            render_viewport=viewport
            dual_viewports=[]
            if view in ('Head','Hand'):
                # A 4:3 projection covers the window, cropping rather than distorting.
                # Fit mode remains available to inspect the complete sensor image.
                render_viewport=mujoco.MjrRect(*sensor_viewport(width,height,fill=fill_camera))
                mujoco.mjr_rectangle(viewport,.025,.03,.035,1.)
            if view=='Dual':
                mujoco.mjr_rectangle(viewport,.025,.03,.035,1.)
                rectangles=dual_sensor_viewports(width,height,header=int(150*scale))
                for label,suffix,rect in zip(('HEAD CAMERA','GRIPPER CAMERA'),
                        ('HeadCamera_HeadCamera','GripperCamera_GripperCamera'),rectangles):
                    sensor=mujoco.MjvCamera();sensor.type=mujoco.mjtCamera.mjCAMERA_FIXED
                    sensor.fixedcamid=next(i for i in range(sim.model.ncam) if sim.model.camera(i).name.endswith(suffix))
                    sensor_rect=mujoco.MjrRect(*rect)
                    mujoco.mjv_updateScene(sim.model,sim.data,option,None,sensor,mujoco.mjtCatBit.mjCAT_ALL,scene)
                    mujoco.mjr_render(sensor_rect,scene,context)
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,sensor_rect,label,'',context)
                    dual_viewports.append({'camera':label,'viewport':list(rect)})
            else:
                mujoco.mjv_updateScene(sim.model,sim.data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
                mujoco.mjr_render(render_viewport,scene,context)
            for i,label in enumerate(labels):
                rect=mujoco.MjrRect(int((10+i*104)*scale),height-int(38*scale),int(98*scale),int(30*scale))
                color=(.12,.33,.48,.96) if view==label else (.12,.15,.18,.96)
                mujoco.mjr_rectangle(rect,*color)
                caption=('Fit 4:3' if fill_camera else 'Fill') if label=='Frame' else ('Hide help' if show_help else 'Help') if label=='Help' else label
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,' '+caption,'',context)
            text=(f'MODE {sim.mode.upper()} | walk {sim.speed_settings["walk_forward_m_s"]:.2f}m/s | lock {sim.lock_phase} | height {sim.data.qpos[2]-sim.floor_z:.3f}m\n'
                '1 Walk   2 Stair   4 Low grasp   5 High grasp\n'
                'W/S forward/back   A/D turn   Space stop   Q/Esc quit\n'
                'Arm: R/T E/G V/B N/M I/O K/L | C close P open H home\n'
                'IK: arrows + PageUp/Down; 7/8 9/0 -/= rotation\n'+sim.message)
            if not show_help:text=text.split('\n',1)[0]
            if hasattr(sim,'collection_status'):
                collection=sim.collection_status()
                text+=f"\nREC {'ON' if collection['recording'] else 'OFF'} | stage {collection['stage']}/14 | frames {collection['frames']} | 50 Hz sim time"
                text+='\nF1 start | F2 next stage | F3 save (unreviewed) | F6 interrupt | Reset starts new scene'
            if hasattr(sim,'policy_status'):
                policy=sim.policy_status()
                text+='\nPI0.5 '+policy.get('checkpoint_label','40000')+' | '+policy['control'].upper()+f" | phase {policy['task_phase']+1} | predictions {policy['inference_count']} | actions {policy['executed_policy_steps']}"
                text+='\nF4/Space manual | F5 model | F2 phase (manual) | '+sim.message
            if sim.arm_ready:
                grip_status=sim.grip_feedback()
                left,right=grip_status['normal_force_N']
                text+=f'\nC4 GRIP L/R: {left:.1f} / {right:.1f} N'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,mujoco.MjrRect(0,0,width,height-int(48*scale)),text,'',context)
            frames+=1
            if frames==150 or (view=='Dual' and frames==10):
                rgb=np.empty((height,width,3),dtype=np.uint8);mujoco.mjr_readPixels(rgb,None,viewport,context)
                Image.fromarray(np.flipud(rgb)).save(HERE/('dual_camera_preview.png' if view=='Dual' else 'interactive_preview.png'))
            glfw.swap_buffers(window)
            if time.time()-last_status>2:
                record={'updated':time.time(),'pid':os.getpid(),'sim_time':float(sim.data.time),'base':sim.data.xpos[sim.base].tolist(),'contacts':int(sim.data.ncon),
                    'mode':sim.mode,'lock_phase':sim.lock_phase,'command':getattr(sim,'command',np.zeros(3)).tolist(),'speed_settings':sim.speed_settings,'viewer_running':True,'view':view,'original_keys':True,'source_baseline_inference':True,
                    'camera_viewport':[render_viewport.left,render_viewport.bottom,render_viewport.width,render_viewport.height],'window_framebuffer':[width,height],
                    'camera_layout':'dual-fit' if view=='Dual' else 'fill' if fill_camera else 'fit',
                    'dual_viewports':dual_viewports,'help_visible':show_help,'camera_near_m':float(sim.model.vis.map.znear*sim.model.stat.extent),
                    'grip':sim.grip_feedback()}
                if hasattr(sim,'policy_status'):record.update(sim.policy_status())
                if hasattr(sim,'collection_status'):record['collection']=sim.collection_status()
                (HERE/'viewer_status.json').write_text(json.dumps(record),encoding='utf-8');last_status=time.time()
                glfw.set_window_title(window,'MuJoCo - PPSUC Hotel | '+sim.mode+(' | PI0.5 '+policy.get('checkpoint_label','40000') if hasattr(sim,'policy_status') else ' | original controls'))
            time.sleep(max(0,1/60-(time.perf_counter()-frame_start)))
    finally:
        if hasattr(sim,'close'):sim.close()
        context.free();glfw.destroy_window(window);glfw.terminate()
        (HERE/'viewer_status.json').write_text(json.dumps({'viewer_running':False,'closed_at':time.time()}),encoding='utf-8')
if __name__=='__main__':main()
