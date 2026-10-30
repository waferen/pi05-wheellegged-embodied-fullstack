"""Existing two calibrated cameras and source-compatible measured state."""
from types import SimpleNamespace as NS
import mujoco,numpy as np,torch
from hotel_runtime import actor_frame_velocity
from camera_calibration import configure_scene_rendering


class Sensors:
    def __init__(self,sim):
        self.sim=sim;self.context=None
        self.scene=mujoco.MjvScene(sim.model,maxgeom=12000)
        configure_scene_rendering(self.scene)
        self.option=mujoco.MjvOption();self.option.geomgroup[3]=0
        self.camera=mujoco.MjvCamera();self.camera.type=mujoco.mjtCamera.mjCAMERA_FIXED
        self.ids=[next(i for i in range(sim.model.ncam) if sim.model.camera(i).name.endswith(suffix))
                  for suffix in ('HeadCamera_HeadCamera','GripperCamera_GripperCamera')]
        self.env=NS(step_dt=.02,scene={'robot':NS(data=None)})

    def attach_context(self,context):
        self.context=context;mujoco.mjr_resizeOffscreen(640,480,context)

    def update(self):
        if self.context is None:raise RuntimeError('Visible renderer is required before recording')
        sim=self.sim;m,d=sim.model,sim.data
        velocity=actor_frame_velocity(m,d,sim.base)
        self.env.scene['robot'].data=NS(root_lin_vel_b=torch.tensor(velocity[3:][None],dtype=torch.float32),
            root_ang_vel_b=torch.tensor(velocity[:3][None],dtype=torch.float32),
            joint_pos=torch.tensor(d.qpos[sim.qadr[16:24]][None],dtype=torch.float32))
        viewport=mujoco.MjrRect(0,0,640,480)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN,self.context)
        try:
            for name,cid in zip(('front_rgb','wrist_rgb'),self.ids):
                self.camera.fixedcamid=cid
                mujoco.mjv_updateScene(m,d,self.option,None,self.camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
                mujoco.mjr_render(viewport,self.scene,self.context)
                rgb=np.empty((480,640,3),np.uint8);mujoco.mjr_readPixels(rgb,None,viewport,self.context)
                self.env.scene[name]=NS(data=NS(output={'rgb':torch.from_numpy(np.flipud(rgb).copy()[None])}))
        finally:mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW,self.context)
        return self.env
