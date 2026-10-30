"""Two-joint goal conditioning in the real migrated Piper rigid-body model."""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from build_scene import HOME,OUTPUT,build

class PiperReachEnv(gym.Env):
    metadata={'render_modes':['rgb_array','human'],'render_fps':25}
    def __init__(self,render_mode=None):
        super().__init__()
        if not OUTPUT.exists():build()
        self.model=mujoco.MjModel.from_xml_path(str(OUTPUT))
        self.data=mujoco.MjData(self.model)
        self.goal_data=mujoco.MjData(self.model)
        self.render_mode=render_mode
        self.action_space=spaces.Box(-1.0,1.0,(2,),np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(8,),np.float32)
        self.home=np.asarray(HOME,dtype=float)
        self.joint_ids=np.array([0,1])
        self.tcp=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'tcp')
        self.target_site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'target')
        self.renderer=None; self.viewer=None
        self.horizon=100
        self.target=np.zeros(2)
    def _obs(self):
        q=self.data.qpos[:2]
        return np.concatenate([q,self.data.qvel[:2]*0.1,self.target,self.target-q]).astype(np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        self.data.qpos[:2]+=self.np_random.uniform([-.12,-.08],[.12,.08])
        self.data.ctrl[:]=self.data.qpos
        self.target=self.home[:2]+self.np_random.uniform([-.6,-.3],[.6,.3])
        if options and 'target' in options:self.target=np.asarray(options['target'],dtype=float)
        self.goal_data.qpos[:]=self.home; self.goal_data.qpos[:2]=self.target
        mujoco.mj_forward(self.model,self.goal_data)
        self.goal_xyz=self.goal_data.site_xpos[self.tcp].copy()
        self.model.site_pos[self.target_site]=self.goal_xyz
        mujoco.mj_forward(self.model,self.data)
        self.steps=0; self.settled=0
        return self._obs(),self._info()
    def _info(self):
        err=float(np.linalg.norm(self.target-self.data.qpos[:2]))
        dist=float(np.linalg.norm(self.goal_xyz-self.data.site_xpos[self.tcp]))
        return {'joint_error_rad':err,'tcp_error_m':dist,'is_success':bool(err<.06 and dist<.025),'steps':self.steps}
    def step(self,action):
        action=np.clip(np.asarray(action,dtype=float),-1,1)
        if not np.isfinite(action).all():raise ValueError('Non-finite action')
        before=np.linalg.norm(self.target-self.data.qpos[:2])
        # 25 Hz delta position commands; remaining six joints hold their home position.
        self.data.ctrl[:]=self.home
        self.data.ctrl[:2]=np.clip(self.data.qpos[:2]+action*.06,[-.8,.5],[.8,1.5])
        mujoco.mj_step(self.model,self.data,nstep=20)
        self.steps+=1
        if not np.isfinite(self.data.qpos).all():raise RuntimeError('Non-finite physics state')
        info=self._info()
        reward=12*(before-info['joint_error_rad'])-info['joint_error_rad']-.005*float(action@action)
        self.settled=self.settled+1 if info['is_success'] else 0
        reward+=.3 if info['is_success'] else 0
        terminated=self.settled>=8
        truncated=self.steps>=self.horizon
        if self.render_mode=='human':self.render()
        return self._obs(),float(reward),terminated,truncated,info
    def render(self):
        if self.render_mode=='human':
            from mujoco import viewer
            if self.viewer is None:
                self.viewer=viewer.launch_passive(self.model,self.data)
                self.viewer.cam.distance=1.3; self.viewer.cam.azimuth=135; self.viewer.cam.elevation=-22
                self.viewer.cam.lookat[:]=[.15,0,.25]
            self.viewer.sync(); return None
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=600,width=800)
        camera=mujoco.MjvCamera()
        camera.distance=1.3; camera.azimuth=135; camera.elevation=-22; camera.lookat[:]=[.15,0,.25]
        self.renderer.update_scene(self.data,camera)
        return self.renderer.render()
    def close(self):
        if self.renderer is not None:self.renderer.close(); self.renderer=None
        if self.viewer is not None:self.viewer.close(); self.viewer=None
