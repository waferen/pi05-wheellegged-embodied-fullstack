"""Shared reset and controls for the real converted competition B2W/Piper."""
from pathlib import Path
import mujoco
import numpy as np
from lighting import apply_lighting

HERE=Path(__file__).resolve().parent
MODEL=HERE/'assets/hotel_b2w.xml'
BASE='World_b2w_piper_base_link'

def actor_frame_velocity(model,data,body):
    """Root COM spatial velocity in link/actor axes, matching IsaacLab.

    mj_objectVelocity(..., local=1) uses the inertial frame for a BODY, which
    differs from the link axes when USD principalAxes is non-identity.
    """
    world=np.zeros(6)
    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,world,0)
    inverse=data.xmat[body].reshape(3,3).T
    return np.r_[inverse@world[:3],inverse@world[3:]]

class HotelSimulation:
    def __init__(self,model_path=None):
        path=Path(model_path or MODEL)
        self.model=(mujoco.MjModel.from_binary_path(str(path)) if path.suffix=='.mjb'
                    else mujoco.MjModel.from_xml_path(str(path)))
        apply_lighting(self.model)
        self.data=mujoco.MjData(self.model)
        self.base=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,BASE)
        # The original 160/5 legs expect an active Isaac locomotion policy.
        # This preview has a separate static stance hold, keeping source torque
        # limits, mass, inertia, geometry and the floating root unchanged.
        for i in range(self.model.nu):
            actuator=self.model.actuator(i).name
            if not actuator.startswith('arm_') and '_foot_' not in actuator:
                self.model.actuator_gainprm[i,0]=600.
                self.model.actuator_biasprm[i,1]=-600.
                self.model.actuator_biasprm[i,2]=-20.
        self.goals={};self.speed=0.;self.turn=0.
        for j in range(1,self.model.njnt):
            # Scene objects have independent free roots; resetData already places them.
            if self.model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):continue
            joint=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)
            value=0.
            if joint.endswith('_hip_joint'):value=.1 if joint.startswith(('FL','RL')) else -.1
            elif joint.endswith('_thigh_joint'):value=.8 if joint.startswith('F') else 1.
            elif joint.endswith('_calf_joint'):value=-1.5
            elif joint=='arm_joint7':value=.045
            elif joint=='arm_joint8':value=-.045
            self.goals[joint]=value
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        self.speed=0.;self.turn=0.
        for joint,value in self.goals.items():self.data.joint(joint).qpos[0]=value
        spawn=self.data.qpos[:3].copy()
        # Query the actual source collision floor with the robot moved away.
        self.data.qpos[:3]=[999,999,5];mujoco.mj_forward(self.model,self.data)
        hit=np.array([-1],dtype=np.int32)
        origin=np.array([spawn[0],spawn[1],1.5]);direction=np.array([0.,0.,-1.])
        distance=mujoco.mj_ray(self.model,self.data,origin,direction,np.array([0,0,0,1,0,0],dtype=np.uint8),1,-1,hit)
        self.floor_z=float(origin[2]-distance) if distance>=0 else -.4
        self.data.qpos[:3]=spawn;mujoco.mj_forward(self.model,self.data)
        minimum=[]
        for i in range(self.model.ngeom):
            body=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[i])) or ''
            if body.endswith('_foot') and self.model.geom_contype[i]:
                rotation=self.data.geom_xmat[i].reshape(3,3)
                center=self.data.geom_xpos[i]+rotation@self.model.geom_aabb[i,:3]
                minimum.append(center[2]-np.abs(rotation[2])@self.model.geom_aabb[i,3:])
        if minimum:self.data.qpos[2]+=self.floor_z-min(minimum)+.015
        self.spawn=self.data.qpos[:3].copy()
        mujoco.mj_forward(self.model,self.data)
        self.control()

    def control(self):
        for i in range(self.model.nu):
            joint_id=int(self.model.actuator_trnid[i,0])
            joint=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_id)
            if joint.endswith('_foot_joint'):
                side=-1 if joint.startswith(('FL','RL')) else 1
                self.data.ctrl[i]=np.clip(self.speed+side*self.turn,-5,5)
            else:self.data.ctrl[i]=self.goals[joint]

    def step(self,n=1):
        self.control();mujoco.mj_step(self.model,self.data,nstep=n)
        if not np.isfinite(self.data.qpos).all():raise RuntimeError('Nonfinite physics state')

    def camera(self,overview=False):
        camera=mujoco.MjvCamera()
        if overview:
            camera.lookat[:]=[11.9,7,.2];camera.distance=42;camera.azimuth=90;camera.elevation=-65
        else:
            camera.lookat[:]=self.data.xpos[self.base]+[0,0,.05]
            camera.distance=3.2;camera.azimuth=115;camera.elevation=-18
        return camera
