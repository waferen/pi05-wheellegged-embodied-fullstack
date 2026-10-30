"""Original competition keys and baseline actors, adapted to MuJoCo state/actuators.

Source: PPSUC2026/scripts/rsl_rl/teleop_dual_cerebellum_arm.py and teleop/ik.py.
No Isaac runtime imports. Simulator transfer is verified separately from key parity.
"""
import ast,json,math,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from hotel_runtime import HotelSimulation,HERE,actor_frame_velocity

SOURCE=HERE.parent/'cloud_snapshot/PPSUC2026'
LEGS=[f'{side}_{part}_joint' for side in ('FR','FL','RR','RL') for part in ('hip','thigh','calf')]
WHEELS=[f'{side}_foot_joint' for side in ('FR','FL','RR','RL')]
ARMS=[f'arm_joint{i}' for i in range(1,9)]
ALL=LEGS+WHEELS+ARMS
LIMITS=np.array([(-2.618,2.618),(0,3.14),(-2.967,0),(-1.745,1.745),(-1.22,1.22),(-2.0944,2.0944)])
JOINT_KEYS={key:(i,sign) for i,pair in enumerate(('RT','EG','VB','NM','IO','KL')) for key,sign in zip(pair,(-1,1))}
IK_KEYS={'UP':(0,1),'DOWN':(0,-1),'LEFT':(1,1),'RIGHT':(1,-1),'PAGE_UP':(2,1),'PAGE_DOWN':(2,-1),'7':(3,-1),'8':(3,1),'9':(4,-1),'0':(4,1),'-':(5,-1),'=':(5,1)}

def load_actor(file):
    checkpoint=torch.load(file,map_location='cpu',weights_only=False)
    state={k[6:]:v for k,v in checkpoint['model_state_dict'].items() if k.startswith('actor.')}
    ids=sorted(int(k.split('.')[0]) for k in state if k.endswith('.weight'))
    layers=[]
    for i,index in enumerate(ids):
        weight=state[f'{index}.weight'];layers.append(torch.nn.Linear(weight.shape[1],weight.shape[0]))
        if i<len(ids)-1:layers.append(torch.nn.ELU())
    actor=torch.nn.Sequential(*layers);actor.load_state_dict(state);actor.eval()
    return actor

class OriginalTeleop(HotelSimulation):
    def __init__(self):
        torch.set_num_threads(2)
        super().__init__()
        self.qadr=np.array([self.model.joint(n).qposadr[0] for n in ALL])
        self.vadr=np.array([self.model.joint(n).dofadr[0] for n in ALL])
        self.act=np.array([self.model.actuator(n+'_drive').id for n in ALL])
        self.defaults=np.array([self.goals[n] if n not in ARMS else 0. for n in ALL])
        self.home=np.array([0.,0.,0.,0.,0.,0.,.045,-.045])
        self.arm_base=self.model.body('World_b2w_piper_arm_base').id
        self.fingers=[self.model.body('World_b2w_piper_arm_link7').id,self.model.body('World_b2w_piper_arm_link8').id]
        robot_bodies=[i for i in range(self.model.nbody) if self.model.body(i).name.startswith('World_b2w_piper_')]
        self.robot_geom_ids=np.flatnonzero(np.isin(self.model.geom_bodyid,robot_bodies))
        self.actors={name:load_actor(SOURCE/'ppsuc_robot_model/baseline'/filename) for name,filename in {
            'walk':'walk_impulse_recovery_cerebellum.pt','stair':'rough_terrain_stair_cerebellum.pt','stabilizer':'height_stabilization_cerebellum.pt'}.items()}
        self.keys=set();self.last_action=np.zeros(16);self.mode='walk';self.lock_phase='off';self.lock_positions=None
        self.lock_elapsed=0.;self.lock_stable_time=0.;self.lock_reason=None;self.lock_metrics={}
        self.target_height=.59;self.home_motion=None;self.ik_target=None;self.ik_velocity=np.zeros(6)
        self.held_time=0.;self.quit=False;self.message='Original baseline actors loaded; simulator transfer under validation.'
        self.speed_settings={'walk_forward_m_s':.65,'walk_backward_m_s':.45,'stair_speed_m_s':.25,'turn_rad_s':.6}
        self._settings_mtime=None;self._settings_next_check=0.
        self.refresh_speed_settings()
        # Exact source drive gains (preview's stronger static hold is not used).
        for i,n in enumerate(ALL):
            a=self.act[i];v=self.vadr[i]
            if n in WHEELS:
                self.model.actuator_gainprm[a,0]=1.;self.model.actuator_biasprm[a,2]=-1.;self.model.dof_damping[v]=0.
                self.model.actuator_ctrlrange[a]=[-50,50]
            else:
                # Cloud teleop overrides shoulder/elbow beyond the asset defaults.
                kp=2000 if n in ARMS[6:] else 800 if n in ('arm_joint2','arm_joint3') else 80 if n in ARMS else 160
                self.model.actuator_gainprm[a,0]=kp;self.model.actuator_biasprm[a,1]=-kp;self.model.actuator_biasprm[a,2]=0.
                self.model.dof_damping[v]=25 if n in ARMS[6:] else 13 if n in ('arm_joint2','arm_joint3') else 4 if n in ARMS else 5
        # Reuse the source bounded Jacobian solve without importing Isaac/Lula.
        ik_file=SOURCE/'source/ppsuc_rl_lab/ppsuc_rl_lab/teleop/ik.py'
        node=next(x for x in ast.parse(ik_file.read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef) and x.name=='bounded_velocity')
        namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ik_file),'exec'),namespace)
        self.bounded_velocity=namespace['bounded_velocity']

    @property
    def arm_ready(self):return self.mode in ('low_grasp','high_grasp') and self.lock_phase=='locked'

    def grip_feedback(self):
        normal=np.zeros(2);counts=np.zeros(2,dtype=int);force=np.zeros(6);objects=set()
        for k in range(self.data.ncon):
            contact=self.data.contact[k]
            a,b=map(int,self.model.geom_bodyid[[contact.geom1,contact.geom2]])
            for i,finger in enumerate(self.fingers):
                other=b if a==finger else a if b==finger else -1
                if other<0:continue
                name=self.model.body(other).name
                if not name.startswith('World_C4_'):continue
                mujoco.mj_contactForce(self.model,self.data,k,force)
                normal[i]+=max(0.,float(force[0]));counts[i]+=1;objects.add(name)
        return {'normal_force_N':normal.tolist(),'contacts':counts.tolist(),'objects':sorted(objects),
            'drive_force_N':self.data.actuator_force[self.act[22:24]].tolist(),
            'finger_position_m':self.data.qpos[self.qadr[22:24]].tolist(),
            'noslip_iterations':int(self.model.opt.noslip_iterations)}

    def refresh_speed_settings(self):
        now=time.monotonic()
        if now<self._settings_next_check:return
        self._settings_next_check=now+.5
        path=HERE/'control_settings.json'
        try:
            stamp=path.stat().st_mtime_ns
            if stamp==self._settings_mtime:return
            values=json.loads(path.read_text(encoding='utf-8-sig'))
            bounds={'walk_forward_m_s':(.25,1.),'walk_backward_m_s':(.1,.45),'stair_speed_m_s':(.1,.4),'turn_rad_s':(.1,.7)}
            updated=dict(self.speed_settings)
            for key,(low,high) in bounds.items():
                value=float(values.get(key,updated[key]))
                if not math.isfinite(value):raise ValueError('Speed settings must be finite')
                updated[key]=float(np.clip(value,low,high))
            self.speed_settings=updated;self._settings_mtime=stamp
        except FileNotFoundError:pass
        except (ValueError,TypeError,OSError) as exc:self.message='Speed settings ignored: '+str(exc)

    def adjust_walk_speed(self,delta):
        self.speed_settings['walk_forward_m_s']=round(float(np.clip(self.speed_settings['walk_forward_m_s']+delta,.25,1.)),2)
        (HERE/'control_settings.json').write_text(json.dumps(self.speed_settings,indent=2),encoding='utf-8')
        self._settings_mtime=(HERE/'control_settings.json').stat().st_mtime_ns
        self.message=f"Walk forward speed: {self.speed_settings['walk_forward_m_s']:.2f} m/s"

    def release_lock(self):
        for i in range(self.model.neq):
            if self.model.equality(i).name.startswith('teleop_lock_'):self.data.eq_active[i]=False
        self.lock_phase='off';self.lock_positions=None

    def lock(self):
        if not self.valid_lock_state():
            self.lock_phase='failed'
            raise RuntimeError('Cannot lock invalid pose or joints')
        self.lock_positions=self.data.qpos[self.qadr[:16]].copy()
        for n,value in zip(ALL[:16],self.lock_positions):
            eq=self.model.equality('teleop_lock_'+n).id
            self.model.eq_data[eq,0]=value;self.data.eq_active[eq]=True
        self.lock_phase='locked';self.message='Internal leg/wheel locks engaged; floating base remains free.'

    def set_mode(self,value):
        if value==self.mode and self.lock_phase in ('lowering','locked'):return
        previous=self.mode;self.release_lock();self.keys.clear();self.ik_target=None
        self.mode=value;self.last_action[:]=0.;self.lock_elapsed=0.;self.lock_stable_time=0.;self.lock_reason=None
        self.target_height=.236 if value=='low_grasp' else .59
        if value=='high_grasp':self.lock();self.lock_reason='immediate'
        elif value=='low_grasp':self.lock_phase='lowering'
        elif value=='walk' and previous in ('low_grasp','high_grasp'):self.start_home()
        self.message=f'Mode: {value}';print('[MODE]',value,flush=True)

    def start_home(self):
        self.home_motion=(self.data.time,self.data.qpos[self.qadr[16:22]].copy())
        self.ik_target=None

    def valid_lock_state(self):
        return bool(np.isfinite(self.data.qpos).all() and np.linalg.norm(self.data.qpos[3:7])>=1e-8)

    def ground_height(self):
        # Source lock queries collision surfaces at the current root location.
        # Group 3 includes BOTH scene and robot collisions. Exclude every robot
        # body, not only base_link; otherwise this sensor measures its own legs.
        origin=self.data.xpos[self.base].copy()+[0.,0.,.01]
        hit=np.array([-1],dtype=np.int32)
        normal=np.zeros(3)
        saved=self.model.geom_group[self.robot_geom_ids].copy()
        try:
            self.model.geom_group[self.robot_geom_ids]=5
            distance=mujoco.mj_ray(self.model,self.data,origin,np.array([0.,0.,-1.]),
                np.array([0,0,0,1,0,0],dtype=np.uint8),1,-1,hit,normal)
        finally:
            self.model.geom_group[self.robot_geom_ids]=saved
        self.ground_hit={'geom':self.model.geom(int(hit[0])).name if hit[0]>=0 else None,
                         'normal':normal.tolist(),'distance':float(distance)}
        return float(origin[2]-distance) if 0<=distance<=3. and normal[2]>.5 else None

    def update_low_lock(self,dt):
        self.lock_elapsed+=dt
        valid=self.valid_lock_state();ground=self.ground_height() if valid else None
        height_error=float(self.data.xpos[self.base,2]-ground-self.target_height) if ground is not None else float('nan')
        tilt=float(np.degrees(np.arccos(np.clip(self.data.xmat[self.base].reshape(3,3)[2,2],-1,1)))) if valid else float('nan')
        qualified=valid and ground is not None and abs(height_error)<=.005 and tilt<=5.
        self.lock_stable_time=self.lock_stable_time+dt if qualified else 0.
        self.lock_metrics={'ground_z':ground,'height_error_m':height_error,'tilt_deg':tilt,'stable_seconds':self.lock_stable_time}
        if valid and (self.lock_stable_time>=.5-1e-9 or self.lock_elapsed>=1.-1e-9):
            self.lock_reason='criteria' if self.lock_stable_time>=.5-1e-9 else 'timeout'
            self.lock()
        elif self.lock_elapsed>=1.-1e-9:self.lock_phase='failed'

    def key(self,key,pressed=True,repeat=False):
        if not pressed:self.keys.discard(key);return
        if key in ('Q','ESCAPE'):self.quit=True;return
        if key in ('1','2','4','5'):
            if not repeat:self.set_mode({'1':'walk','2':'stair','4':'low_grasp','5':'high_grasp'}[key])
            return
        if key=='SPACE':self.keys.clear();self.ik_target=None;return
        self.keys.add(key)
        if key in ('F1','F2','F3'):self.message='Recording adapter is not enabled; F1/F2/F3 remain reserved for original recording.';return
        if not self.arm_ready:return
        if key in JOINT_KEYS:
            i,sign=JOINT_KEYS[key];self.goals[ARMS[i]]+=sign*.035;self.home_motion=None;self.ik_target=None
        elif key in ('C','P'):
            self.home_motion=None
            opening=.001 if key=='C' else .045
            self.goals[ARMS[6]]=opening;self.goals[ARMS[7]]=-opening
        elif key=='H':self.start_home()
        elif key in IK_KEYS and not repeat:
            if self.ik_target is None:self.ik_target=list(self.arm_pose())
            axis,sign=IK_KEYS[key]
            if axis<3:self.ik_target[0][axis]+=sign*.01
            else:
                rot=np.zeros(3);rot[axis-3]=sign*math.radians(5)
                self.ik_target[1]=Rotation.from_rotvec(rot).as_matrix()@self.ik_target[1]
            self.held_time=0.;self.home_motion=None
        self.clamp_arm()

    def clamp_arm(self):
        for n,value in zip(ARMS[:6],np.clip([self.goals[n] for n in ARMS[:6]],LIMITS[:,0],LIMITS[:,1])):self.goals[n]=float(value)

    def observation(self,command,stabilizer=False):
        rotation=self.data.xmat[self.base].reshape(3,3)
        velocity=actor_frame_velocity(self.model,self.data,self.base)
        q=self.data.qpos[self.qadr]-self.defaults;dq=self.data.qvel[self.vadr]
        # PhysX eREVOLUTE wheel sensors wrap in [-2*pi, 2*pi], whereas
        # MuJoCo hinge coordinates accumulate turns. Preserve the source actor's
        # sensor range, keeping the physical MuJoCo coordinates unmodified.
        q[12:16]=(q[12:16]+2*np.pi)%(4*np.pi)-2*np.pi
        count=24 if stabilizer else 16
        obs=np.concatenate([np.clip(velocity[:3],-100,100)*.25,rotation.T@np.array([0,0,-1.]),command,np.clip(q[:count],-100,100),np.clip(dq[:count],-100,100)*.05,self.last_action])
        # Training ground is Z=0. The hotel is offset downward; convert this
        # height sensor to the same ground-relative coordinate for sim transfer.
        if stabilizer:
            ground=self.ground_height()
            obs=np.r_[obs,self.target_height,self.data.qpos[2]-(self.floor_z if ground is None else ground)]
        assert len(obs)==(75 if stabilizer else 57)
        return torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    def arm_pose(self):
        rotation=self.data.xmat[self.arm_base].reshape(3,3)
        position=rotation.T@(self.data.xpos[self.fingers].mean(axis=0)-self.data.xpos[self.arm_base])
        orientation=rotation.T@self.data.xmat[self.fingers[0]].reshape(3,3)
        return position,orientation

    def update_ik(self,dt):
        if self.ik_target is None:return
        position,orientation=self.arm_pose();translation=[k for k in self.keys if k in IK_KEYS and IK_KEYS[k][0]<3]
        self.held_time+=dt
        if self.held_time>.18 and translation:
            delta=np.zeros(3)
            for key in translation:axis,sign=IK_KEYS[key];delta[axis]+=sign
            self.ik_target[0]+=delta/max(1,np.linalg.norm(delta))*.15*dt
            lead=self.ik_target[0]-position;self.ik_target[0]=position+lead*min(1,.015/max(np.linalg.norm(lead),1e-12))
        err=self.ik_target[0]-position
        rot_err=Rotation.from_matrix(self.ik_target[1]@orientation.T).as_rotvec()
        twist=np.r_[err*min(6,.15/max(np.linalg.norm(err),1e-12)),rot_err*min(5,math.radians(30)/max(np.linalg.norm(rot_err),1e-12))]
        jac=np.zeros((6,self.model.nv));first=np.zeros_like(jac)
        for body in self.fingers:
            jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(self.model,self.data,jp,jr,body)
            first+=np.vstack([jp,jr])*.5
        rotation=self.data.xmat[self.arm_base].reshape(3,3)
        jac[:3]=rotation.T@first[:3];jac[3:]=rotation.T@first[3:]
        q=self.data.qpos[self.qadr[16:22]]
        self.ik_velocity=self.bounded_velocity(jac[:,self.vadr[16:22]],twist,q,self.ik_velocity,LIMITS,np.full(6,1.5),np.full(6,8.),dt,damping=.015)
        for n,value in zip(ARMS[:6],q+self.ik_velocity*dt):self.goals[n]=float(value)

    def control_frame(self,dt=.02,command_override=None):
        self.refresh_speed_settings()
        direction=int('W' in self.keys)-int('S' in self.keys)
        speed=(self.speed_settings['walk_forward_m_s'] if direction>=0 else self.speed_settings['walk_backward_m_s']) if self.mode=='walk' else self.speed_settings['stair_speed_m_s']
        command=np.array([speed*direction,0.,self.speed_settings['turn_rad_s']*(int('A' in self.keys)-int('D' in self.keys))])
        if command_override is not None:
            command=np.asarray(command_override,dtype=float).copy()
            if command.shape!=(3,) or not np.isfinite(command).all():raise ValueError('Expected finite base command[3]')
        if self.mode not in ('walk','stair'):command[:]=0.
        if self.home_motion:
            start,q=self.home_motion;alpha=min(1,(self.data.time-start)/1.);alpha=alpha*alpha*(3-2*alpha)
            for n,value in zip(ARMS[:6],q*(1-alpha)):self.goals[n]=float(value)
            if alpha>=1:self.home_motion=None
        elif self.arm_ready:self.update_ik(dt)
        self.data.qfrc_applied[:]=0.
        if self.arm_ready:self.data.qfrc_applied[self.vadr[16:22]]=self.data.qfrc_bias[self.vadr[16:22]]
        if self.lock_phase=='locked':
            for i,n in enumerate(ALL[:16]):self.data.ctrl[self.act[i]]=self.lock_positions[i] if i<12 else 0.
            self.last_action[:]=0.
        else:
            actor_name=self.mode if self.mode in ('walk','stair') else 'stabilizer'
            with torch.inference_mode():action=self.actors[actor_name](self.observation(command,actor_name=='stabilizer')).numpy()[0]
            if not np.isfinite(action).all():raise RuntimeError('Nonfinite source actor output')
            self.last_action=np.clip(action,-100,100)
            for i,n in enumerate(LEGS):self.data.ctrl[self.act[i]]=self.defaults[i]+self.last_action[i]*(.125 if '_hip_' in n else .25)
            self.data.ctrl[self.act[12:16]]=np.clip(self.last_action[12:16]*8,-50,50)
        for i,n in enumerate(ARMS):self.data.ctrl[self.act[16+i]]=self.goals[n]
        self.step_physics(dt,command)
        if not np.isfinite(self.data.qpos).all():raise RuntimeError('Nonfinite MuJoCo state')
        if self.lock_phase=='lowering':
            self.update_low_lock(dt)
        self.command=command

    def step_physics(self,dt,command):
        """Hook after targets are submitted, before integration (collection)."""
        # Isaac's implicit gripper drive has velocity_limit_sim=1 m/s.
        # MuJoCo has no equivalent joint-velocity limit here: slew only the
        # internal drive reference at physical substeps. The policy/recorded
        # position target and final contact holding force remain unchanged.
        indices=self.act[22:24]
        requested=self.data.ctrl[indices].copy()
        if not hasattr(self,'_gripper_drive_target'):
            self._gripper_drive_target=self.data.qpos[self.qadr[22:24]].copy()
        limit=float(self.model.opt.timestep) # 1 metre/second
        try:
            for _ in range(round(dt/self.model.opt.timestep)):
                self._gripper_drive_target+=np.clip(requested-self._gripper_drive_target,-limit,limit)
                self.data.ctrl[indices]=self._gripper_drive_target
                mujoco.mj_step(self.model,self.data)
        finally:self.data.ctrl[indices]=requested

def prepare_locks():
    import xml.etree.ElementTree as ET
    path=HERE/'assets/hotel_b2w.xml';tree=ET.parse(path);root=tree.getroot()
    eq=root.find('equality')
    if eq is None:eq=ET.SubElement(root,'equality')
    names={x.get('name') for x in eq}
    for n in ALL[:16]:
        if 'teleop_lock_'+n not in names:ET.SubElement(eq,'joint',name='teleop_lock_'+n,joint1=n,polycoef='0 0 0 0 0',active='false',solref='.002 1')
    tree.write(path,encoding='utf-8',xml_declaration=True)

if __name__=='__main__':prepare_locks()
