"""MuJoCo binding around the original Collector/EpisodeWriter lifecycle."""
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
from .schema import Collector,EpisodeWriter,FPS,STAGE_VERSION,STAGES
from .sensors import Sensors
from .image_storage import EpisodeFile,DEFAULT_JPEG_QUALITY
from original_teleop import OriginalTeleop,ARMS,HERE


class MujocoWriter(EpisodeWriter):
    def __init__(self,directory,metadata):
        super().__init__(directory)
        self.file=EpisodeFile(self.file,quality=DEFAULT_JPEG_QUALITY)
        for key,value in metadata.items():self.file.attrs[key]=value
        self.file.attrs['created_utc_unix']=time.time()
        self.file.attrs['fps']=FPS
        self.file.attrs['simulator']='mujoco'
        self.file.attrs['collection_schema']='mujoco_source_contract_v2_jpeg'
        self.file.attrs['image_encoding']='jpeg'
        self.file.attrs['jpeg_quality']=DEFAULT_JPEG_QUALITY
        self.file.attrs['jpeg_subsampling']=0
        self.file.attrs['result']='unreviewed'
        self.file.attrs['observation_action_alignment']='pre_step_observation_and_submitted_absolute_targets'
        self.first_time=None

    def append(self,sim_time,state,command,images,stage_id):
        super().append(sim_time,state,command,images,stage_id)
        if self.first_time is None:self.first_time=float(sim_time)
        for name,value in [('timestamp',(self.count-1)/FPS),('sim_time',float(sim_time))]:
            if name not in self.file:self.file.create_dataset(name,shape=(0,),maxshape=(None,),dtype='float64',chunks=True)
            ds=self.file[name];ds.resize(self.count,axis=0);ds[-1]=value


class MujocoCollector(Collector):
    def __init__(self,env,directory,status_path,metadata):
        self.metadata=metadata;self.last_episode=None
        super().__init__(env,list(range(8)),str(directory),.001,.045)
        self._status_path=str(status_path);self._publish_status()

    def start(self):
        if self.writer is not None:return
        try:self.writer=MujocoWriter(self.directory,self.metadata)
        except Exception as exc:
            self.notify(f'Recording failed: {exc}',error=True)
            raise
        self._observation=self._pending_frame=None
        self.stage_id=self._submitted_stage=1;self._stage_start=0;self._pending_stage=False
        self.last_episode=str(self.writer.path)
        self._show_stage()

    def stop(self,publish=True):
        if self.writer is not None:
            self.writer.file.attrs['last_stage']=self.stage_id
            self.writer.file.attrs['complete_stage_sequence']=self.stage_id==14
            self.writer.file.attrs['closed_utc_unix']=time.time()
        super().stop(publish)


class CollectionTeleop(OriginalTeleop):
    def __init__(self,directory,status_path,purpose='demonstration',operator='local'):
        super().__init__()
        self.sensors=Sensors(self)
        metadata={'purpose':purpose,'operator':operator,'control_source':'human_teleop',
            'physics_timestep':float(self.model.opt.timestep),'simulator_version':__import__('mujoco').__version__,
            'model_xml_sha256':hashlib.sha256((HERE/'assets/hotel_b2w.xml').read_bytes()).hexdigest(),
            'controller_sha256':hashlib.sha256((HERE/'original_teleop.py').read_bytes()).hexdigest(),
            'lighting_settings_json':(HERE/'lighting_settings.json').read_text(encoding='utf-8'),
            'control_settings_json':json.dumps(self.speed_settings,sort_keys=True),
            'stages_json':json.dumps(STAGES,ensure_ascii=False),'created_utc_unix':time.time(),
            'low_pose_note':'Mode4 retains 1-second timeout; target height accuracy not yet accepted.'}
        self.collector=MujocoCollector(self.sensors.env,Path(directory),status_path,metadata)
        self.message='F1 record | F2 next stage | F3 save | F6 preserve interrupted episode'

    def attach_context(self,context):self.sensors.attach_context(context)

    def key(self,key,pressed=True,repeat=False):
        if key in ('F1','F2','F3','F6'):
            if pressed and not repeat:
                {'F1':self.collector.start,'F2':self.collector.next_stage,
                 'F3':self.collector.stop,'F6':lambda:self.collector.stop(False)}[key]()
            return
        return super().key(key,pressed,repeat)

    def step_physics(self,dt,command):
        targets=torch.tensor([[self.data.ctrl[self.act[16+i]] for i in range(8)]],dtype=torch.float32)
        self.collector.command_submitted(command,targets,self.mode,arm_ready=self.arm_ready)
        super().step_physics(dt,command)

    def control_frame(self,dt=.02,command_override=None):
        if not np.isclose(dt,1/FPS,atol=1e-9,rtol=0):raise ValueError('Collection requires 50 Hz simulation control')
        if self.collector.writer is not None:
            self.sensors.update()
            self.collector.capture_observation(float(self.data.time),self.mode)
        try:
            super().control_frame(dt,command_override)
            self.collector.sample()
        except BaseException:
            self.collector.stop(False)
            raise

    def collection_status(self):
        c=self.collector
        return {'recording':c.writer is not None,'stage':c.stage_id,
            'frames':c.writer.count if c.writer else 0,'fps':FPS,'last_episode':c.last_episode,
            'message':c._status_message}

    def reset_scene(self):
        self.collector.stop(False)
        self.release_lock();self.keys.clear();self.home_motion=None;self.ik_target=None;self.ik_velocity[:]=0.
        self.last_action[:]=0.;self.mode='walk';self.target_height=.59
        self.goals.update({n:float(v) for n,v in zip(ARMS,self.home)})
        self.reset()

    def close(self):
        self.collector.stop(False)
