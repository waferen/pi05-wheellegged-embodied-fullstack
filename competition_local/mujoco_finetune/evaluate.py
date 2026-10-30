"""Bounded autonomous evaluation through the existing visible dual-camera GUI."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'mujoco_hotel'))
if (HERE/'scene_cache.json').exists():
    from cache_scene import fingerprint,BINARY
    cache=json.loads((HERE/'scene_cache.json').read_text())
    if cache['passed'] and cache['fingerprint']==fingerprint():
        import hotel_runtime
        hotel_runtime.MODEL=BINARY
from pi05_teleop import Pi05Teleop
from native_viewer import main as viewer


class Trial(Pi05Teleop):
    def __init__(self,port,seconds):
        self.out=HERE/args.output;self.out.mkdir(exist_ok=True)
        super().__init__(port=port,execute_steps=25,run_dir=self.out)
        self.start_sim=float(self.data.time);self.start_wall=time.monotonic();self.limit=seconds
        self.mode_counts=collections.Counter();self.positions=[];self.last_sample=-1.;self.reason=None
        self.inside_seconds=0.;self.fall_seconds=0.;self.writer=None
        self.metrics=(self.out/'state_trace.jsonl').open('a',encoding='utf-8',buffering=1)
        self.source=json.loads((HERE/'serve_checkpoint.json').read_text())

    def policy_status(self):
        return {**super().policy_status(),'checkpoint_label':'40000 + ep3 LoRA 500',
                'finetune_updates':500,'checkpoint':self.source['checkpoint'] if hasattr(self,'source') else ''}

    def objects(self):
        result={}
        for b in range(self.model.nbody):
            name=self.model.body(b).name
            if not name.startswith('World_C4_'):continue
            p=self.data.xipos[b];inside=bool(22.59<p[0]<23.09 and 2.62<p[1]<2.99 and -.32<p[2]<.05)
            result[name]={'center':p.tolist(),'inside_vessel_region':inside}
        return result

    def finish(self,reason):
        self.reason=reason;self.quit=True

    def control_frame(self,dt=.02):
        if self.quit:return False
        advanced=super().control_frame(dt)
        if not self.model_control:
            self.finish('policy_error' if self.last_error else 'manual_takeover');return advanced
        if not advanced:return advanced
        elapsed=float(self.data.time-self.start_sim);self.mode_counts[self.mode]+=1
        self.positions.append((elapsed,self.data.xpos[self.base].copy(),self.data.qpos[self.qadr[16:22]].copy()))
        while self.positions and elapsed-self.positions[0][0]>60.:self.positions.pop(0)
        objects=self.objects()
        correct=['World_C4_easy_C4_1','World_C4_easy_C4_2']
        released=not self.grip_feedback()['objects']
        self.inside_seconds=self.inside_seconds+dt if released and all(objects[n]['inside_vessel_region'] for n in correct) else 0.
        floor_height=float(self.data.qpos[2]-self.floor_z)
        self.fall_seconds=self.fall_seconds+dt if floor_height<.08 else 0.
        if self.inside_seconds>=4.:self.finish('both_targets_in_vessel_region_and_released_4s')
        elif self.fall_seconds>=3.:self.finish('base_below_8cm_for_3s')
        elif elapsed>=self.limit:self.finish('time_limit')
        elif time.monotonic()-self.start_wall>=1800.:self.finish('wall_time_limit')
        elif self.positions and elapsed-self.positions[0][0]>=59.9:
            pos=np.array([x[1] for x in self.positions]);arm=np.array([x[2] for x in self.positions])
            if np.linalg.norm(np.ptp(pos,axis=0))<.10 and np.max(np.ptp(arm,axis=0))<.10:self.finish('no_base_or_arm_progress_for_60s')
        if elapsed-self.last_sample>=.5:
            self.last_sample=elapsed
            self.metrics.write(json.dumps({'elapsed_sim':elapsed,'base':self.data.xpos[self.base].tolist(),
                'mode':self.mode,'phase':self.client.task_phase,'objects':objects,'grip':self.grip_feedback()})+'\n')
            env=self._environment(capture=True)
            image=np.concatenate(self.images,axis=1)
            if self.writer is None:
                import imageio_ffmpeg
                self.writer=imageio_ffmpeg.write_frames(str(self.out/'closed_loop.mp4'),(1280,480),fps=2,codec='libx264',quality=7,macro_block_size=16,output_params=['-threads','1'])
                self.writer.send(None)
            self.writer.send(np.ascontiguousarray(image))
            if int(elapsed)%10==0:
                from PIL import Image
                Image.fromarray(image).save(self.out/'latest_cameras.png')
        return advanced

    def close(self):
        if self.writer is not None:self.writer.close()
        objects=self.objects()
        result={'completed':True,'termination':self.reason or 'user_closed_window',
            'success_region_check':self.reason=='both_targets_in_vessel_region_and_released_4s',
            'elapsed_sim_seconds':float(self.data.time-self.start_sim),'elapsed_wall_seconds':time.monotonic()-self.start_wall,
            'policy':self.policy_status(),'mode_control_steps':dict(self.mode_counts),'objects':objects,
            'final_base':self.data.xpos[self.base].tolist(),'finite':bool(np.isfinite(self.data.qpos).all()),
            'human_actions_used':not self.model_control and not self.last_error,
            'model_xml_sha256':hashlib.sha256((ROOT/'mujoco_hotel/assets/hotel_b2w.xml').read_bytes()).hexdigest(),
            'note':'One autonomous visible trial in the demonstration scene; region check is not a multi-seed success-rate estimate.'}
        (self.out/'result.json').write_text(json.dumps(result,indent=2));self.metrics.close()
        super().close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);p.add_argument('--seconds',type=float,default=420);p.add_argument('--output',default='evaluation_pilot500_retry');args=p.parse_args()
    viewer('Dual',sim_factory=lambda:Trial(args.port,args.seconds))
