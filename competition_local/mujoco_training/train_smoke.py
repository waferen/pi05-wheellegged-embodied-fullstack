"""CPU PPO smoke run, held-out evaluation, checkpoint reload and visual proof."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from piper_env import PiperReachEnv

OUT=Path(__file__).resolve().parent/'runs'/'smoke'

def evaluate(model,seeds):
    env=PiperReachEnv(); records=[]
    for seed in seeds:
        obs,initial=env.reset(seed=int(seed)); reward_sum=0
        while True:
            action=np.zeros(2) if model is None else model.predict(obs,deterministic=True)[0]
            obs,reward,done,truncated,info=env.step(action); reward_sum+=reward
            if done or truncated:break
        records.append(dict(seed=int(seed),reward=reward_sum,initial_joint_error=initial['joint_error_rad'],**info))
    env.close()
    return {'episodes':records,'success_rate':float(np.mean([r['is_success'] for r in records])),
            'mean_final_joint_error_rad':float(np.mean([r['joint_error_rad'] for r in records])),
            'mean_final_tcp_error_m':float(np.mean([r['tcp_error_m'] for r in records])),
            'mean_return':float(np.mean([r['reward'] for r in records]))}

class Progress(BaseCallback):
    def _on_step(self):
        if self.num_timesteps%4096==0:
            print(json.dumps({'training_steps':self.num_timesteps}),flush=True)
        return True

def video(model,out):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    env=PiperReachEnv(render_mode='rgb_array')
    with imageio.get_writer(out/'trained_piper.mp4',fps=25,codec='libx264',macro_block_size=2) as writer:
        for seed in [20001,20002,20003]:
            obs,_=env.reset(seed=seed)
            for step in range(100):
                action=model.predict(obs,deterministic=True)[0]
                obs,_,done,truncated,info=env.step(action)
                frame=Image.fromarray(env.render()); draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,800,66),fill=(20,25,33))
                draw.text((15,12),'PPSUC | Piper + MuJoCo | CPU-trained PPO',fill='white')
                draw.text((15,36),f"Held-out goal {seed} | error {info['joint_error_rad']:.3f} rad | TCP {info['tcp_error_m']*1000:.1f} mm",fill=(80,240,160))
                writer.append_data(np.asarray(frame))
                if seed==20001 and step==0:frame.save(out/'preview.png')
                if done or truncated:
                    for _ in range(15):writer.append_data(np.asarray(frame))
                    break
    env.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=32768); p.add_argument('--no-video',action='store_true'); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    env=PiperReachEnv(); check_env(env,warn=True); env.close()
    seeds=list(range(10000,10032))
    baseline=evaluate(None,seeds)
    vec=make_vec_env(PiperReachEnv,n_envs=4,seed=42)
    model=(PPO.load(OUT/'policy',env=vec,device='cpu') if a.resume else
           PPO('MlpPolicy',vec,device='cpu',seed=42,n_steps=256,batch_size=256,n_epochs=10,learning_rate=5e-4,gamma=.98,gae_lambda=.95,ent_coef=0.0,policy_kwargs={'net_arch':[64,64]},verbose=0))
    before={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    model.set_logger(configure(str(OUT),['csv']))
    started=time.perf_counter()
    model.learn(total_timesteps=a.steps,callback=Progress(),reset_num_timesteps=not a.resume)
    elapsed=time.perf_counter()-started
    model.save(OUT/'policy')
    changed=any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items())
    trained=evaluate(model,seeds)
    reloaded=PPO.load(OUT/'policy',device='cpu')
    reload_eval=evaluate(reloaded,seeds[:4])
    reload_match=all(abs(x['reward']-y['reward'])<1e-6 for x,y in zip(trained['episodes'][:4],reload_eval['episodes']))
    passed=changed and reload_match and trained['success_rate']>=.8 and trained['mean_final_joint_error_rad']<baseline['mean_final_joint_error_rad']*.5
    result={'passed':passed,'task':'Fixed-base migrated Piper: two-joint random target reaching; other joints held. Not full competition or VLA training.',
            'device':'cpu','seed':42,'total_timesteps':model.num_timesteps,'training_seconds':elapsed,'steps_per_second':a.steps/elapsed,
            'weights_changed':changed,'save_reload_reproduces':reload_match,'baseline':baseline,'trained':trained,
            'thresholds':{'success_rate':.8,'joint_error_ratio_vs_baseline':.5},'checkpoint':str(OUT/'policy.zip')}
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['baseline','trained']},ensure_ascii=False),flush=True)
    print(json.dumps({'baseline':{k:v for k,v in baseline.items() if k!='episodes'},'trained':{k:v for k,v in trained.items() if k!='episodes'}}),flush=True)
    vec.close()
    if not a.no_video:video(reloaded,OUT)
    if not passed:raise SystemExit('Training ran, but quality acceptance failed; inspect result.json.')

if __name__=='__main__':main()
