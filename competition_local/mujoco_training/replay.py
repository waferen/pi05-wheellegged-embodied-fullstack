"""Replay a saved policy, either interactively or into an MP4."""
import argparse,json,time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from piper_env import PiperReachEnv
from train_smoke import OUT,evaluate,video

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed',type=int,default=30000)
    args=parser.parse_args()
    torch.set_num_threads(2)
    policy=PPO.load(OUT/'policy',device='cpu')
    if args.video:video(policy,OUT)
    if args.viewer:
        env=PiperReachEnv(render_mode='human')
        for i in range(args.episodes):
            obs,_=env.reset(seed=args.seed+i)
            while True:
                start=time.perf_counter()
                action,_=policy.predict(obs,deterministic=True)
                obs,_,done,truncated,info=env.step(action)
                if env.viewer is not None and not env.viewer.is_running():env.close();return
                time.sleep(max(0,.04-(time.perf_counter()-start)))
                if done or truncated:break
        env.close()
    else:
        result=evaluate(policy,range(args.seed,args.seed+args.episodes))
        (OUT/'independent_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='episodes'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
