"""Detached WSL jobs, with explicit PID/logs and duplicate-process protection."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def launch(job):
    status=HERE/'status.json'
    state=json.loads(status.read_text()) if status.exists() else {}
    old=state.get(job,{})
    if old.get('pid'):
        try:
            cmd=Path(f"/proc/{old['pid']}/cmdline").read_bytes()
            if b'mujoco_finetune' in cmd or b'convert_mujoco' in cmd:
                print('Already running',old);return
        except FileNotFoundError:pass
    commands={
        'convert':[sys.executable,str(ROOT/'mujoco_hotel/convert_mujoco_to_lerobot.py'),
                   '--source',str(ROOT/'datasets/mujoco_raw/episode_000003.hdf5'),
                   '--output',str(ROOT/'datasets/mujoco_success_ep3_v1'),'--repo-id','local/mujoco_success_ep3'],
        'prepare':[sys.executable,str(HERE/'prepare.py')],
        'train':[sys.executable,str(HERE/'train.py'),'--steps','500'],
        'serve':[sys.executable,str(HERE/'serve.py'),'--port','8766'],
    }
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_ALLOCATOR='platform',WANDB_MODE='disabled',
        OPENPI_DATA_HOME=str(ROOT/'cloud_snapshot/cache/openpi'),OMP_NUM_THREADS='4')
    log=HERE/f'{job}.log'
    with log.open('ab',buffering=0) as output:
        proc=subprocess.Popen(commands[job],cwd=HERE,env=env,stdout=output,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,start_new_session=True)
    state[job]={'pid':proc.pid,'started_unix':time.time(),'log':str(log),'command':commands[job]}
    status.write_text(json.dumps(state,indent=2))
    print(json.dumps(state[job],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',choices=['convert','prepare','train','serve']);args=parser.parse_args()
    launch(args.job)
