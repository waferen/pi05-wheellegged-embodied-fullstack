"""Sequential, logged, restartable single-episode conversion and training."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
    XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_ALLOCATOR='platform',WANDB_MODE='disabled',
    OPENPI_DATA_HOME=str(ROOT/'cloud_snapshot/cache/openpi'),OMP_NUM_THREADS='4')
status={'pid':os.getpid(),'started':time.time(),'completed':False}


def stage(name,command):
    status.update(stage=name,command=command,updated=time.time())
    (HERE/'status.json').write_text(json.dumps(status,indent=2))
    print('START',name,flush=True)
    with (HERE/f'{name}.log').open('ab',buffering=0) as log:
        subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)


def main():
    native=Path('/home/eai/ppsuc/runs/mujoco_ep0_v2');native.mkdir(parents=True,exist_ok=True)
    lock=(native/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=ROOT/'datasets/mujoco_success_ep0_v2'
    try:
        if not (output/'conversion_report.json').exists():
            stage('convert',[sys.executable,str(ROOT/'mujoco_hotel/convert_mujoco_to_lerobot.py'),
                '--source',str(ROOT/'datasets/mujoco_raw/episode_000000.hdf5'),
                '--output',str(output),'--repo-id','local/mujoco_success_ep0_v2'])
        if not (HERE/'prepare_result.json').exists():stage('prepare',[sys.executable,str(HERE/'prepare.py')])
        if not (HERE/'train_result.json').exists():stage('train',[sys.executable,str(HERE/'train.py'),'--steps','3000'])
        result=json.loads((HERE/'train_result.json').read_text());assert result['passed'] and result['steps_target']==3000
        checkpoint=Path(result['checkpoint_dir'])/'2999';assert (checkpoint/'params').is_dir()
        (HERE/'serve_checkpoint.json').write_text(json.dumps({'checkpoint':str(checkpoint),'port':8767}))
        stage('export',[sys.executable,str(HERE/'export_checkpoint.py')])
        status.update(completed=True,stage='complete',checkpoint=str(checkpoint),finished=time.time())
    except BaseException as exc:
        status.update(stage='failed',error=repr(exc));raise
    finally:
        (HERE/'status.json').write_text(json.dumps(status,indent=2))
        fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()
