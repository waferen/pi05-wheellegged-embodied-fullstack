"""Start/resume the authorized pilot; preserve full optimizer checkpoints."""
import argparse,importlib.util,json,logging,time,traceback,shutil,gc,ctypes
from experiment import config,OPENPI,HERE,INITIAL


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=3000);args=parser.parse_args()
    deadline=time.monotonic()+7200
    while not (HERE/'prepare_result.json').exists():
        if (HERE/'prepare_failure.json').exists():raise RuntimeError((HERE/'prepare_failure.json').read_text())
        if time.monotonic()>deadline:raise TimeoutError('Preparation did not complete in two hours')
        time.sleep(5)
    cfg=config(args.steps)
    assert json.loads((HERE/'prepare_result.json').read_text())['passed']
    spec=importlib.util.spec_from_file_location('_mujoco_trainer',OPENPI/'scripts/train.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    import orbax.checkpoint as ocp
    def initialize(path,*,keep_period,overwrite,resume):
        assert not overwrite
        path.mkdir(parents=True,exist_ok=True)
        manager=ocp.CheckpointManager(path,item_handlers={
            'assets':trainer._checkpoints.CallbackHandler(),
            'train_state':ocp.PyTreeCheckpointHandler(save_concurrent_gb=4,restore_concurrent_gb=4),
            'params':ocp.PyTreeCheckpointHandler(save_concurrent_gb=4,restore_concurrent_gb=4)},
            options=ocp.CheckpointManagerOptions(max_to_keep=1,keep_period=keep_period,create=False,
                enable_async_checkpointing=False))
        return manager,bool(resume and manager.all_steps())
    trainer._checkpoints.initialize_checkpoint_dir=initialize
    original_save=trainer._checkpoints.save_state
    def save(*args,**kwargs):
        original_save(*args,**kwargs);args[0].wait_until_finished()
        gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
    trainer._checkpoints.save_state=save
    original_init=trainer.init_train_state
    def initialize_state(*args,**kwargs):
        state,sharding=original_init(*args,**kwargs)
        if not kwargs.get('resume',False):
            import jax
            jax.block_until_ready(state)
        gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
        return state,sharding
    trainer.init_train_state=initialize_state
    started=time.time()
    (HERE/'train_config.json').write_text(json.dumps({'initial_checkpoint':str(INITIAL),'config':repr(cfg),
        'train_steps':args.steps,'started_unix':started,'train_only_lora':True},indent=2))
    trainer.main(cfg)
    import jax
    result={'passed':True,'steps_target':args.steps,'elapsed_seconds':time.time()-started,
            'checkpoint_dir':str(cfg.checkpoint_dir),'gpu_memory':jax.devices()[0].memory_stats()}
    (HERE/'train_result.json').write_text(json.dumps(result,indent=2,default=str));print(json.dumps(result,default=str))


if __name__=='__main__':
    try:main()
    except BaseException:
        (HERE/'train_failure.json').write_text(traceback.format_exc());raise
