"""Local-only BF16/LoRA memory probe. Default is a non-training preflight."""
import argparse,dataclasses,hashlib,importlib.util,json,logging,os,platform,shutil,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OPENPI=Path(os.environ.get('PPSUC_OPENPI_ROOT',str(ROOT/'cloud_snapshot/openpi')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--steps',type=int,default=10);args=parser.parse_args()
    if not 1<=args.steps<=100:raise ValueError('This probe is capped at 100 steps')
    reference=HERE/'reference_openpi'
    sources={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in reference.glob('*.py')}
    result={'inspected_at':time.time(),'platform':platform.system(),'source_files':sources,'runtime_verified':False,
            'requested_profile':{'microbatch':1,'fsdp_devices':1,'dtype':'bfloat16','lora_branches':['gemma_2b_lora','gemma_300m_lora'],'train_only_lora':True,'ema':False,'action_horizon':50,'camera_contract':'original two cameras and masked third view'},
            'blockers':[]}
    if platform.system()!='Linux':result['blockers'].append('JAX CUDA probe requires a separate Linux/WSL environment; current process is not Linux.')
    if not (OPENPI/'scripts/train.py').exists():result['blockers'].append('OpenPI source migration is not complete yet.')
    if not (ROOT/'cloud_snapshot/cache/openpi/openpi-assets/checkpoints/pi05_base/params').is_dir():result['blockers'].append('Official base model initialization is missing.')
    (HERE/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.run:
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if result['blockers']:raise SystemExit('Preflight blocked: '+'; '.join(result['blockers']))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('WANDB_MODE','disabled')
    sys.path.insert(0,str(OPENPI/'src'))
    import jax
    import flax.nnx as nnx
    from openpi.shared import nnx_utils
    from openpi.training import config,weight_loaders
    devices=jax.devices()
    if len(devices)!=1 or devices[0].platform!='gpu':raise SystemExit('Use one local CUDA device for this probe.')
    base=config.get_config('pi05_b2w_piper')
    model=dataclasses.replace(base.model,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',dtype='bfloat16')
    data=dataclasses.replace(base.data,
        assets=dataclasses.replace(base.data.assets,assets_dir=str(OPENPI/'assets/pi05_b2w_piper')),
        base_config=dataclasses.replace(base.data.base_config,local_root=str(ROOT/'cloud_snapshot/lerobot_datasets/b2w_piper_c4_transport')))
    run_dir=Path(os.environ.get('PPSUC_RUN_ROOT',str(HERE/'runs')));run_dir.mkdir(parents=True,exist_ok=True)
    # The competition PhaseFromFrame transform reads TrainConfig.assets_dirs,
    # independently of AssetsConfig.assets_dir used for normalization statistics.
    phase_assets=HERE/'assets/pi05_b2w_piper_local_lora_probe/b2w_piper_c4_transport'
    phase_assets.mkdir(parents=True,exist_ok=True)
    for filename in ('phase_boundaries.json','norm_stats.json'):
        shutil.copyfile(OPENPI/'assets/pi05_b2w_piper/b2w_piper_c4_transport'/filename,phase_assets/filename)
    cfg=dataclasses.replace(base,name='pi05_b2w_piper_local_lora_probe',exp_name=time.strftime('%Y%m%d_%H%M%S'),model=model,data=data,
        freeze_filter=nnx.Not(nnx_utils.PathRegex('.*lora.*')),ema_decay=None,batch_size=1,fsdp_devices=1,num_workers=0,
        num_train_steps=args.steps,log_interval=1,save_interval=args.steps,keep_period=None,wandb_enabled=False,resume=False,overwrite=False,
        checkpoint_base_dir=str(run_dir),assets_base_dir=str(HERE/'assets'),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ROOT/'cloud_snapshot/cache/openpi/openpi-assets/checkpoints/pi05_base/params')))
    spec=importlib.util.spec_from_file_location('openpi_local_training_entry',OPENPI/'scripts/train.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    logging.basicConfig(level=logging.INFO)
    started=time.perf_counter()
    trainer.main(cfg)
    result.update(runtime_verified=True,training_steps=args.steps,elapsed_seconds=time.perf_counter()-started,
                  device=str(devices[0]),device_memory_stats=devices[0].memory_stats(),output=str(cfg.checkpoint_dir))
    (HERE/'probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if '--run' in sys.argv:
            failure={'failed_at':time.time(),'runtime_verified':False,'error_type':type(exc).__name__,
                     'error':str(exc),'traceback':traceback.format_exc(),'argv':sys.argv[1:]}
            (HERE/'probe_failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
