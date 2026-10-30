"""Isolated single-demonstration LoRA config initialized from run01/40000."""
import dataclasses,os,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OPENPI=Path('/home/eai/ppsuc/openpi')
sys.path.insert(0,str(OPENPI/'src'))
REPO='local/mujoco_success_ep3'
NAME='pi05_mujoco_ep3_lora'
DATA=ROOT/'datasets/mujoco_success_ep3_v1'
INITIAL=ROOT/'cloud_snapshot/openpi/checkpoints/pi05_b2w_piper/run01/40000'


def config(steps=500,resume=True):
    import flax.nnx as nnx
    from openpi.shared import nnx_utils
    from openpi.training import config as cfg,optimizer,weight_loaders
    base=cfg.get_config('pi05_b2w_piper')
    model=dataclasses.replace(base.model,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',dtype='bfloat16')
    data=dataclasses.replace(base.data,repo_id=REPO,
        assets=cfg.AssetsConfig(asset_id=REPO),
        base_config=dataclasses.replace(base.data.base_config,local_root=str(DATA),video_backend='pyav'))
    result=dataclasses.replace(base,name=NAME,exp_name='pilot500',model=model,data=data,
        freeze_filter=nnx.Not(nnx_utils.PathRegex('.*lora.*')),ema_decay=None,batch_size=1,fsdp_devices=1,num_workers=0,
        num_train_steps=steps,log_interval=10,save_interval=50,keep_period=None,wandb_enabled=False,
        resume=resume,overwrite=False,checkpoint_base_dir='/home/eai/ppsuc/runs/mujoco_ep3/checkpoints',assets_base_dir=str(HERE/'assets'),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=25,peak_lr=5e-5,decay_steps=500,decay_lr=5e-6),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(INITIAL/'params')),
        policy_metadata={**base.policy_metadata,'checkpoint_label':'MuJoCo episode3 LoRA pilot',
                         'source_checkpoint_step':40000,'finetune_dataset':REPO})
    return result
