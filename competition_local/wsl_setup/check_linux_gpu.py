"""Small CUDA/BF16 and MuJoCo checks; does not load pi05 or train a policy."""
import json,os,platform,subprocess,time
from pathlib import Path

ROOT=Path(os.environ['PPSUC_ASSET_ROOT'])
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax,jax.numpy as jnp
import torch
import mujoco
import numpy as np

devices=jax.devices()
assert len(devices)==1 and devices[0].platform=='gpu',f'Expected one CUDA GPU, got {devices}'
a=jnp.ones((512,512),dtype=jnp.bfloat16)
product=(a@a).block_until_ready()
assert bool(jnp.all(product==512))
assert not torch.version.cuda,'PyTorch should be CPU-only; JAX owns the CUDA stack'
model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_training/assets/piper_reach.xml'))
data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
mujoco.mj_step(model,data,nstep=10)
assert np.isfinite(data.qpos).all()
record={'checked_at':time.time(),'linux':platform.platform(),'jax':jax.__version__,'jax_devices':[str(d) for d in devices],
        'device_kind':devices[0].device_kind,'bf16_matmul_passed':True,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'mujoco':mujoco.__version__,'mujoco_steps':10,'mujoco_nq':model.nq,'memory_stats':devices[0].memory_stats()}
(ROOT/'wsl_setup/linux_gpu_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2,default=str))
