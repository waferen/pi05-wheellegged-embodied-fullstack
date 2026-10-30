"""Teacher-observation diagnostics only: no training and no simulated actions."""
import fcntl
import json
import logging
import os
import sys
import time
from pathlib import Path

from experiment import HERE, ROOT, config

os.environ.setdefault('OPENPI_DATA_HOME', str(ROOT/'cloud_snapshot/cache/openpi'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.65')
sys.path.insert(0, str(ROOT/'mujoco_hotel'))

import numpy as np
from collection.image_storage import open_episode

MODES = ['walk', 'stair', 'low_grasp', 'high_grasp']
OUT = HERE/'offline_diagnosis_v1'


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = config()
    checkpoint = cfg.checkpoint_dir/'2999'
    raw = ROOT/'datasets/mujoco_raw/episode_000000.hdf5'
    source_stat = (raw.stat().st_size, raw.stat().st_mtime_ns)
    prep = json.loads((HERE/'prepare_result.json').read_text())
    assert prep['passed'] and prep['frames'] == 17674
    with open_episode(raw) as f:
        assert f.attrs['result'] == 'success' and f.attrs['purpose'] == 'demonstration'
        state, command, stage = f['state'][:], f['command'][:], f['stage_id'][:]
        n = len(state)
        print('Image streams:', list(f['images']), flush=True)
        cameras = list(f['images'])
        assert set(cameras) == {'front', 'wrist'}, cameras
        boundary = prep['source']['phase1_start_frame']
        modes = command[:, 9:13].argmax(-1)
        switches = (np.flatnonzero(modes[1:] != modes[:-1])+1).tolist()
        stages = (np.flatnonzero(stage[1:] != stage[:-1])+1).tolist()
        uniform = list(range(0, n, 50)) + [n-1]
        dense = set()
        for b in switches + stages + [boundary]:
            dense.update(b+d for d in [-100,-50,-25,-10,-1,0,1,10,25,50,100] if 0 <= b+d < n)
        selected = sorted(set(uniform) | dense)
        manifest = {'source': str(raw), 'source_stat': source_stat,
                    'source_sha256_from_prepare': prep['source']['sha256'],
                    'checkpoint': str(checkpoint), 'boundary': boundary,
                    'uniform_frames': uniform, 'selected_frames': selected,
                    'mode_switches': [{'frame':i,'from':MODES[modes[i-1]],'to':MODES[modes[i]]} for i in switches],
                    'stage_switches': stages, 'sampling': '1Hz plus dense switch neighborhoods',
                    'note': 'Training-episode observations, not held-out success or autonomous evaluation.'}
        manifest_path = OUT/'manifest.json'
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text()) == json.loads(json.dumps(manifest)), 'Changed experiment; use a new output directory'
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        records_path = OUT/'predictions.jsonl'
        records = [json.loads(x) for x in records_path.read_text().splitlines()] if records_path.exists() else []
        done = {tuple(x['key']) for x in records}
        # Restore on the host and upload largest arrays first to reduce GPU
        # allocation fragmentation with the user's other graphics apps open.
        import jax
        from openpi.models import model as model_module
        restore = model_module.restore_params
        def host_restore(*args, **kwargs):
            import gc
            params = restore(*args, **{**kwargs, 'restore_type': np.ndarray})
            leaves, tree = jax.tree.flatten(params)
            uploaded = [None]*len(leaves)
            for k in sorted(range(len(leaves)), key=lambda k: leaves[k].nbytes, reverse=True):
                uploaded[k] = jax.device_put(leaves[k]).block_until_ready()
                leaves[k] = None
            gc.collect()
            return jax.tree.unflatten(tree, uploaded)
        model_module.restore_params = host_restore
        from openpi.policies.policy_config import create_trained_policy
        policy = create_trained_policy(cfg, checkpoint)
        # The same explicit noise per frame permits paired phase-input comparisons.
        def infer(i, phase, seed=0):
            key = (i, phase, seed)
            if key in done:
                return
            obs = {'observation/state': state[i].copy(), 'observation/task_phase': np.int32(phase),
                   'observation/image': f['images/front'][i],
                   'observation/wrist_image': f['images/wrist'][i],
                   'prompt': 'Pick up both target objects and place them into the designated bin'}
            noise = np.random.default_rng(100000*seed+i).standard_normal((50, cfg.model.action_dim)).astype(np.float32)
            start = time.monotonic()
            actions = np.asarray(policy.infer(obs, noise=noise)['actions'])
            assert actions.shape == (50,14) and np.isfinite(actions).all()
            item = {'key':key,'frame':i,'recorded_seconds':i/50,'stage':int(stage[i]),
                    'obs_mode':MODES[int(state[i,9:13].argmax())], 'correct_phase':int(i>=boundary),
                    'infer_seconds':time.monotonic()-start,'actions':actions.tolist()}
            with records_path.open('a') as output:
                output.write(json.dumps(item)+'\n')
            records.append(item);done.add(key)
        for j,i in enumerate(selected):
            infer(i,int(i>=boundary))
            if j%25 == 0:
                print(f'Correct observations {j+1}/{len(selected)} frame={i}',flush=True)
                (OUT/'progress.json').write_text(json.dumps({'completed':False,'frames_done':j+1,'total':len(selected)}))
        # Independent noise seeds on stable intervals of all modes, and switches.
        edges = [0]+switches+[n]
        anchors = sorted(set([(a+b)//2 for a,b in zip(edges[:-1],edges[1:])]
                             +switches+[i-d for i in switches for d in [25,1]]
                             +[boundary-50,boundary-25,boundary-1,boundary,boundary+50]))
        for i in anchors:
            for seed in [0,1,2]:
                infer(i,int(i>=boundary),seed)
        # Phase feedback counterfactual: keep images/state/noise identical.
        paired = sorted(set(anchors + list(range(0,boundary,500))))
        for i in paired:
            infer(i,int(i>=boundary))
            infer(i,1-int(i>=boundary))
        assert source_stat == (raw.stat().st_size,raw.stat().st_mtime_ns), 'Recording changed during diagnosis'
    def metric(items, lo=0, hi=25):
        confusion = np.zeros((4,4),dtype=int);phase_conf=np.zeros((2,2),dtype=int)
        errors=[]
        for r in items:
            i=r['frame'];a=np.asarray(r['actions'])[lo:hi]
            ix=np.minimum(i+np.arange(lo,hi),n-1)
            gt=command[ix];truth=gt[:,9:13].argmax(-1);pred=a[:,9:13].argmax(-1)
            np.add.at(confusion,(truth,pred),1)
            phase=(np.minimum(ix+1,n-1)>=boundary).astype(int)
            np.add.at(phase_conf,(phase,(a[:,13]>=.5).astype(int)),1)
            errors.append(np.abs(a[:,:9]-gt[:,:9]))
        return {'rows':int(confusion.sum()),'mode_confusion_truth_rows':confusion.tolist(),
                'mode_accuracy':float(confusion.trace()/confusion.sum()),
                'phase_confusion_truth_rows':phase_conf.tolist(),
                'phase_accuracy':float(phase_conf.trace()/phase_conf.sum()),
                'physical_mae_per_channel':np.concatenate(errors).mean(0).tolist()}
    core = [r for r in records if r['frame'] in uniform and r['key'][1]==r['correct_phase'] and r['key'][2]==0]
    summary={'completed':True,'records':len(records),'uniform_observations':len(core),'modes':MODES,
             'first_row':metric(core,0,1),'first25':metric(core),'runtime_prefetch_rows15to39':metric(core,15,40),
             'source':str(raw),'checkpoint':str(checkpoint),
             'note':'Uniform-sampled training episode, correct recorded observations and phase; descriptive fit, not generalization.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'progress.json').write_text(json.dumps({'completed':True,'records':len(records)}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
