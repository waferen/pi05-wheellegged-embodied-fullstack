"""Compute dedicated OpenPI delta-action stats and verify converted samples."""
import json,shutil,sys,time,traceback
import numpy as np
from experiment import config,HERE,ROOT,DATA,REPO


def main():
    sys.path.insert(0,str(ROOT/'collection_deps_wsl'))
    import h5py
    from openpi.shared import normalize
    from openpi.transforms import DeltaActions
    from openpi.policies.b2w_piper_full_policy import JOINT_DELTA_MASK
    cfg=config();assets=cfg.assets_dirs/REPO;assets.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'datasets/mujoco_raw/episode_000000.hdf5'
    with h5py.File(raw,'r') as f:
        assert f.attrs['result']=='success' and f.attrs['purpose']=='demonstration'
        state=f['state'][:];command=f['command'][:];n=len(state)
    assert n==17674
    horizon=cfg.model.action_horizon
    indices=np.minimum(np.arange(n)[:,None]+np.arange(horizon)[None,:],n-1)
    actions=command[indices].copy()
    # This is the actual OpenPI transform, batched over every starting frame.
    actions=DeltaActions(JOINT_DELTA_MASK)({'state':state,'actions':actions})['actions']
    stats={}
    for key,values in [('state',state),('actions',actions)]:
        running=normalize.RunningStats();running.update(values.astype(np.float64));stats[key]=running.get_statistics()
    normalize.save(assets,stats)
    print('Computed OpenPI statistics from all',n,'states and',n*horizon,'delta action rows',flush=True)
    deadline=time.monotonic()+7200
    while not (DATA/'conversion_report.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Conversion did not finish in two hours')
        time.sleep(5)
    report=json.loads((DATA/'conversion_report.json').read_text())
    assert report['passed'] and report['episodes']==1 and report['frames']==n
    assert not report['contains_test_data'] and not report['contains_unreviewed']
    assert len(report['sources'])==1 and report['sources'][0]['path'].endswith('episode_000000.hdf5')
    shutil.copyfile(DATA/'phase_boundaries.json',assets/'phase_boundaries.json')
    from openpi.training import data_loader
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    dataset=data_loader.create_torch_dataset(data,horizon,cfg.model)
    transformed=data_loader.transform_dataset(dataset,data)
    checks=[]
    for i in [0,4000,9391,9392,n-1]:
        sample=dataset[i]
        for tr in [*data.repack_transforms.inputs,*data.data_transforms.inputs]:sample=tr(sample)
        np.testing.assert_array_equal(sample['state'],state[i])
        np.testing.assert_array_equal(sample['actions'],actions[i])
        encoded=transformed[i]
        assert encoded['actions'].shape==(50,32) and np.isfinite(encoded['actions']).all()
        expect=(np.minimum(i+np.arange(50)+1,n-1)>=9392)*2-1
        np.testing.assert_array_equal(encoded['actions'][:,13],expect)
        assert encoded['state'][13]==(1 if i>=9392 else -1)
        checks.append({'frame':i,'action_shape':list(encoded['actions'].shape),'observation_phase':int(i>=9392)})
    result={'passed':True,'frames':n,'episodes':1,'norm_state_count':n,'norm_action_count':n*horizon,
            'norm_uses_actual_delta_transform':True,'sample_checks':checks,
            'source':report['sources'][0],'assets':str(assets),'config_checkpoint':str(cfg.checkpoint_dir)}
    (HERE/'prepare_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        (HERE/'prepare_failure.json').write_text(traceback.format_exc());raise
