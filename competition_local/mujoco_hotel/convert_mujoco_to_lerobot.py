"""MuJoCo raw -> local LeRobot v3 + unchanged two-phase manifest. No training."""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
from collection.schema import FPS,STATE_NAMES,ROOT
from collection.pipeline import selected_sources
from collection.image_storage import open_episode
import h5py,numpy as np

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_DATASETS_OFFLINE','1')

def file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo-id',default='local/mujoco_b2w_piper')
    p.add_argument('--allow-unreviewed',action='store_true');p.add_argument('--allow-smoke',action='store_true')
    args=p.parse_args();root=args.output.resolve();pending=root.with_name(root.name+'.partial')
    if root.exists() or pending.exists():raise FileExistsError('Output/partial exists; choose a new directory')
    records,excluded=selected_sources(args.source,args.allow_unreviewed,args.allow_smoke)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features={'observation.state':{'dtype':'float32','shape':(13,),'names':list(STATE_NAMES)},
        'action':{'dtype':'float32','shape':(13,),'names':list(STATE_NAMES)},
        'stage_id':{'dtype':'int32','shape':(1,),'names':['stage_id']}}
    for camera in ('front','wrist'):
        features[f'observation.images.{camera}']={'dtype':'video','shape':(480,640,3),'names':['height','width','channels']}
    for record in records:
        path=Path(record['path']);record['sha256']=file_hash(path)
    root.parent.mkdir(parents=True,exist_ok=True)
    dataset=LeRobotDataset.create(args.repo_id,fps=FPS,features=features,root=pending,
        robot_type='ppsuc_b2w_piper_mujoco',use_videos=True,video_backend='pyav',image_writer_threads=2)
    try:
        for record in records:
            with open_episode(record['path']) as f:
                prompts=f['prompt'].asstr()[:]
                for i in range(record['frames']):
                    stage=int(f['stage_id'][i])
                    frame={'observation.state':f['state'][i],'action':f['command'][i],
                        'stage_id':np.array([stage],np.int32),'task':prompts[stage-1]}
                    frame.update({f'observation.images.{cam}':f[f'images/{cam}'][i] for cam in ('front','wrist')})
                    dataset.add_frame(frame)
                dataset.save_episode()
                print('Converted',record['path'],record['frames'],flush=True)
    finally:dataset.finalize()
    # Reuse the original stage8 + 2s / 1e-4 timestamp-tolerance builder verbatim.
    spec=importlib.util.spec_from_file_location('_phase_builder',ROOT/'cloud_snapshot/openpi/scripts/build_b2w_phase_boundaries.py')
    phase_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase_module)
    manifest=phase_module.build_boundaries(pending,args.repo_id);manifest['dataset_root']=str(root)
    (pending/'phase_boundaries.json').write_text(json.dumps(manifest,indent=2))
    report=verify(pending,args.repo_id,records,manifest)
    report.update({'sources':records,'excluded':excluded,'repo_id':args.repo_id,'output':str(root),
        'contains_unreviewed':any(r['result']!='success' for r in records),
        'contains_test_data':any(r['purpose']!='demonstration' for r in records),
        'training_started':False,'hub_upload':False})
    (pending/'conversion_report.json').write_text(json.dumps(report,indent=2))
    pending.rename(root)
    print(json.dumps({k:report[k] for k in ('passed','frames','episodes','output','contains_test_data')},indent=2))


def verify(root,repo_id,records,manifest):
    import av,pyarrow.dataset as arrow,pyarrow.compute as compute
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    table=arrow.dataset(str(root/'data'),format='parquet').to_table()
    report={'passed':True,'frames':sum(r['frames'] for r in records),'episodes':len(records),'videos':[]}
    if table.num_rows!=report['frames']:raise ValueError('Parquet frame count mismatch')
    for episode,record in enumerate(records):
        rows=table.filter(compute.equal(table['episode_index'],episode))
        with open_episode(record['path']) as f:
            for key,original in [('observation.state','state'),('action','command')]:
                np.testing.assert_array_equal(np.asarray(rows[key].to_pylist()),f[original][:])
            np.testing.assert_array_equal(np.asarray(rows['stage_id'].to_pylist()).reshape(-1),f['stage_id'][:])
            np.testing.assert_allclose(np.asarray(rows['timestamp'].to_pylist()),f['timestamp'][:],atol=1e-4,rtol=0)
    for camera in ('front','wrist'):
        count=0
        for path in sorted((root/'videos'/f'observation.images.{camera}').rglob('*.mp4')):
            with av.open(str(path)) as container:
                if abs(float(container.streams.video[0].average_rate)-FPS)>1e-6:raise ValueError('Video FPS mismatch')
                for frame in container.decode(video=0):
                    if (frame.width,frame.height)!=(640,480):raise ValueError('Video dimensions mismatch')
                    count+=1
        if count!=report['frames']:raise ValueError('Missing video frames')
        report['videos'].append({'camera':camera,'frames':count})
    sequence=LeRobotDataset(repo_id,root=root,video_backend='pyav',delta_timestamps={'action':[i/FPS for i in range(50)]})
    offset=0;rgb_checks=[]
    for episode,record in enumerate(records):
        for index in sorted({0,record['frames']-1,record['phase1_start_frame']-1,record['phase1_start_frame']}):
            sample=sequence[offset+index]
            if tuple(sample['action'].shape)!=(50,13):raise ValueError('Wrong action window')
            with open_episode(record['path']) as f:
                if sample['task']!=f['prompt'].asstr()[int(f['stage_id'][index])-1]:raise ValueError('Task text mismatch')
                for camera in ('front','wrist'):
                    image=sample[f'observation.images.{camera}'].numpy().transpose(1,2,0)*255
                    error=float(np.abs(image-f[f'images/{camera}'][index].astype(float)).mean())
                    if error>20:raise ValueError('Video / original frame mismatch')
                    rgb_checks.append({'episode':episode,'frame':index,'camera':camera,'mae':error})
            if index==record['frames']-1 and int(sample['action_is_pad'].sum())!=49:raise ValueError('Action window crosses episode')
        offset+=record['frames']
        if file_hash(record['path'])!=record['sha256']:raise ValueError('Source modified during conversion')
    report['rgb_checks']=rgb_checks
    return report


if __name__=='__main__':main()
