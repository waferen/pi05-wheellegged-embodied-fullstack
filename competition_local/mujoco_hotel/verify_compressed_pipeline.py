"""Create an explicitly synthetic JPEG HDF5 using real recorded camera samples."""
import json,time
import numpy as np
from collection.schema import ROOT
from collection.adapter import MujocoWriter
from collection.image_storage import open_episode
from collection.pipeline import inspect_episode


def main():
    out=ROOT/'datasets/mujoco_jpeg_contract';out.mkdir(exist_ok=True)
    raw=ROOT/'datasets/mujoco_raw/episode_000003.hdf5'
    with open_episode(raw) as f:
        ids=np.linspace(0,len(f['state'])-1,14,dtype=int)
        originals=[{cam:f['images/'+cam][int(i)] for cam in ('front','wrist')} for i in ids]
        states=[f['state'][int(i)] for i in ids];commands=[f['command'][int(i)] for i in ids]
        old_bytes=sum(f.raw['images/'+cam].id.get_storage_size() for cam in ('front','wrist'))/len(f['state'])
    w=MujocoWriter(out,{'purpose':'smoke_test','operator':'compression_contract_only'})
    started=time.perf_counter();rows=[]
    for stage in range(1,15):
        for _ in range(105 if stage==8 else 1):
            w.append(len(rows)*.02,states[stage-1],commands[stage-1],originals[stage-1],stage)
            rows.append(stage-1)
    w.close();encode_seconds=time.perf_counter()-started
    errors=[]
    with open_episode(w.path) as f:
        np.testing.assert_array_equal(f['state'][:],np.array([states[i] for i in rows]))
        np.testing.assert_array_equal(f['command'][:],np.array([commands[i] for i in rows]))
        for k,src in enumerate(rows):
            for cam in ('front','wrist'):
                errors.append(float(np.abs(f['images/'+cam][k].astype(float)-originals[src][cam]).mean()))
    result={'passed':True,'purpose':'smoke_test','source':str(raw),'output':str(w.path),
        'frames':len(rows),'file_bytes':w.path.stat().st_size,'mean_rgb_mae':float(np.mean(errors)),
        'max_rgb_mae':max(errors),'encode_seconds':encode_seconds,
        'source_average_raw_image_bytes_per_frame':old_bytes,
        'note':'Synthetic stage sequence; real RGB samples only. Excluded from training. Quality is measured; storage ratio depends on image content.'}
    result['inspection']=inspect_episode(w.path,True)
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
