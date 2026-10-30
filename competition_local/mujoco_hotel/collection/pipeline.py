"""Validate, review or replay raw MuJoCo episodes. No training or upload."""
import argparse,json,time
from pathlib import Path
from .schema import FPS,validate_stages
import h5py,numpy as np
from .image_storage import open_episode


def inspect_episode(path,complete=False):
    path=Path(path)
    with open_episode(path) as f:
        prompts,stages=validate_stages(f);n=len(stages)
        if f.attrs.get('simulator')!='mujoco':raise ValueError('Expected MuJoCo collection metadata')
        if int(f.attrs.get('fps',0))!=FPS:raise ValueError('Expected 50 Hz metadata')
        if 'timestamp' not in f or 'sim_time' not in f:raise ValueError('Missing explicit clock data')
        timestamps=f['timestamp'][:];sim=f['sim_time'][:]
        if timestamps.shape!=(n,) or sim.shape!=(n,) or not np.isfinite(sim).all():raise ValueError('Invalid timestamp shapes or values')
        if not np.allclose(timestamps,np.arange(n)/FPS,atol=1e-7,rtol=0):raise ValueError('Invalid relative timestamps')
        if n>1 and not np.allclose(np.diff(sim),1/FPS,atol=1e-7,rtol=0):raise ValueError('Simulation gaps / reset inside episode')
        if complete and int(stages[-1])!=14:raise ValueError('Training episode must reach stage 14')
        for key in ('state','command'):
            grip=f[key][:,8]
            if np.any((grip<0)|(grip>1)):raise ValueError('Gripper normalization out of range')
        # Read first and last RGB frames to catch truncated image chunks.
        for name in ('front','wrist'):
            for index in sorted({0,n-1}):f[f'images/{name}'][index]
        stage8=np.flatnonzero(stages==8)
        boundary=None
        if len(stage8):
            b=int(np.searchsorted(timestamps,timestamps[stage8[0]]+2.-1e-4))
            if b<n and stages[b]==8:boundary=b
        if complete and boundary is None:raise ValueError('Stage 8 must contain its first frame + 2 second boundary')
        return {'path':str(path.resolve()),'frames':n,'seconds':n/FPS,'last_stage':int(stages[-1]),
            'result':str(f.attrs.get('result','unreviewed')),'purpose':str(f.attrs.get('purpose','unknown')),
            'phase1_start_frame':boundary,'fps':FPS,'valid':True}


def selected_sources(directory,allow_unreviewed=False,allow_smoke=False):
    selected=[];excluded=[]
    source=Path(directory)
    paths=[source] if source.is_file() else sorted(source.glob('*.hdf5'))
    for path in paths:
        try:
            record=inspect_episode(path,complete=True)
            if record['purpose']!='demonstration' and not allow_smoke:raise ValueError('Non-demonstration purpose excluded')
            if record['result']!='success' and not (allow_unreviewed and record['result']=='unreviewed'):
                raise ValueError('Only manually reviewed success episodes are accepted')
            selected.append(record)
        except (OSError,ValueError,KeyError) as exc:excluded.append({'path':str(path),'reason':str(exc)})
    if not selected:raise ValueError('No eligible episodes: '+json.dumps(excluded,ensure_ascii=False))
    return selected,excluded


def replay(path,start=0):
    import tkinter as tk
    from PIL import Image,ImageTk
    inspect_episode(path)
    with open_episode(path) as f:
        root=tk.Tk();root.title('MuJoCo 原始回合回放（仅查看）');label=tk.Label(root);label.pack()
        info=tk.StringVar();tk.Label(root,textvariable=info,font=('Microsoft YaHei UI',11),wraplength=1200).pack()
        slider=tk.Scale(root,from_=0,to=len(f['state'])-1,orient='horizontal',length=1000);slider.pack()
        slider.set(start);prompts=f['prompt'].asstr()[:]
        def show(_=None):
            i=int(slider.get());stage=int(f['stage_id'][i]);canvas=Image.new('RGB',(1280,480))
            for j,name in enumerate(('front','wrist')):canvas.paste(Image.fromarray(f[f'images/{name}'][i]),(j*640,0))
            photo=ImageTk.PhotoImage(canvas);label.configure(image=photo);label.image=photo
            info.set(f"frame {i} | {f['timestamp'][i]:.2f}s | stage {stage}/14 | result={f.attrs['result']}\n{prompts[stage-1]}")
        slider.configure(command=show);show();root.mainloop()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    v=sub.add_parser('validate');v.add_argument('path',type=Path);v.add_argument('--complete',action='store_true')
    r=sub.add_parser('review');r.add_argument('path',type=Path);r.add_argument('--result',choices=['success','failure','exclude'],required=True);r.add_argument('--note',default='')
    q=sub.add_parser('replay');q.add_argument('path',type=Path)
    args=p.parse_args()
    if args.operation=='validate':
        paths=sorted(args.path.glob('*.hdf5')) if args.path.is_dir() else [args.path]
        if not paths:raise ValueError('No completed HDF5 files')
        print(json.dumps([inspect_episode(path,args.complete) for path in paths],ensure_ascii=False,indent=2))
    elif args.operation=='review':
        if args.path.suffix!='.hdf5':raise ValueError('Review only finalized .hdf5 episodes')
        inspect_episode(args.path,complete=args.result=='success')
        with h5py.File(args.path,'r+') as f:
            if f.attrs.get('purpose')!='demonstration' and args.result=='success':raise ValueError('Smoke tests cannot be marked successful demonstrations')
            history=json.loads(f.attrs.get('review_history','[]'))
            history.append({'previous':str(f.attrs['result']),'result':args.result,'note':args.note,'time':time.time()})
            f.attrs['result']=args.result;f.attrs['review_history']=json.dumps(history,ensure_ascii=False)
        print(json.dumps(inspect_episode(args.path),ensure_ascii=False,indent=2))
    else:replay(args.path)


if __name__=='__main__':main()
