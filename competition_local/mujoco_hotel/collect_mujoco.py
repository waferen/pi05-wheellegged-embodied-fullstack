"""Visible manual collection; reuse original F1/F2/F3 and 14-stage schema."""
import argparse,subprocess,sys
from pathlib import Path
from datetime import datetime
from collection.adapter import CollectionTeleop
from native_viewer import main


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'datasets/mujoco_raw')
    p.add_argument('--operator',default='local')
    p.add_argument('--no-panel',action='store_true')
    p.add_argument('--smoke-test',action='store_true',help='Synthetic stage progression in visible GUI; excluded from training by default')
    args=p.parse_args()
    if args.smoke_test:
        from collection.smoke import SmokeTeleop
        factory=SmokeTeleop
    else:factory=CollectionTeleop
    status=args.output/'status'/f'collector_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    status.parent.mkdir(parents=True,exist_ok=True)
    panel=None
    try:
        if not args.no_panel:
            panel=subprocess.Popen([sys.executable,'-m','collection.stage_panel','--status',str(status)],cwd=Path(__file__).resolve().parent)
        main('Dual',sim_factory=lambda:factory(args.output,status,operator=args.operator))
    finally:
        if panel is not None and panel.poll() is None:panel.terminate();panel.wait(timeout=5)
