"""Interactive dual-camera demo of the completed episode3 LoRA pilot."""
import argparse,json,sys
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mujoco_hotel'))
from cache_scene import fingerprint,BINARY
import hotel_runtime
cache=json.loads((HERE/'scene_cache.json').read_text())
if not cache['passed'] or cache['fingerprint']!=fingerprint():
    raise RuntimeError('Scene cache is stale; rebuild with cache_scene.py before opening the demo')
hotel_runtime.MODEL=BINARY
from pi05_teleop import Pi05Teleop
from native_viewer import main


class Demo(Pi05Teleop):
    def __init__(self,port):
        super().__init__(port=port,execute_steps=25,
            run_dir=HERE/'live_demos'/datetime.now().strftime('%Y%m%d_%H%M%S'))
        if self.client.remote.get_server_metadata().get('finetune_dataset')!='local/mujoco_success_ep3':
            self.close()
            raise RuntimeError('Connected service is not the episode3 fine-tuned policy')

    def policy_status(self):
        return {**super().policy_status(),'checkpoint_label':'40000 + MuJoCo ep3 LoRA 500','finetune_updates':500}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    main('Dual',sim_factory=lambda:Demo(args.port))
