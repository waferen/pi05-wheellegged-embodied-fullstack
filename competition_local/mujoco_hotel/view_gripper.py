"""Manual dual-camera gripper check using the current production controls."""
import json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mujoco_finetune'))
from cache_scene import fingerprint,BINARY
import hotel_runtime
cache=json.loads((HERE.parent/'mujoco_finetune/scene_cache.json').read_text())
if cache['passed'] and cache['fingerprint']==fingerprint():hotel_runtime.MODEL=BINARY
from original_teleop import OriginalTeleop
from native_viewer import main


def manual():
    sim=OriginalTeleop()
    for _ in range(50):sim.control_frame(.02)
    sim.set_mode('high_grasp')
    sim.message='MANUAL GRIPPER CHECK | C close, P open | 1 walk, 4 low, 5 high | Q/Esc exit'
    return sim


if __name__=='__main__':main('Dual',sim_factory=manual)
