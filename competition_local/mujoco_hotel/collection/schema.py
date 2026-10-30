"""Load the original simulator-independent collection modules without Isaac."""
import sys,types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'cloud_snapshot/PPSUC2026'
package=types.ModuleType('_mujoco_source_collection')
package.__path__=[str(SOURCE/'source/ppsuc_rl_lab/ppsuc_rl_lab/data_collection')]
sys.modules.setdefault(package.__name__,package)
from _mujoco_source_collection.episode import EpisodeWriter,FPS,STATE_NAMES,mode_vector,validate_stages as _validate_stages
from _mujoco_source_collection.isaac_adapter import Collector
from _mujoco_source_collection.stages import STAGES,STAGE_PROMPTS,STAGE_COUNT,STAGE_VERSION
from .image_storage import EpisodeFile

def validate_stages(file):
    return _validate_stages(file if isinstance(file,EpisodeFile) else EpisodeFile(file))
