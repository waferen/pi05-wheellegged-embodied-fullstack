"""Load the single stage definition once per process, independent of cwd."""
import json
from pathlib import Path

STAGES_PATH = Path(__file__).resolve().parents[4] / "configs" / "stages.json"


def load_stages(path=STAGES_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version, stages = data.get("version"), data.get("stages")
    if type(version) is not int or version < 1:
        raise ValueError("Stage version must be a positive integer")
    if not isinstance(stages, list) or len(stages) != 14:
        raise ValueError("Exactly 14 stages are required")
    for expected, stage in enumerate(stages, 1):
        if not isinstance(stage, dict) or type(stage.get("id")) is not int or stage["id"] != expected:
            raise ValueError("Stage IDs must be consecutive from 1 to 14")
        for key in ("instruction_en", "instruction_zh"):
            value = stage.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Stage {expected}: missing {key}")
        if not stage["instruction_en"].isascii():
            raise ValueError(f"Stage {expected}: English must be ASCII")
    return version, tuple(stages)


STAGE_VERSION, STAGES = load_stages()
STAGE_COUNT = len(STAGES)
STAGE_PROMPTS = tuple(stage["instruction_en"] for stage in STAGES)
STAGE_PROMPTS_ZH = tuple(stage["instruction_zh"] for stage in STAGES)
