"""Paths shared by the PPSUC package and the repository assets."""

import os
from pathlib import Path


def _find_project_root() -> Path:
    configured_root = os.environ.get("PPSUC_PROJECT_ROOT")
    search_starts = []
    if configured_root:
        search_starts.append(Path(configured_root).expanduser())
    search_starts.extend((Path.cwd(), Path(__file__).resolve()))

    for start in search_starts:
        for candidate in (start, *start.parents):
            if (candidate / "ppsuc_robot_model").is_dir() and (candidate / "scripts").is_dir():
                return candidate.resolve()

    raise RuntimeError(
        "Cannot locate PPSUC project root. Run from the project directory or set PPSUC_PROJECT_ROOT."
    )


PROJECT_ROOT = _find_project_root()
ROBOT_MODEL_ROOT = PROJECT_ROOT / "ppsuc_robot_model"
