"""Build stage-8 + 2-second labels without modifying the LeRobot dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


TIMESTAMP_TOLERANCE_S = 1e-4  # Matches the dataset timestamp tolerance.


def build_boundaries(root: Path, repo_id: str) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    columns = ["episode_index", "frame_index", "timestamp", "stage_id"]
    table = ds.dataset(str(root / "data"), format="parquet").to_table(columns=columns)
    values = {key: np.asarray(table[key].to_pylist()).reshape(-1) for key in columns}
    if len(values["frame_index"]) != info["total_frames"]:
        raise ValueError("Data row count does not match meta/info.json")
    episodes = np.unique(values["episode_index"])
    if not np.array_equal(episodes, np.arange(info["total_episodes"])):
        raise ValueError("Expected every episode listed in meta/info.json")
    records = []
    for episode in episodes:
        mask = values["episode_index"] == episode
        order = np.argsort(values["frame_index"][mask])
        frames, times, stages = (values[key][mask][order] for key in columns[1:])
        if (not np.array_equal(frames, np.arange(len(frames))) or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0)):
            raise ValueError(f"Episode {episode}: invalid frame indices or timestamps")
        stage8 = np.flatnonzero(stages == 8)
        if not len(stage8) or np.any(np.diff(stage8) != 1):
            raise ValueError(f"Episode {episode}: stage 8 must be one nonempty contiguous segment")
        first = int(stage8[0])
        target_time = float(times[first]) + 2.0
        boundary = int(np.searchsorted(times, target_time - TIMESTAMP_TOLERANCE_S))
        if boundary >= len(frames) or stages[boundary] != 8:
            raise ValueError(f"Episode {episode}: stage 8 does not contain the 2-second boundary")
        records.append({
            "episode_index": int(episode),
            "length": len(frames),
            "stage8_start_frame": int(frames[first]),
            "stage8_start_timestamp": float(times[first]),
            "phase1_start_frame": int(frames[boundary]),
            "phase1_start_timestamp": float(times[boundary]),
        })
    return {
        "version": 1,
        "repo_id": repo_id,
        "dataset_root": str(root.resolve()),
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "fps": info["fps"],
        "stage_id": 8,
        "offset_seconds": 2.0,
        "timestamp_tolerance_s": TIMESTAMP_TOLERANCE_S,
        "episodes": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_boundaries(args.root, args.repo_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {len(manifest['episodes'])} episode boundaries to {args.output}")
    print(f"Episode 0: {manifest['episodes'][0]}")


if __name__ == "__main__":
    main()
