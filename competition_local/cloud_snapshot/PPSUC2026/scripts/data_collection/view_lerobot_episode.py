#!/usr/bin/env python3
"""Visualize a local LeRobot episode with synchronized stage annotations."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_episode(repo_id, root, episode_index):
    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode_index], video_backend="pyav")
    # Installed LeRobot v3 can load all rows despite episodes=[...]. Select only
    # this episode in memory; no parquet, video, or installed package is changed.
    episode_ids = np.asarray(dataset.hf_dataset.select_columns(["episode_index"])
                             .with_format("numpy")[:]["episode_index"]).reshape(-1)
    indices = np.flatnonzero(episode_ids == episode_index)
    dataset.hf_dataset = dataset.hf_dataset.select(indices.tolist())
    return dataset


def scalar(value):
    return np.asarray(value).reshape(-1).item()


class EpisodeAnnotations:
    """Read only numeric metadata; video decoding stays in LeRobotDataset."""

    def __init__(self, dataset, episode_index):
        columns = ("episode_index", "frame_index", "timestamp", "stage_id", "task_index")
        missing = set(columns) - set(dataset.hf_dataset.column_names)
        if missing:
            raise ValueError(f"Dataset lacks required columns: {sorted(missing)}")
        rows = dataset.hf_dataset.select_columns(list(columns)).with_format("numpy")[:]
        self.frames = np.asarray(rows["frame_index"]).reshape(-1).astype(np.int64)
        self.timestamps = np.asarray(rows["timestamp"]).reshape(-1).astype(float)
        self.stages = np.asarray(rows["stage_id"]).reshape(-1).astype(np.int64)
        task_ids = np.asarray(rows["task_index"]).reshape(-1).astype(np.int64)
        episodes = np.asarray(rows["episode_index"]).reshape(-1)
        self.episode_index = episode_index
        if not len(self.frames) or not np.all(episodes == episode_index):
            raise ValueError("Loaded data is empty or contains a different episode")
        if np.any(np.diff(self.frames) != 1) or not np.all(np.isfinite(self.timestamps)):
            raise ValueError("Episode frames must be consecutive and timestamps finite")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Episode timestamps must be strictly increasing")
        self.tasks = {int(t): str(dataset.meta.tasks.iloc[int(t)].name) for t in np.unique(task_ids)}
        self.task_ids = task_ids
        starts = np.r_[0, np.flatnonzero(self.stages[1:] != self.stages[:-1]) + 1]
        self.stage_start = np.maximum.accumulate(
            np.where(np.isin(np.arange(len(self.frames)), starts), np.arange(len(self.frames)), 0)
        )

    def text(self, index):
        frame = int(self.frames[index])
        timestamp = float(self.timestamps[index])
        stage = int(self.stages[index])
        task_id = int(self.task_ids[index])
        stage_elapsed = timestamp - self.timestamps[self.stage_start[index]]
        lines = [
            f"## Stage {stage}",
            f"Episode: **{self.episode_index}** | Frame: **{frame}** | Timestamp: **{timestamp:.2f} s**  ",
            f"Episode elapsed: **{timestamp - self.timestamps[0]:.2f} s** | Stage elapsed: **{stage_elapsed:.2f} s** | task_index: **{task_id}**",
            "", "**Dataset instruction**", "", self.tasks[task_id],
            "", "> Dataset stage labels are annotations. The Full policy uses one fixed instruction throughout the task.",
        ]
        return "\n".join(lines)


def image_uint8(value):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"Expected RGB image in CHW layout, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError(f"Unsupported image dtype: {array.dtype}")
    return np.ascontiguousarray(array.transpose(1, 2, 0))


def feature_names(dataset, key):
    feature = dataset.meta.features[key]
    size = feature["shape"][0]
    names = feature.get("names")
    return names if isinstance(names, list) and len(names) == size else [str(i) for i in range(size)]


def send_blueprint(dataset):
    def plot(name, paths):
        return rrb.TimeSeriesView(name=name, origin="/", contents=paths)
    state = [f"/state/{name}" for name in feature_names(dataset, "observation.state")]
    action = [f"/action/{name}" for name in feature_names(dataset, "action")]
    stages = ["/annotation/stage_id"]
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
        rrb.TextDocumentView(origin="/annotation/text", name="Stage & Instruction"),
        rrb.Horizontal(*[rrb.Spatial2DView(origin=f"/{key}", name=key.rsplit(".", 1)[-1])
                         for key in dataset.meta.camera_keys]),
        rrb.Horizontal(plot("Stage", stages), plot("State", state), plot("Action", action)),
        row_shares=[3, 5, 2],
    )))


def log_sample(dataset, annotations, index, sample):
    # Local row indices address the filtered dataset; timestamps come from the sample.
    if int(scalar(sample["episode_index"])) != annotations.episode_index:
        raise ValueError("Sample episode does not match requested episode")
    if int(scalar(sample["frame_index"])) != int(annotations.frames[index]):
        raise ValueError("Sample frame and annotation frame do not match")
    if str(sample["task"]) != annotations.tasks[int(annotations.task_ids[index])]:
        raise ValueError("Sample instruction and metadata do not match")
    rr.set_time_sequence("frame_index", int(scalar(sample["frame_index"])))
    rr.set_time_seconds("timestamp", float(scalar(sample["timestamp"])))
    rr.log("annotation/text", rr.TextDocument(annotations.text(index), media_type=rr.MediaType.MARKDOWN))
    rr.log("annotation/stage_id", rr.Scalar(int(annotations.stages[index])))
    for key in dataset.meta.camera_keys:
        rr.log(key, rr.Image(image_uint8(sample[key])).compress(jpeg_quality=85))
    for key, group in (("observation.state", "state"), ("action", "action")):
        for name, value in zip(feature_names(dataset, key), sample[key].reshape(-1), strict=True):
            rr.log(f"{group}/{name}", rr.Scalar(float(value)))
    for key in ("next.done", "next.reward", "next.success"):
        if key in sample:
            rr.log(key, rr.Scalar(float(scalar(sample[key]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True, help="Existing local LeRobot dataset")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--viewer-port", type=int, default=9881)
    parser.add_argument("--save", type=Path, metavar="FILE.rrd", help="Export an RRD instead of opening a viewer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_threads < 1:
        parser.error("--batch-size and --num-threads must be positive")
    if not (args.root / "meta/info.json").is_file():
        parser.error("--root must point to an existing local LeRobot dataset")
    info = json.loads((args.root / "meta/info.json").read_text())
    if not 0 <= args.episode_index < info["total_episodes"]:
        parser.error(f"--episode-index must be between 0 and {info['total_episodes'] - 1}")
    if args.save is not None and args.save.exists():
        parser.error(f"Refusing to overwrite existing recording: {args.save}")
    torch.set_num_threads(args.num_threads)
    dataset = load_episode(args.repo_id, args.root, args.episode_index)
    annotations = EpisodeAnnotations(dataset, args.episode_index)
    if len(dataset) != len(annotations.frames):
        raise ValueError("Filtered dataset length does not match the episode metadata")
    rr.init(f"PPSUC2026 LeRobot/{args.repo_id}/episode_{args.episode_index}", spawn=False)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save)
    else:
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        rr.spawn(port=args.viewer_port)
    try:
        send_blueprint(dataset)
        # No global EpisodeSampler here: load_episode has explicitly filtered the rows.
        # Decode in the main process to avoid PyAV / multiprocessing fork conflicts.
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        index = 0
        for batch in tqdm(loader, desc=f"Episode {args.episode_index}"):
            for offset in range(len(batch["frame_index"])):
                sample = {key: value[offset] for key, value in batch.items()}
                log_sample(dataset, annotations, index, sample)
                if index == 0 or annotations.stages[index] != annotations.stages[index - 1]:
                    print(f"\nframe={annotations.frames[index]} time={annotations.timestamps[index]:.2f}s "
                          f"stage_id={annotations.stages[index]} task_index={annotations.task_ids[index]}\n"
                          f"{sample['task']}", flush=True)
                index += 1
    finally:
        rr.disconnect()
    print(f"Saved: {args.save}" if args.save else "Loaded into Rerun. Use the timeline to replay or seek.")


if __name__ == "__main__":
    main()
