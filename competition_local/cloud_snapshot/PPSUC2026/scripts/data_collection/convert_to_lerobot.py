#!/usr/bin/env python3
"""Convert selected HDF5 folders into separate verified local LeRobot v3 datasets.

Run with the lerobot Conda environment. Source HDF5 files are always read-only.
No Hub upload, simulation startup, or training is performed.
"""
import argparse
import json
import sys
import types
from pathlib import Path
import time

import av
import h5py
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path('/root/gpufree-data/datasets')
OUTPUT_ROOT = Path('/root/gpufree-data/lerobot_datasets')
DEFAULT_DATASETS = ('lyq',)
SOURCES = ()
REPO_ID = ''
CAMERAS = ('front', 'wrist')

# Reuse the project's schema validator without importing Isaac Lab.
# episode.py has a relative stages import; load it through a minimal private package.
package = types.ModuleType('_ppsuc_conversion_schema')
package.__path__ = [str(PROJECT / 'source/ppsuc_rl_lab/ppsuc_rl_lab/data_collection')]
sys.modules[package.__name__] = package
from _ppsuc_conversion_schema.episode import FPS, STATE_NAMES, validate_stages


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_sources():
    require(bool(SOURCES), 'No HDF5 files found in the selected source folder')
    records = []
    for path in SOURCES:
        with h5py.File(path, 'r') as f:
            prompts, stages = validate_stages(f)
            require(f.attrs.get('stage_schema_version') == 1, f'Unsupported schema: {path}')
            require(stages[-1] == 14, f'Incomplete stage sequence: {path}')
            records.append(dict(path=str(path), frames=len(stages), size=path.stat().st_size,
                                mtime_ns=path.stat().st_mtime_ns, prompts=prompts.tolist()))
    return records


def wait_for_encoder(dataset):
    # LeRobot 0.6.2 feed_frame drops frames after a 0.1 s full-queue timeout.
    # This is the only producer: wait before add_frame so every camera has room.
    encoder = dataset.writer._streaming_encoder
    deadline = time.monotonic() + 120
    for key, queue in encoder._frame_queues.items():
        while queue.full():
            require(encoder._threads[key].is_alive(), f'Encoder stopped: {key}')
            require(time.monotonic() < deadline, f'Encoder stalled: {key}')
            time.sleep(0.005)


def update_quantiles(root):
    path = root / 'meta/stats.json'
    stats = json.loads(path.read_text())
    for source_key, feature in [('state', 'observation.state'), ('command', 'action')]:
        arrays = []
        for source in SOURCES:
            with h5py.File(source, 'r') as f:
                arrays.append(f[source_key][:])
        values = np.concatenate(arrays)
        # Exact full-dataset quantiles, rather than aggregated episode envelopes.
        keys = {k for k in stats[feature] if k.startswith('q') and k[1:].isdigit()} | {'q01', 'q99'}
        for key in keys:
            stats[feature][key] = np.quantile(values, int(key[1:]) / 100, axis=0).tolist()
    path.write_text(json.dumps(stats, indent=2) + '\n')


def validate_output(root, records):
    dataset = LeRobotDataset(REPO_ID, root=root, video_backend='torchcodec')
    expected = sum(r['frames'] for r in records)
    require(len(dataset) == expected, 'Total frame count mismatch')
    require(dataset.meta.total_episodes == len(records), 'Episode count mismatch')
    table = dataset.hf_dataset.with_format('numpy')
    report = {'frames': expected, 'episodes': len(records), 'fps': FPS, 'videos': [], 'samples': []}
    offset = 0
    for episode, (source, record) in enumerate(zip(SOURCES, records, strict=True)):
        n = record['frames']
        rows = table[offset:offset+n]
        with h5py.File(source, 'r') as f:
            for key, original in [('observation.state', 'state'), ('action', 'command')]:
                require(np.array_equal(rows[key], f[original][:]), f'{key} values differ in episode {episode}')
            stages = f['stage_id'][:]
            require(np.array_equal(np.asarray(rows['stage_id']).reshape(-1), stages), 'Stage mismatch')
            require(np.all(rows['episode_index'] == episode), 'Episode indices mismatch')
            require(np.array_equal(rows['frame_index'], np.arange(n)), 'Frame indices mismatch')
            require(np.array_equal(rows['timestamp'], (np.arange(n)/FPS).astype(np.float32)), 'Timestamp mismatch')
            prompts = f['prompt'].asstr()[:]
            # Task lookup comes from LeRobot's own metadata, not numeric stage assumptions.
            for stage in np.unique(stages):
                indices = np.flatnonzero(stages == stage)
                task_ids = np.unique(rows['task_index'][indices])
                require(len(task_ids) == 1, 'Multiple task IDs within a stage')
                sample = dataset[offset + int(indices[0])]
                require(sample['task'] == prompts[stage-1], 'Task text mismatch')
            boundaries = np.flatnonzero(np.diff(stages))
            samples = sorted({0, n-1, *boundaries.tolist(), *(boundaries+1).tolist()})
            for index in samples:
                sample = dataset[offset + index]
                errors = {}
                for camera in CAMERAS:
                    image = sample[f'observation.images.{camera}'].numpy().transpose(1, 2, 0)
                    require(image.shape == (480, 640, 3), 'Decoded image shape mismatch')
                    error = float(np.abs(image*255 - f[f'images/{camera}'][index].astype(np.float32)).mean())
                    require(error < 15, f'Unexpected video mismatch: episode {episode}, frame {index}, {camera}, MAE={error}')
                    errors[camera] = error
                report['samples'].append({'episode': episode, 'frame': index, 'rgb_mae': errors})
        offset += n
    # Decode every video frame, not just container metadata, to catch truncation.
    for camera in CAMERAS:
        total = 0
        for path in sorted((root/'videos'/f'observation.images.{camera}').rglob('*.mp4')):
            count = 0
            with av.open(str(path)) as container:
                require(float(container.streams.video[0].average_rate) == FPS, 'Video FPS mismatch')
                for frame in container.decode(video=0):
                    require((frame.width, frame.height) == (640, 480), 'Video resolution mismatch')
                    count += 1
            total += count
            report['videos'].append({'path': str(path.relative_to(root)), 'frames': count})
        require(total == expected, f'Video frame count mismatch for {camera}: {total}')
    sequence = LeRobotDataset(REPO_ID, root=root, video_backend='torchcodec',
                             delta_timestamps={'action': [i/FPS for i in range(50)]})
    offset = 0
    for record in records:
        first = sequence[offset]
        last = sequence[offset + record['frames']-1]
        require(tuple(first['action'].shape) == (50, 13), 'Action sequence shape mismatch')
        require(int(last['action_is_pad'].sum()) == 49, 'Action padding crosses episode boundary')
        offset += record['frames']
    for source, record in zip(SOURCES, records, strict=True):
        require(source.stat().st_size == record['size'] and source.stat().st_mtime_ns == record['mtime_ns'],
                f'Source changed while converting: {source}')
    report['validation'] = 'passed'
    return report


def convert_dataset(source_dir, root):
    global SOURCES, REPO_ID
    SOURCES = tuple(sorted(source_dir.glob('*.hdf5')))
    REPO_ID = f'local/{source_dir.name}'
    root = root.resolve()
    print(f'Converting {source_dir} -> {root} ({REPO_ID})', flush=True)
    pending = root.with_name(root.name + '.partial')
    require(not root.exists(), f'Output already exists: {root}')
    require(not pending.exists(), f'Previous incomplete output exists: {pending}')
    records = inspect_sources()
    root.parent.mkdir(parents=True, exist_ok=True)
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (13,), 'names': list(STATE_NAMES)},
        'action': {'dtype': 'float32', 'shape': (13,), 'names': list(STATE_NAMES)},
        'stage_id': {'dtype': 'int32', 'shape': (1,), 'names': ['stage_id']},
    }
    for camera in CAMERAS:
        features[f'observation.images.{camera}'] = {'dtype': 'video', 'shape': (480, 640, 3),
                                                    'names': ['height', 'width', 'channels']}
    started = time.monotonic()
    dataset = LeRobotDataset.create(REPO_ID, fps=FPS, features=features, root=pending,
        robot_type='ppsuc_b2w_piper', use_videos=True, streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec='h264', crf=18, preset='fast', g=2),
        encoder_threads=2, encoder_queue_maxsize=32)
    try:
        for episode, (source, record) in enumerate(zip(SOURCES, records, strict=True)):
            with h5py.File(source, 'r') as f:
                for index in range(record['frames']):
                    wait_for_encoder(dataset)
                    stage = int(f['stage_id'][index])
                    frame = {'observation.state': f['state'][index], 'action': f['command'][index],
                             'stage_id': np.array([stage], dtype=np.int32), 'task': record['prompts'][stage-1]}
                    frame.update({f'observation.images.{camera}': f[f'images/{camera}'][index] for camera in CAMERAS})
                    dataset.add_frame(frame)
                    if (index+1) % 500 == 0:
                        print(f'episode={episode} frames={index+1}/{record["frames"]}', flush=True)
                dataset.save_episode()
                print(f'episode={episode} saved', flush=True)
    finally:
        dataset.finalize()
    update_quantiles(pending)
    print('Validating converted dataset...', flush=True)
    report = validate_output(pending, records)
    report.update(sources=records, elapsed_seconds=round(time.monotonic()-started, 2),
                  output=str(root), encoding={'codec': 'h264', 'crf': 18, 'pixel_format': 'yuv420p'},
                  bytes=sum(p.stat().st_size for p in pending.rglob('*') if p.is_file()))
    pending.rename(root)
    print(json.dumps({k: report[k] for k in ['output', 'frames', 'episodes', 'validation', 'elapsed_seconds', 'bytes']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS),
                        help='Source folder names under /root/gpufree-data/datasets')
    parser.add_argument('--output', type=Path,
                        help='Override the output directory when converting one dataset')
    args = parser.parse_args()
    require(len(set(args.datasets)) == len(args.datasets), 'Duplicate dataset names')
    require(args.output is None or len(args.datasets) == 1, '--output requires exactly one dataset')
    jobs = []
    # Check every destination before starting the first conversion.
    for name in args.datasets:
        require(name not in ('.', '..') and Path(name).name == name, f'Invalid dataset name: {name}')
        source = SOURCE_ROOT / name
        root = (args.output or OUTPUT_ROOT / name).resolve()
        require(source.is_dir() and any(source.glob('*.hdf5')), f'No HDF5 files found in {source}')
        require(not root.exists(), f'Output already exists: {root}')
        require(not root.with_name(root.name + '.partial').exists(),
                f'Previous incomplete output exists: {root.name}.partial')
        jobs.append((source, root))
    for source, root in jobs:
        convert_dataset(source, root)


if __name__ == '__main__':
    main()
