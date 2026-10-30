"""HDF5 schema and fixed-simulation-time sampling; no Isaac dependency."""
from pathlib import Path
import h5py
import numpy as np
from .stages import STAGE_PROMPTS, STAGE_COUNT, STAGE_VERSION

FPS = 50

MODES = ("walk", "stair", "low_grasp", "high_grasp")
STATE_NAMES = ("vx", "wz", *[f"joint_{i}" for i in range(1, 7)], "gripper",
               *[f"mode_{mode}" for mode in MODES])
COMMAND_NAMES = STATE_NAMES


def mode_vector(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown control mode: {mode}")
    return [float(mode == name) for name in MODES]


def validate_vectors(state, command):
    if state.shape[-1:] != (13,) or command.shape != state.shape:
        raise ValueError("Expected matching state[...,13] and command[...,13]")
    for values in (state, command):
        if not np.isfinite(values).all():
            raise ValueError("Non-finite state/command")
        flags = values[..., 9:13]
        if not np.isin(flags, [0, 1]).all() or not (flags.sum(axis=-1) == 1).all():
            raise ValueError("Mode must be four-way one-hot")
    grasp = command[..., 11:13].sum(axis=-1) == 1
    if np.any(np.abs(command[..., :2][grasp]) > 1e-6):
        raise ValueError("Grasp mode requires zero base command")


class EpisodeWriter:
    """Append bounded-size frames; publish a complete file only on close."""
    def __init__(self, directory):
        if not all(isinstance(p, str) and p.strip() and p.isascii() for p in STAGE_PROMPTS):
            raise ValueError("Dataset instructions must be nonempty ASCII English text")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index = 0
        while True:
            self.path = directory / f"episode_{index:06d}.hdf5"
            self.partial = self.path.with_suffix(".hdf5.partial")
            if not self.path.exists():
                try:
                    self.file = h5py.File(self.partial, "x")
                    break
                except FileExistsError:
                    pass
            index += 1
        self.file.create_dataset("prompt", data=STAGE_PROMPTS, dtype=h5py.string_dtype("utf-8"))
        self.file.attrs["stage_schema_version"] = STAGE_VERSION
        self.count = 0
        self.last_time = None
        self.keys = None

    def append(self, sim_time, state, command, images, stage_id):
        state, command = np.asarray(state, np.float32), np.asarray(command, np.float32)
        if state.shape != (13,) or command.shape != (13,):
            raise ValueError("Expected state[13] and command[13]")
        validate_vectors(state, command)
        if self.last_time is not None and not np.isclose(sim_time-self.last_time, 1/FPS, atol=1e-7, rtol=0):
            raise ValueError("Sampling gap: refusing a non-50Hz episode")
        if not isinstance(stage_id, (int, np.integer)) or not 1 <= stage_id <= len(STAGE_PROMPTS):
            raise ValueError("Invalid stage_id")
        previous = int(self.file["stage_id"][-1]) if self.count else 1
        if stage_id not in (previous, previous + 1) or (not self.count and stage_id != 1):
            raise ValueError("Stages must start at 1 and advance one at a time")
        if set(images) != {"front", "wrist"}:
            raise ValueError("Both native camera images are required")
        arrays = {"state": state, "command": command, "stage_id": np.asarray(stage_id, np.int32)}
        if "front" not in images:
            raise ValueError("Front RGB is required")
        for name, value in images.items():
            value = np.asarray(value)
            if value.dtype != np.uint8 or value.shape != (480, 640, 3):
                raise ValueError(f"Invalid RGB: {name}")
            arrays[f"images/{name}"] = value
        if self.keys is not None and set(arrays) != self.keys:
            raise ValueError("Camera set changed within episode")
        for name, value in arrays.items():
            if self.count == 0:
                self.file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                         dtype=value.dtype, chunks=(1, *value.shape), compression="lzf")
            dataset = self.file[name]
            if dataset.shape[1:] != value.shape:
                raise ValueError(f"Shape changed: {name}")
            dataset.resize(self.count+1, axis=0)
            dataset[self.count] = value
        self.keys = set(arrays)
        self.last_time = sim_time
        self.count += 1

    def close(self, publish=True):
        if not self.file.id.valid:
            return
        self.file.close()
        if self.count and publish:
            self.partial.rename(self.path)
            print(f"[RECORD] saved {self.path} frames={self.count} fps={FPS}", flush=True)
        elif not self.count:
            self.partial.unlink()
        else:
            print(f"[RECORD] incomplete: {self.partial}", flush=True)


def validate_stages(file):
    """Require the new format; deliberately reject legacy episodes."""
    if "stage_id" not in file or file["prompt"].shape != (STAGE_COUNT,):
        raise ValueError("New format required: prompt[14] and stage_id[T]; legacy files are unsupported")
    prompts = file["prompt"].asstr()[:]
    ids = file["stage_id"][:]
    n = len(file["command"])
    if file["state"].shape != (n, 13) or file["command"].shape != (n, 13):
        raise ValueError("Only four-mode state[T,13], command[T,13] are supported")
    validate_vectors(file["state"][:], file["command"][:])
    if not all(isinstance(p, str) and p.strip() and p.isascii() for p in prompts):
        raise ValueError("Dataset instructions must be nonempty ASCII English text")
    if ids.shape != (n,) or ids.dtype != np.dtype("int32") or not n:
        raise ValueError("Expected nonempty stage_id[T] int32")
    if ids[0] != 1 or np.any((ids < 1) | (ids > len(prompts))) or not np.isin(np.diff(ids), [0, 1]).all():
        raise ValueError("Invalid stage sequence")
    if set(file["images"]) != {"front", "wrist"}:
        raise ValueError("Both native camera streams are required")
    for ds in file["images"].values():
        if ds.shape != (n, 480, 640, 3) or ds.dtype != np.dtype("uint8"):
            raise ValueError("Expected RGB images[T,480,640,3] uint8")
    return prompts, ids
