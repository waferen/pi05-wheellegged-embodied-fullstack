"""View an episode in Rerun after validating its data."""
import argparse
from pathlib import Path
import sys
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/ppsuc_rl_lab/ppsuc_rl_lab"))
from data_collection.episode import FPS, STATE_NAMES, COMMAND_NAMES, validate_stages


def check(path):
    with h5py.File(path, "r") as f:
        prompts, stages = validate_stages(f)
        state, command = f["state"][:], f["command"][:]
        n = len(state)
        assert n > 0 and state.shape == (n, 13) and command.shape == (n, 13), "Invalid dimensions"
        assert np.isfinite(state).all() and np.isfinite(command).all(), "Non-finite values"
        assert (np.abs(command[:, :2]) <= 1.00001).all(), "Base speed limit"
        limits = np.array([[-2.618,2.618],[0,3.14],[-2.967,0],[-1.745,1.745],[-1.22,1.22],[-2.0944,2.0944]])
        assert (command[:, 2:8] >= limits[:,0]-1e-5).all() and (command[:, 2:8] <= limits[:,1]+1e-5).all(), "Arm limits"
        assert ((command[:, 8] >= 0) & (command[:, 8] <= 1)).all(), "Gripper target"
        cameras = {}
        assert "front" in f["images"], "Missing front camera"
        for name, ds in f["images"].items():
            assert ds.shape[0] == n and ds.ndim == 4 and ds.shape[-1] == 3 and ds.dtype == np.uint8
            samples = ds[np.unique(np.linspace(0, n-1, min(n, 10), dtype=int))]
            assert samples.std() > 1, f"Blank image: {name}"
            cameras[name] = {"shape": list(ds.shape), "sample_std": float(samples.std()),
                             "first_last_mean_difference": float(np.abs(ds[-1].astype(float)-ds[0]).mean())}
        return {"path": str(path), "frames": n, "nominal_fps": FPS, "duration_seconds": n/FPS,
                "state_shape": list(state.shape), "command_shape": list(command.shape),
                "prompt": prompts.tolist(), "stage_ids": np.unique(stages).tolist(), "cameras": cameras,
                "note": "Cadence is enforced during capture; no timestamps are stored."}


def export(path, output=None, *, view=False, port=9880):
    import rerun as rr
    import rerun.blueprint as rrb
    rr.init("PPSUC2026 episode", spawn=False)
    if view:
        import os
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        rr.spawn(port=port)
    else:
        rr.save(str(output))
    with h5py.File(path, "r") as f:
        prompts, stages = validate_stages(f)
        image_views = [rrb.Spatial2DView(origin=f"images/{name}", name=name)
                       for name in ("front", "wrist")]
        def series_view(name, fields):
            return rrb.TimeSeriesView(name=name, origin="/", contents=fields)
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
            rrb.TextDocumentView(origin="task", name="Stage Instruction"),
            rrb.Horizontal(*image_views),
            rrb.Horizontal(
                series_view("Base (m/s, rad/s)", ["/state/vx", "/state/wz", "/command/vx", "/command/wz"]),
                series_view("Arm (rad)", [f"/{group}/joint_{i}" for group in ("state", "command") for i in range(1,7)])),
            rrb.Horizontal(
                series_view("Gripper [0,1]", ["/state/gripper", "/command/gripper"]),
                series_view("Control Modes", [f"/{group}/{name}" for group in ("state", "command") for name in STATE_NAMES[9:]])))))
        for i in range(len(f["command"])):
            rr.set_time_seconds("sim_time", i/FPS)
            if i == 0 or stages[i] != stages[i-1]:
                rr.log("task", rr.TextDocument(f"Stage {stages[i]}/14: {prompts[stages[i]-1]}"))
            for name, ds in f["images"].items():
                rr.log(f"images/{name}", rr.Image(ds[i]).compress(jpeg_quality=85))
            state, command = f["state"][i], f["command"][i]
            for j, name in enumerate(COMMAND_NAMES):
                rr.log(f"command/{name}", rr.Scalar(float(command[j])))
            for j, name in enumerate(STATE_NAMES):
                rr.log(f"state/{name}", rr.Scalar(float(state[j])))
    rr.disconnect()
    print("HDF5 loaded into Rerun viewer (no RRD written)" if view else f"Rerun recording: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--viewer-port", type=int, default=9880)
    args = parser.parse_args()
    check(args.episode)
    export(args.episode, view=True, port=args.viewer_port)
