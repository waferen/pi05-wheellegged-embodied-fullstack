"""Small contract checks: cadence, repeated commands, and interrupted output."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import h5py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/ppsuc_rl_lab/ppsuc_rl_lab"))
from data_collection.episode import EpisodeWriter, STATE_NAMES, STAGE_PROMPTS, MODES, mode_vector, validate_vectors
from view_episode import check

class CollectionContract(unittest.TestCase):
    def test_repeated_command_and_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            w = EpisodeWriter(directory)
            state, command = np.array([0.]*9 + mode_vector("walk"), np.float32), np.array([0.]*9 + mode_vector("walk"), np.float32)
            images = {"front": np.zeros((480, 640, 3), np.uint8)}
            images["wrist"] = images["front"].copy()
            w.append(1.00, state, command, images, 1)
            state[0] = 0.1
            w.append(1.02, state, command, images, 2)
            with self.assertRaises(ValueError):
                w.append(1.06, state, command, images, 2)
            w.close()
            with h5py.File(w.path) as f:
                np.testing.assert_array_equal(f["command"][0], f["command"][1])
                self.assertNotEqual(f["state"][0,0], f["state"][1,0])
                self.assertEqual(set(f.keys()), {"state", "images", "command", "prompt", "stage_id"})
            other = EpisodeWriter(directory)
            self.assertNotEqual(other.path, w.path)
            other.close()

    def test_state_schema_and_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            w = EpisodeWriter(directory)
            state = np.array([0.]*9 + mode_vector("walk"), np.float32)
            command = np.array([0.]*9 + mode_vector("walk"), np.float32)
            images = {"front": np.resize(np.arange(256, dtype=np.uint8), (480,640,3))}
            images["wrist"] = images["front"].copy()
            with self.assertRaises(ValueError):
                w.append(0., np.zeros(18), command, images, 1)
            for i in range(4):
                state[9:13] = np.eye(4)[i]
                command[9:13] = np.eye(4)[i]
                w.append(i / 50, state, command, images, i+1)
            w.close()
            self.assertEqual(len(STATE_NAMES), 13)
            self.assertFalse(any(name.startswith("enabled_") for name in STATE_NAMES))
            self.assertEqual(check(w.path)["state_shape"], [4, 13])
            with h5py.File(w.path) as f:
                np.testing.assert_array_equal(f["state"][:,9:13], np.eye(4))
            self.assertEqual(STATE_NAMES[9], "mode_walk")

    def test_stage_controls_and_legacy_rejection(self):
        from types import SimpleNamespace
        from data_collection.isaac_adapter import Collector
        from data_collection.episode import validate_stages
        class Arm:
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.zeros((1, 8), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(step_dt=.02, scene={"robot": object()})
            c = Collector(env, [], directory, .001, .035)
            c.next_stage()
            c.start()
            original = c.writer
            c.start()
            self.assertIs(c.writer, original)
            images = {name: np.zeros((480,640,3),np.uint8) for name in ("front", "wrist")}
            c.next_stage()
            c.command_submitted((0,0,0), Arm(), "walk")
            self.assertEqual(c.stage_id, 1)
            for stage in range(1, 15):
                c.writer.append((stage-1)/50, np.array([0.]*9 + mode_vector("walk")), np.array([0.]*9 + mode_vector("walk")), images, c._submitted_stage)
                c.next_stage()
                c.next_stage()
                self.assertEqual(c._submitted_stage, stage)
                c.command_submitted((0,0,0), Arm(), "walk")
                self.assertEqual(c.stage_id, min(stage+1,14))
            c.stop()
            with h5py.File(original.path) as f:
                prompts, ids = validate_stages(f)
                self.assertEqual(prompts.tolist(), list(STAGE_PROMPTS))
                np.testing.assert_array_equal(ids, np.arange(1,15))
            c.start()
            self.assertEqual(c.stage_id,1)
            c.stop()
            legacy = Path(directory)/"legacy.hdf5"
            with h5py.File(legacy,"w") as f:
                f.create_dataset("prompt",data="old")
            for reader in (check,):
                with self.assertRaisesRegex(ValueError,"legacy"):
                    reader(legacy)

    def test_modes_and_invalid_vectors(self):
        for i, mode in enumerate(MODES):
            np.testing.assert_array_equal(mode_vector(mode), np.eye(4)[i])
            state = np.array([0.]*9 + mode_vector(mode))
            command = state.copy()
            validate_vectors(state, command)
            if i >= 2:
                command[0] = .1
                with self.assertRaisesRegex(ValueError, "zero base"):
                    validate_vectors(state, command)
        with self.assertRaises(ValueError):
            mode_vector("stabilize")
        with self.assertRaises(ValueError):
            validate_vectors(np.zeros(16), np.zeros(10))
        with self.assertRaisesRegex(ValueError, "one-hot"):
            validate_vectors(np.zeros(13), np.zeros(13))

    def test_pre_step_observation_and_recording_boundaries(self):
        from types import SimpleNamespace
        from data_collection.isaac_adapter import Collector

        class Tensor(np.ndarray):
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.asarray(self)

        def tensor(values):
            return np.asarray(values, dtype=np.float32).view(Tensor)

        data = SimpleNamespace(joint_pos=tensor([[.1] * 8]),
                               root_lin_vel_b=tensor([[.2, 0, 0]]),
                               root_ang_vel_b=tensor([[0, 0, .3]]))
        rgb = {name: np.full((1, 480, 640, 3), 17, np.uint8).view(Tensor)
               for name in ('front', 'wrist')}
        scene = {'robot': SimpleNamespace(data=data)}
        scene.update({name + '_rgb': SimpleNamespace(data=SimpleNamespace(output={'rgb': value}))
                      for name, value in rgb.items()})
        env = SimpleNamespace(step_dt=.02, scene=scene)
        arm = tensor([[.4] * 8])
        with tempfile.TemporaryDirectory() as directory:
            c = Collector(env, list(range(8)), directory, 0., 1.)
            # F1 during a step must not adopt that step's earlier command.
            c.capture_observation(0., 'walk')
            c.command_submitted((1., 0., 0.), arm, 'walk')
            c.start()
            c.sample()
            self.assertEqual(c.writer.count, 0)
            path = c.writer.path
            c.capture_observation(.02, 'walk')
            c.command_submitted((0., 0., 0.), arm, 'low_grasp')
            # Simulate execution mutating live state, camera and target buffers.
            data.joint_pos[:] = .9
            data.root_lin_vel_b[:] = 9.
            for value in rgb.values(): value[:] = 99
            arm[:] = .8
            c.sample()
            c.sample()  # No duplicate frame.
            self.assertEqual(c.writer.count, 1)
            # F2 during execution labels the next command, not this pending frame.
            c.capture_observation(.04, 'low_grasp')
            c.command_submitted((0., 0., 0.), arm, 'low_grasp')
            c.next_stage()
            c.sample()
            c.capture_observation(.06, 'low_grasp')
            c.command_submitted((0., 0., 0.), arm, 'high_grasp')
            c.sample()
            # Stop/reset discards the in-flight pair. A restarted recording is clean.
            c.capture_observation(.08, 'high_grasp')
            c.command_submitted((0., 0., 0.), arm, 'high_grasp')
            c.stop()
            c.start()
            c.sample()
            self.assertEqual(c.writer.count, 0)
            c.stop()
            with h5py.File(path) as f:
                self.assertEqual(f['state'].shape, (3, 13))
                np.testing.assert_allclose(f['state'][0, :2], [.2, .3])
                np.testing.assert_allclose(f['state'][0, 2:9], [.1] * 7)
                np.testing.assert_allclose(f['command'][0, 2:9], [.4] * 7)
                np.testing.assert_array_equal(f['state'][0, 9:], mode_vector('walk'))
                np.testing.assert_array_equal(f['command'][0, 9:], mode_vector('low_grasp'))
                np.testing.assert_array_equal(f['stage_id'][:], [1, 1, 2])
                for name in rgb:
                    self.assertTrue(np.all(f['images/' + name][0] == 17))

    def test_interrupted_output(self):
        with tempfile.TemporaryDirectory() as directory:
            w = EpisodeWriter(directory)
            w.append(0., np.array([0.]*9 + mode_vector("walk")), np.array([0.]*9 + mode_vector("walk")), {name: np.zeros((480,640,3),np.uint8) for name in ("front", "wrist")}, 1)
            w.close(publish=False)
            self.assertTrue(w.partial.exists())
            self.assertFalse(w.path.exists())

if __name__ == "__main__":
    unittest.main()
