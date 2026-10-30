"""Checks for recording alignment, episode isolation and training selection."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from .schema import mode_vector
from .schema import EpisodeWriter,validate_stages
TEST_ROOT=Path(__file__).resolve().parents[1]/"runs/collection_unit_tests"
TEST_ROOT.mkdir(parents=True,exist_ok=True)
from .adapter import MujocoCollector,MujocoWriter
from .pipeline import inspect_episode,selected_sources
import h5py,numpy as np,torch
from .image_storage import open_episode


def environment():
    d=NS(root_lin_vel_b=torch.tensor([[.2,0.,0.]]),root_ang_vel_b=torch.tensor([[0.,0.,.3]]),
         joint_pos=torch.tensor([[.1,.2,-.3,.4,-.5,.6,.023,-.023]]))
    image=torch.zeros((1,480,640,3),dtype=torch.uint8)
    return NS(step_dt=.02,scene={'robot':NS(data=d),
        **{name:NS(data=NS(output={'rgb':image.clone()})) for name in ('front_rgb','wrist_rgb')}})


class Pipeline(unittest.TestCase):
    def collector(self,directory):return MujocoCollector(environment(),directory,Path(directory)/'status.json',{'purpose':'demonstration'})

    def frame(self,c,t=.0):
        c.capture_observation(t,'walk');c.command_submitted((.4,0.,-.2),torch.tensor([[.4,.5,-.6,.7,-.8,.9,.045,-.045]]),'walk');c.sample()

    def test_pre_step_measurements_and_actual_submitted_targets(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            c=self.collector(directory);c.start();c.capture_observation(5.,'walk')
            targets=torch.tensor([[.4,.5,-.6,.7,-.8,.9,.045,-.045]])
            c.command_submitted((.4,0.,-.2),targets,'walk')
            targets[:]=0;c.robot.data.joint_pos[:]=9;c.env.scene['front_rgb'].data.output['rgb'][:]=255
            c.sample();c.stop()
            with open_episode(c.last_episode) as f:
                self.assertAlmostEqual(float(f['state'][0,2]),.1,places=6)
                self.assertAlmostEqual(float(f['command'][0,2]),.4,places=6)
                self.assertEqual(int(f['images/front'][0].max()),0)
                self.assertEqual(float(f['timestamp'][0]),0.);self.assertEqual(float(f['sim_time'][0]),5.)
                self.assertEqual(f.attrs['result'],'unreviewed')

    def test_stage_applies_next_submitted_action(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            c=self.collector(directory);c.start();c.next_stage();self.assertFalse(c._pending_stage)
            self.frame(c);c.next_stage();c.next_stage();self.frame(c,.02);c.stop()
            with h5py.File(c.last_episode,'r') as f:np.testing.assert_array_equal(f['stage_id'][:],[1,2])

    def test_reset_pending_does_not_leak_into_next_episode(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            c=self.collector(directory);c.start();self.frame(c)
            c.capture_observation(.02,'walk');c.command_submitted((0,0,0),torch.zeros((1,8)),'walk')
            c.stop(False);c.start();c.sample();self.assertEqual(c.writer.count,0)
            self.frame(c,0.);c.stop()
            self.assertEqual(len(list(Path(directory).glob('*.partial'))),1)
            self.assertEqual(inspect_episode(c.last_episode)['frames'],1)

    def test_non_50hz_refused(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            c=self.collector(directory);c.start();self.frame(c)
            with self.assertRaisesRegex(ValueError,'Sampling gap'):self.frame(c,.04)
            c.stop(False);self.assertFalse(list(Path(directory).glob('*.hdf5')))

    def test_training_selection_requires_review_and_phase_boundary(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            c=self.collector(directory);c.start();self.frame(c);c.stop()
            with self.assertRaises(ValueError):selected_sources(directory)
            with self.assertRaisesRegex(ValueError,'stage 14'):inspect_episode(c.last_episode,True)

    def test_full_schema_and_test_data_filter(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            w=MujocoWriter(directory,{'purpose':'smoke_test'})
            vector=np.array([0.]*9+mode_vector('walk'),np.float32)
            images={name:np.zeros((480,640,3),np.uint8) for name in ('front','wrist')}
            count=0
            for stage in range(1,15):
                for _ in range(105 if stage==8 else 1):
                    w.append(count*.02,vector,vector,images,stage);count+=1
            w.close();record=inspect_episode(w.path,True)
            self.assertEqual(record['phase1_start_frame'],107)
            with self.assertRaises(ValueError):selected_sources(directory,True,False)
            selected,_=selected_sources(directory,True,True);self.assertEqual(len(selected),1)

    def test_jpeg_payload_decodes_and_keeps_numeric_fields_exact(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            w=MujocoWriter(directory,{'purpose':'smoke_test'})
            v=np.array([0.]*9+mode_vector('walk'),np.float32)
            image=np.random.default_rng(42).integers(0,256,(480,640,3),dtype=np.uint8)
            w.append(0.,v,v,{'front':image,'wrist':image},1);w.close()
            with h5py.File(w.path,'r') as raw:
                self.assertEqual(raw['images/front'].shape,(1,))
                self.assertEqual(raw['images/front'].attrs['encoding'],'jpeg')
                np.testing.assert_array_equal(raw['state'][0],v)
                np.testing.assert_array_equal(raw['command'][0],v)
            with open_episode(w.path) as f:
                restored=f['images/front'][0]
                self.assertEqual(restored.shape,image.shape);self.assertEqual(restored.dtype,image.dtype)
                self.assertLess(np.abs(restored.astype(float)-image).mean(),8.)
                self.assertEqual(f['images/front'][:].shape,(1,480,640,3))
            with h5py.File(w.path,'r+') as raw:raw['images/front'][0]=np.array([1,2,3],np.uint8)
            with self.assertRaises((OSError,ValueError)):inspect_episode(w.path)

    def test_original_lzf_hdf5_stays_readable(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            w=EpisodeWriter(directory);v=np.array([0.]*9+mode_vector('walk'),np.float32)
            image=np.full((480,640,3),123,np.uint8)
            w.append(0.,v,v,{'front':image,'wrist':image},1);w.close()
            with open_episode(w.path) as f:
                validate_stages(f);np.testing.assert_array_equal(f['images/front'][0],image)
                self.assertEqual(f['images/front'].shape,(1,480,640,3))


if __name__=='__main__':unittest.main(verbosity=2)
