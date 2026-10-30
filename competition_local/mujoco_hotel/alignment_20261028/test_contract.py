"""Regression checks for the source contract and asynchronous takeover boundary."""
import concurrent.futures
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from policy_interface import PolicyInterface, MODES, decode_action
from full_client import FullClient
from original_teleop import OriginalTeleop
from pi05_teleop import Pi05Teleop
from material_resolver import variant_values, resolve
from pxr import Usd, UsdShade


class Remote:
    def close(self): pass


def env(opening=.023):
    q = torch.tensor([[.1, .2, -.3, .4, -.5, .6, opening, -opening]])
    d = NS(joint_pos=q, root_lin_vel_b=torch.tensor([[.7, -.2, .3]]),
           root_ang_vel_b=torch.tensor([[.2, .3, -.4]]))
    image = torch.arange(480 * 640 * 3, dtype=torch.int32).to(torch.uint8).reshape(1, 480, 640, 3)
    return NS(scene={'robot': NS(data=d), 'front_rgb': NS(data=NS(output={'rgb': image})),
                     'wrist_rgb': NS(data=NS(output={'rgb': image.flip(2)}))})


def prediction(mode=0, phase=0):
    a = np.zeros((50, 14));a[:, 9 + mode] = 1;a[:, 13] = phase
    a[:, :2] = [1.5, -1.5];a[:, 2:8] = [.1, .2, -.3, .4, -.5, .6];a[:, 8] = .5
    return {'actions': a}


class Contract(unittest.TestCase):
    def new(self, steps=25): return PolicyInterface(execute_steps=steps, remote=Remote())

    def test_observations_match_unchanged_source(self):
        for mode in MODES:
            for opening in (.001, .023, .045):
                captured = {}
                def infer(obs):
                    captured.update(obs)
                    return {'actions': prediction()['actions'][0]}
                c = object.__new__(FullClient)
                c.ids=list(range(8));c.closed=.001;c.opened=.045;c.vx=c.wz=1.
                c.task_phase=1;c.model_control=True;c._control_generation=0
                c.broker=NS(infer=infer, reset=lambda: None)
                e=env(opening);c.step(e,mode)
                local=self.new();local.task_phase=1;actual=local.observation(e,mode)
                self.assertEqual(set(captured),set(actual))
                for key in captured:np.testing.assert_equal(actual[key],captured[key])

    def test_chunk_horizon_and_decode(self):
        for horizon in (1,25,50):
            for mode in range(4):
                c=self.new(horizon);p=prediction(mode)
                self.assertTrue(c.accept_prediction(c.generation,p))
                for _ in range(horizon):
                    token=c.take_action('walk');expected=decode_action(p['actions'][0])
                    for a,b in zip(token[0],expected):np.testing.assert_equal(a,b)
                self.assertFalse(c.actions)

    def test_phase_only_after_executed_row_and_replan(self):
        c=self.new();c.accept_prediction(0,prediction(phase=1));token=c.take_action('walk')
        self.assertEqual(c.task_phase,0);self.assertTrue(c.actions)
        c.commit(token);self.assertEqual(c.task_phase,1);self.assertFalse(c.actions)
        c.accept_prediction(c.generation,prediction(phase=0));c.commit(c.take_action('walk'))
        self.assertEqual(c.task_phase,1)

    def test_future_row_cannot_advance_phase(self):
        c=self.new();p=prediction();p['actions'][1,13]=1;c.accept_prediction(0,p)
        c.commit(c.take_action('walk'));self.assertEqual(c.task_phase,0)
        c.commit(c.take_action('walk'));self.assertEqual(c.task_phase,1)

    def test_prefetch_skips_elapsed_rows_without_latching_their_phase(self):
        c=self.new();p=prediction();p['actions'][:15,13]=1
        p['actions'][:,0]=np.arange(50)/100
        self.assertTrue(c.accept_prediction(0,p,start_index=15))
        token=c.take_action('walk');self.assertAlmostEqual(token[0][0][0],.15)
        c.commit(token);self.assertEqual(c.task_phase,0);self.assertEqual(len(c.actions),24)
        self.assertFalse(c.accept_prediction(0,p,start_index=50))

    def test_pending_prefetch_does_not_pause_current_actions(self):
        sim=object.__new__(Pi05Teleop);sim.client=self.new();sim.mode='walk'
        sim.client.accept_prediction(0,prediction());sim.future=concurrent.futures.Future()
        ready,token=sim._take_prediction()
        self.assertTrue(ready);self.assertIsNotNone(token);self.assertEqual(len(sim.client.actions),24)

    def test_ready_prefetch_aligns_to_execution_time(self):
        sim=object.__new__(Pi05Teleop);sim.client=self.new();sim.mode='walk'
        sim.executed_steps=40;sim.prediction_start_step=25;sim.prefetch_steps=15;sim.inference_count=1
        p=prediction();p['actions'][:,0]=np.arange(50)/100
        sim.future=concurrent.futures.Future();sim.future.set_result((0,p,.2))
        ready,token=sim._take_prediction()
        self.assertTrue(ready);self.assertAlmostEqual(token[0][0][0],.15)
        self.assertEqual(sim.prediction_offset_steps,15)

    def test_takeover_before_commit_drops_phase(self):
        c=self.new();c.accept_prediction(0,prediction(phase=1));token=c.take_action('walk')
        c.set_model_control(False);c.commit(token);self.assertEqual(c.task_phase,0)

    def test_takeover_resume_rejects_inflight_prediction(self):
        c=self.new();generation=c.generation
        c.set_model_control(False);c.set_model_control(True)
        self.assertFalse(c.accept_prediction(generation,prediction(phase=1)))
        self.assertEqual(c.task_phase,0);self.assertFalse(c.actions)

    def test_grasp_gate_rejects_entire_row(self):
        for mode in ('low_grasp','high_grasp'):
            for next_mode in range(4):
                c=self.new();c.accept_prediction(0,prediction(next_mode,1));token=c.take_action(mode)
                if MODES[next_mode] not in (mode,'walk'):
                    self.assertIsNone(token);self.assertEqual(c.task_phase,0);self.assertFalse(c.actions)
                else:self.assertIsNotNone(token)

    def test_manual_phase_and_reset_invalidate(self):
        c=self.new();c.toggle_task_phase();self.assertEqual(c.task_phase,0)
        c.set_model_control(False);c.toggle_task_phase();self.assertEqual(c.task_phase,1)
        generation=c.generation;c.reset();self.assertEqual(c.task_phase,0)
        self.assertGreater(c.generation,generation)

    def test_idempotent_control_preserves_buffer(self):
        c=self.new();c.accept_prediction(0,prediction());c.set_model_control(True)
        self.assertEqual(len(c.actions),25);self.assertEqual(c.generation,0)

    def test_bad_predictions(self):
        for a in (np.zeros((49,14)),np.full((50,14),np.nan)):
            with self.assertRaises(ValueError):self.new().accept_prediction(0,{'actions':a})
        for steps in (0,51,True,2.5):
            with self.assertRaises(ValueError):self.new(steps)

    def test_adapter_stale_future_does_not_execute(self):
        sim=object.__new__(Pi05Teleop);sim.client=self.new();sim.mode='walk';sim.inference_count=0
        future=concurrent.futures.Future();future.set_result((0,prediction(phase=1),.1));sim.future=future
        sim.client.set_model_control(False);sim.client.set_model_control(True)
        self.assertEqual(sim._take_prediction(),(False,None));self.assertEqual(sim.client.task_phase,0)

    def test_repeated_mode_preserves_home(self):
        sim=object.__new__(Pi05Teleop);sim.mode='walk';marker=object();sim.home_motion=marker
        sim.set_mode('walk');self.assertIs(sim.home_motion,marker)

    def test_low_lock_criteria_timeout_invalid(self):
        for qualified,valid,expected_time,reason in [(True,True,.5,'criteria'),(False,True,1.,'timeout'),(False,False,1.,None)]:
            sim=object.__new__(OriginalTeleop);sim.base=0;sim.target_height=.236
            sim.lock_elapsed=sim.lock_stable_time=0.;sim.lock_phase='lowering';sim.lock_reason=None
            sim.data=NS(xpos=np.array([[0,0,.236 if qualified else .4]]),xmat=np.array([np.eye(3).ravel()]))
            sim.valid_lock_state=lambda:valid;sim.ground_height=lambda:0.
            sim.lock=lambda:setattr(sim,'lock_phase','locked')
            for _ in range(50):
                if sim.lock_phase!='lowering':break
                sim.update_low_lock(.02)
            self.assertAlmostEqual(sim.lock_elapsed,expected_time)
            self.assertEqual(sim.lock_reason,reason)
            self.assertEqual(sim.lock_phase,'locked' if valid else 'failed')

    def test_ground_ray_excludes_all_robot_bodies(self):
        import mujoco
        sim=object.__new__(OriginalTeleop)
        sim.model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="2 2 .1" group="3"/>
          <body name="World_b2w_piper_base_link" pos="0 0 .6"><freejoint/>
            <geom name="base_collision" type="box" size=".2 .2 .1" group="3"/>
            <body name="World_b2w_piper_leg" pos="0 0 -.2">
              <geom name="leg_collision" type="box" size=".1 .1 .05" group="3"/>
            </body></body></worldbody></mujoco>''')
        sim.data=mujoco.MjData(sim.model);sim.base=sim.model.body('World_b2w_piper_base_link').id
        sim.robot_geom_ids=np.array([sim.model.geom(n).id for n in ('base_collision','leg_collision')])
        original=sim.model.geom_group.copy()
        for z in (.6,.3):
            sim.data.qpos[2]=z;mujoco.mj_forward(sim.model,sim.data)
            self.assertAlmostEqual(sim.ground_height(),0.)
            self.assertEqual(sim.ground_hit['geom'],'floor')
            np.testing.assert_equal(sim.model.geom_group,original)

    def test_material_variant_uses_weathered_values(self):
        d=json.loads((HERE.parent/'material_defaults.json').read_text())['materials']
        p=variant_values(d['Plaster_Wall.mdl'],'Plaster_Wall_Weathered')
        np.testing.assert_allclose(p['diffuse_tint'],[.973,.815,.697]);self.assertEqual(p['diffuse_brightness'],.23)
        m=variant_values(d['Mortar.mdl'],'Mortar_Antique_White')
        np.testing.assert_allclose(m['mortar_tint'],[1,.940601,.766744])

    def test_material_authored_color_aliases(self):
        from pxr import Sdf
        stage=Usd.Stage.CreateInMemory();shader=UsdShade.Shader.Define(stage,'/Shader')
        shader.CreateInput('diffuse_tint',Sdf.ValueTypeNames.Color3f).Set((.02,.03,.04))
        color,texture,_=resolve(shader,{},None,Path,HERE)
        np.testing.assert_allclose(color,[.02,.03,.04]);self.assertIsNone(texture)

    def test_omnipbr_map_does_not_multiply_fallback_constant(self):
        from pxr import Sdf
        stage=Usd.Stage.CreateInMemory();shader=UsdShade.Shader.Define(stage,'/Shader')
        shader.GetPrim().CreateAttribute('info:mdl:sourceAsset',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
        shader.CreateInput('diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set((.2,.2,.2))
        texture=HERE.parent/'assets/tex_97e2f393cada6570.png'
        shader.CreateInput('diffuse_texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
        color,image,_=resolve(shader,{},None,Path,HERE)
        np.testing.assert_allclose(color,[1,1,1]);self.assertEqual(image,texture)
        shader.CreateInput('diffuse_tint',Sdf.ValueTypeNames.Color3f).Set((.3,.4,.5))
        color,_,_=resolve(shader,{},None,Path,HERE)
        np.testing.assert_allclose(color,[.3,.4,.5])

    def test_omnipbr_without_map_keeps_constant_color(self):
        from pxr import Sdf
        stage=Usd.Stage.CreateInMemory();shader=UsdShade.Shader.Define(stage,'/Shader')
        shader.GetPrim().CreateAttribute('info:mdl:sourceAsset',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
        shader.CreateInput('diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set((.2,.3,.4))
        color,image,_=resolve(shader,{},None,Path,HERE)
        np.testing.assert_allclose(color,[.2,.3,.4]);self.assertIsNone(image)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contract)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (HERE/'contract_results.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,
        'failures':[(str(t),s) for t,s in result.failures], 'errors':[(str(t),s) for t,s in result.errors]},indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
