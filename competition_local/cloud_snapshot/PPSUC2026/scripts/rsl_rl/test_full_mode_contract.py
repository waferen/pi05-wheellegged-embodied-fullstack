"""Exercise actual teleop control code on CPU, without Isaac Sim or a server.

The entry point launches Isaac Sim on import, so extract its shared helpers and
control-loop prefix. Robot state and the lock are test doubles; physics is not tested.
"""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

SOURCE = Path(__file__).with_name('teleop_dual_cerebellum_arm.py')
TREE = ast.parse(SOURCE.read_text())
MAIN = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
HELPERS = {'_print_mode', '_clamp_arm_target', '_cancel_arm_home_motion', '_set_arm_home',
           '_update_arm_home_motion', '_arm_enabled', '_set_mode', '_grasp', '_set_control',
           '_publish_full_status', '_toggle_task_phase'}
CONSTANTS = {n.targets[0].id: ast.literal_eval(n.value) for n in TREE.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in {'ARM_HOME', 'ARM_HARD_LIMITS'}}
LOOP = next(n for n in ast.walk(MAIN) if isinstance(n, ast.While)
            and ast.unparse(n.test) == "simulation_app.is_running() and (not should_quit['value'])")
START = next(i for i, n in enumerate(LOOP.body) if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'full_arm_action')
END = next(i for i, n in enumerate(LOOP.body) if isinstance(n, ast.Expr)
           and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
           and n.value.func.id == '_clamp_arm_target')

def code(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec')

HELPER_CODE = code([n for n in MAIN.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS])
CONTROL_CODE = code(LOOP.body[START:END+1])

class Lock:
    def __init__(self):
        self.phase = 'off'
        self.requests = self.immediate = 0
    @property
    def locked(self):
        return self.phase == 'locked'
    def release(self):
        self.phase = 'off'
    def request(self, height, height_error):
        self.phase = 'lowering'
        self.requests += 1
    def lock_current(self):
        self.phase = 'locked'
        self.immediate += 1

class Controller:
    def __init__(self):
        initial = torch.tensor([[.1, .2, -.3, .4, .1, -.2, .023, -.023]])
        teleop = NS(ik_cancel=False, ik_keys=set(), ik_pending=[], joint_keys=set())
        def reset():
            teleop.ik_cancel = True
            teleop.ik_keys.clear()
            teleop.ik_pending.clear()
            teleop.joint_keys.clear()
        teleop.reset = reset
        self.ns = dict(CONSTANTS, torch=torch, Path=Path, json=json, device='cpu', print=lambda *a, **k: None,
            args_cli=NS(gripper_closed=.001, gripper_open=.045, arm_home_duration=1.,
                        min_height=.236, stand_height=.59, record_warmup=0, policy_full=True,
                        execute_steps=25),
            robot=NS(data=NS(joint_pos=initial.clone())), arm_joint_ids=list(range(8)),
            arm_target=initial.clone(), gripper_value={'value': .023}, mode={'value': 'walk'},
            arm_home_motion={'active': False}, low_lock=Lock(), teleop=teleop,
            control={'model': True},
            full_status={'path': None, 'last': None, 'status': 'starting', 'message': ''},
            target_height={'value': .59}, base_command={'value': (0., 0., 0.)},
            step=0, dt=.02, raw_env=None, collector=None,
            ik=NS(settling=False, cancel=lambda: None),
            _set_base_command=lambda env, cmd: None, _set_height=lambda env, height: None)
        self.control_changes = []
        self.ns['policy_client'] = NS(step=lambda env, mode: self.response,
            set_model_control=self.control_changes.append)
        exec(HELPER_CODE, self.ns)
    def tick(self, mode, gripper=1.):
        self.response = ((.2, 0., .3), mode, [.6, .7, -.8, .9, .5, -.6], gripper)
        exec(CONTROL_CODE, self.ns)
        self.ns['step'] += 1

class ModeContract(unittest.TestCase):
    def test_manual_phase_callback_requires_manual_and_connected_client(self):
        c = Controller()
        s = c.ns
        changes = []
        s['policy_client'].toggle_task_phase = lambda: changes.append(True) or True
        s['_toggle_task_phase']()
        self.assertEqual(changes, [])
        s['_set_control'](False)
        s['_toggle_task_phase']()
        self.assertEqual(changes, [True])
        s['policy_client'] = None
        s['_toggle_task_phase']()  # F2 during startup must be harmless.

    def test_status_reports_current_control_phase_and_arm_state(self):
        c = Controller()
        s = c.ns
        s['policy_client'].task_phase = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            s['full_status']['path'] = str(path)
            s['_publish_full_status'](status='running')
            data = json.loads(path.read_text())
            self.assertEqual((data['control'], data['mode'], data['task_phase']),
                             ('model', 'walk', 0))
            self.assertEqual(data['execute_steps'], 25)
            before = path.stat().st_mtime_ns
            s['_publish_full_status']()
            self.assertEqual(path.stat().st_mtime_ns, before)
            s['mode']['value'] = 'high_grasp'
            s['low_lock'].phase = 'locked'
            s['policy_client'].task_phase = 1
            s['_set_control'](False)
            data = json.loads(path.read_text())
            self.assertEqual((data['control'], data['mode'], data['task_phase']),
                             ('manual', 'high_grasp', 1))
            self.assertTrue(data['arm_ready'])
            s['arm_home_motion']['active'] = True
            s['_publish_full_status']()
            self.assertTrue(json.loads(path.read_text())['homing'])
            self.assertFalse(path.with_suffix('.tmp').exists())
            s['full_status']['path'] = str(Path(directory) / 'missing' / 'status.json')
            s['_publish_full_status'](status='error', message='test error')

    def test_grasp_modes_only_exit_through_walk(self):
        for grasp, other in [('low_grasp', 'high_grasp'), ('high_grasp', 'low_grasp')]:
            for lock_phase in ('lowering', 'locked'):
                with self.subTest(grasp=grasp, lock_phase=lock_phase):
                    c = Controller()
                    s = c.ns
                    s['mode']['value'] = grasp
                    s['low_lock'].phase = lock_phase
                    s['target_height']['value'] = .3
                    before = s['arm_target'].clone()
                    s['_set_mode']('stair')()
                    s['_grasp'](other, .59)
                    self.assertEqual(s['mode']['value'], grasp)
                    self.assertEqual(s['low_lock'].phase, lock_phase)
                    self.assertEqual(s['target_height']['value'], .3)
                    self.assertEqual(s['gripper_value']['value'], .023)
                    torch.testing.assert_close(s['arm_target'], before)

    def test_takeover_preserves_mode_targets_and_ongoing_home(self):
        c = Controller()
        s = c.ns
        c.tick('high_grasp', gripper=.5)
        c.tick('walk')
        before = s['arm_target'].clone()
        home_start = s['arm_home_motion']['start_step']
        for active in (False, False, True, True):
            s['_set_control'](active)
            self.assertEqual(s['control']['model'], active)
            self.assertEqual(s['mode']['value'], 'walk')
            self.assertTrue(s['arm_home_motion']['active'])
            self.assertEqual(s['arm_home_motion']['start_step'], home_start)
            self.assertEqual(s['gripper_value']['value'], .023)
            torch.testing.assert_close(s['arm_target'], before)
        self.assertEqual(c.control_changes, [False, True])

    def test_manual_control_skips_model_and_enables_ik(self):
        c = Controller()
        s = c.ns
        c.tick('high_grasp', gripper=.5)
        s['_set_control'](False)
        def unexpected(*args):
            raise AssertionError('Model inference called during manual control')
        s['policy_client'].step = unexpected
        calls = []
        def ik_step(keys, dt, current, pending):
            calls.append((set(keys), list(pending)))
            return current + .01
        s['ik'].step = ik_step
        s['ik_pause'] = {'until': 0}
        s['teleop'].ik_keys.add('UP')
        s['teleop'].ik_pending.append('UP')
        s['command'] = (.4, 0., -.2)
        before = s['arm_target'][:, :6].clone()
        exec(CONTROL_CODE, s)
        self.assertEqual(calls, [({'UP'}, ['UP'])])
        self.assertEqual(s['mode']['value'], 'high_grasp')
        self.assertEqual(s['command'], (0., 0., 0.))
        torch.testing.assert_close(s['arm_target'][:, :6], before + .01)
        s['_set_mode']('walk')()
        s['command'] = (.4, 0., -.2)
        exec(CONTROL_CODE, s)
        self.assertEqual(s['command'], (.4, 0., -.2))

    def test_rejected_policy_row_holds_targets(self):
        c = Controller()
        s = c.ns
        c.tick('high_grasp', gripper=.5)
        before = s['arm_target'].clone()
        s['policy_client'].step = lambda env, mode: None
        exec(CONTROL_CODE, s)
        self.assertEqual(s['mode']['value'], 'high_grasp')
        self.assertEqual(s['command'], (0., 0., 0.))
        self.assertTrue(s['low_lock'].locked)
        torch.testing.assert_close(s['arm_target'], before)

    def test_arm_and_gripper_gate(self):
        for mode in ('walk', 'stair', 'low_grasp', 'high_grasp'):
            for locked in (False, True):
                with self.subTest(mode=mode, locked=locked):
                    c = Controller()
                    s = c.ns
                    s['mode']['value'] = mode
                    s['low_lock'].phase = 'locked' if locked else 'lowering'
                    before = s['arm_target'].clone()
                    c.tick(mode)
                    allowed = mode in ('low_grasp', 'high_grasp') and locked
                    if allowed:
                        torch.testing.assert_close(s['arm_target'][0, :6], torch.tensor(c.response[2]))
                        self.assertAlmostEqual(s['gripper_value']['value'], .045)
                    else:
                        torch.testing.assert_close(s['arm_target'], before)
                        self.assertEqual(s['gripper_value']['value'], .023)
                    if mode in ('low_grasp', 'high_grasp'):
                        self.assertEqual(s['command'], (0., 0., 0.))

    def test_grasp_entry_uses_existing_lock(self):
        c = Controller()
        before = c.ns['arm_target'].clone()
        c.tick('low_grasp')
        self.assertEqual(c.ns['low_lock'].phase, 'lowering')
        torch.testing.assert_close(c.ns['arm_target'], before)
        c.tick('low_grasp')
        self.assertEqual(c.ns['low_lock'].requests, 1)
        c.ns['low_lock'].phase = 'locked'
        c.tick('low_grasp')
        torch.testing.assert_close(c.ns['arm_target'][0, :6], torch.tensor(c.response[2]))
        c = Controller()
        c.ns['target_height']['value'] = .4
        c.tick('high_grasp')
        self.assertTrue(c.ns['low_lock'].locked)
        self.assertEqual(c.ns['target_height']['value'], .4)
        c.tick('high_grasp')
        self.assertEqual(c.ns['low_lock'].immediate, 1)

    def test_141_and_151_home_while_walking_and_hold_gripper(self):
        for grasp in ('low_grasp', 'high_grasp'):
            with self.subTest(grasp=grasp):
                c = Controller()
                s = c.ns
                c.tick(grasp)
                s['low_lock'].phase = 'locked'
                c.tick(grasp, gripper=.5)
                held_gripper = s['gripper_value']['value']
                measured = s['robot'].data.joint_pos[:, :6].clone()
                c.tick('walk', gripper=0.)
                self.assertEqual(s['mode']['value'], 'walk')
                self.assertEqual(s['low_lock'].phase, 'off')
                self.assertTrue(s['arm_home_motion']['active'])
                torch.testing.assert_close(s['arm_target'][:, :6], measured)
                start = s['arm_home_motion']['start_step']
                for i in range(1, 51):
                    c.tick('walk', gripper=float(i % 2))
                    alpha = i / 50
                    alpha = alpha * alpha * (3 - 2 * alpha)
                    home = torch.tensor([CONSTANTS['ARM_HOME'][:6]])
                    torch.testing.assert_close(s['arm_target'][:, :6], measured*(1-alpha)+home*alpha)
                    self.assertEqual(s['arm_home_motion']['start_step'], start)
                    self.assertEqual(s['gripper_value']['value'], held_gripper)
                    self.assertEqual(s['command'], (.2, 0., .3))
                    self.assertEqual(s['arm_home_motion']['active'], i < 50)
                c.tick('walk')
                self.assertFalse(s['arm_home_motion']['active'])
                self.assertEqual(s['gripper_value']['value'], held_gripper)
                torch.testing.assert_close(s['arm_target'][:, :6], home)

    def test_stair_predictions_do_not_cancel_home(self):
        c = Controller()
        s = c.ns
        c.tick('high_grasp', gripper=.5)
        c.tick('walk')
        c.tick('stair')
        self.assertTrue(s['arm_home_motion']['active'])
        self.assertAlmostEqual(s['gripper_value']['value'], .023)
        self.assertGreater(s['arm_target'][0, 0].item(), 0.)
        self.assertLess(s['arm_target'][0, 0].item(), s['robot'].data.joint_pos[0, 0].item())

class KeyboardBase:
    def reset(self):
        self._base_command[:] = 0.


EVENTS = NS(KEY_PRESS=1, KEY_RELEASE=2, KEY_REPEAT=3)
KEYBOARD_NS = dict(Se2Keyboard=KeyboardBase, np=np,
    carb=NS(input=NS(KeyboardEventType=EVENTS)),
    IK_KEYS={'UP': (0, 1)}, ALIASES={})
exec(code([n for n in TREE.body if isinstance(n, ast.ClassDef)
           and n.name == 'StateBasedSe2Keyboard']), KEYBOARD_NS)


class KeyboardContract(unittest.TestCase):
    def setUp(self):
        cls = KEYBOARD_NS['StateBasedSe2Keyboard']
        self.keyboard = k = cls.__new__(cls)
        k._pressed_keys, k._record_keys = set(), set()
        k.ik_keys, k.ik_pending, k.joint_keys = set(), [], set()
        k.ik_cancel = False
        k._base_command = np.zeros(3)
        k.arm_is_ready = lambda: True
        self.manual = False
        k.manual_control_enabled = lambda: self.manual
        self.callbacks = []
        k._additional_callbacks = {key: (lambda key=key: self.callbacks.append(key))
            for key in ('1', '2', '4', '5', 'R', 'C', 'P', 'H', 'Q', 'ESCAPE', 'F1', 'F2', 'F3')}
        def owner(manual):
            self.manual = manual
            k.reset()
        k._additional_callbacks['F4'] = lambda: owner(True)
        k._additional_callbacks['F5'] = lambda: owner(False)

    def event(self, key, event_type=EVENTS.KEY_PRESS):
        return self.keyboard._on_keyboard_event(NS(input=key, type=event_type))

    def test_manual_f2_fires_once_per_press(self):
        self.event('F2')
        self.assertEqual(self.callbacks, [])
        self.event('F4')
        self.event('KEY_F2')
        self.event('F2', EVENTS.KEY_REPEAT)
        self.event('F2')
        self.assertEqual(self.callbacks, ['F2'])
        self.event('F2', EVENTS.KEY_RELEASE)
        self.event('F2')
        self.assertEqual(self.callbacks, ['F2', 'F2'])
        self.event('F5')
        self.event('F2', EVENTS.KEY_RELEASE)
        self.event('F2')
        self.assertEqual(self.callbacks, ['F2', 'F2'])

    def test_model_blocks_ordinary_controls_and_allows_exit(self):
        for key in ('W', 'A', 'S', 'D', '1', '2', '4', '5', 'R', 'C', 'P', 'H', 'UP',
                    'SPACE', 'F1', 'F2', 'F3'):
            self.event(key)
        k = self.keyboard
        np.testing.assert_array_equal(k._base_command, 0.)
        self.assertFalse(k._pressed_keys or k.ik_keys or k.ik_pending or k.joint_keys)
        self.assertFalse(k.ik_cancel)
        self.assertEqual(self.callbacks, [])
        for key in ('Q', 'ESCAPE'):
            self.event(key)
        self.assertEqual(self.callbacks, ['Q', 'ESCAPE'])

    def test_manual_keys_and_no_stale_keys_after_resume(self):
        self.event('KEY_F4')
        self.assertTrue(self.manual)
        self.event('W')
        self.event('UP')
        self.event('R')
        self.event('C')
        np.testing.assert_array_equal(self.keyboard._base_command, [1., 0., 0.])
        self.assertEqual(self.keyboard.ik_keys, {'UP'})
        self.assertEqual(self.keyboard.joint_keys, {'R'})
        self.assertEqual(self.callbacks, ['R', 'C'])
        self.event('F5')
        self.assertFalse(self.manual)
        for key in ('W', 'UP', 'R'):
            self.event(key, EVENTS.KEY_RELEASE)
        self.event('F4')
        k = self.keyboard
        np.testing.assert_array_equal(k._base_command, 0.)
        self.assertFalse(k._pressed_keys or k.ik_keys or k.ik_pending or k.joint_keys)

    def test_manual_mode_callbacks_and_space(self):
        self.event('F4')
        for key in ('1', '2', '4', '5', 'P', 'H'):
            self.event(key)
        self.assertEqual(self.callbacks, ['1', '2', '4', '5', 'P', 'H'])
        self.event('W')
        self.event('SPACE')
        np.testing.assert_array_equal(self.keyboard._base_command, 0.)


if __name__ == '__main__':
    unittest.main()
