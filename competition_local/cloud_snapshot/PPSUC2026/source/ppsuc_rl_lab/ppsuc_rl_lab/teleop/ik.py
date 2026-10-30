"""Cartesian end-effector servo with bounded differential inverse kinematics.

Lula provides the model joint limits. PhysX supplies the Jacobian of the actual
finger-midpoint TCP. Joint limits are enforced before submitting each command.
"""
from pathlib import Path
import math
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import (
    apply_delta_pose, compute_pose_error, quat_apply_inverse, quat_conjugate,
    quat_mul, matrix_from_quat,
)
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot_motion.motion_generation")
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

IK_KEYS = {
    "UP": (0, 1), "DOWN": (0, -1),
    "LEFT": (1, 1), "RIGHT": (1, -1),
    "PAGE_UP": (2, 1), "PAGE_DOWN": (2, -1),
    "7": (3, -1), "8": (3, 1),
    "9": (4, -1), "0": (4, 1),
    "-": (5, -1), "=": (5, 1),
}
ALIASES = dict(zip(("SEVEN", "EIGHT", "NINE", "ZERO"), "7890"))
ALIASES.update({"MINUS": "-", "EQUAL": "=", "EQUALS": "="})
ALIASES.update({"PAGEUP": "PAGE_UP", "PAGEDOWN": "PAGE_DOWN"})
ROTATION_KEYS = frozenset(("7", "8", "9", "0", "-", "="))


def bounded_velocity(jacobian, twist, q, previous, limits, vmax, acceleration, dt, damping=0.):
    """Scale a truncated Jacobian solve while preserving the requested TCP ray."""
    q = np.clip(q, limits[:, 0], limits[:, 1])
    lower = np.maximum(-vmax, (limits[:, 0] - q) / dt)
    upper = np.minimum(vmax, (limits[:, 1] - q) / dt)
    # Distance-based braking envelope, in addition to per-step position bounds.
    lower = np.maximum(lower, -(np.sqrt((acceleration*dt)**2 + 2 * acceleration * np.maximum(q-limits[:, 0], 0))-acceleration*dt))
    upper = np.minimum(upper, (np.sqrt((acceleration*dt)**2 + 2 * acceleration * np.maximum(limits[:, 1]-q, 0))-acceleration*dt))
    lo = np.maximum(lower, previous - acceleration * dt)
    hi = np.minimum(upper, previous + acceleration * dt)
    # At an already-violated/measured boundary, position limits take priority
    # over continuing an outward velocity from the preceding controller.
    invalid = lo > hi
    lo[invalid], hi[invalid] = lower[invalid], upper[invalid]
    weights = np.array([1., 1., 1., .25, .25, .25])
    weighted = weights[:, None] * jacobian
    # Scale the entire task-space request before applying joint bounds.
    # Independent clipping would change translation/rotation direction.
    if damping > 0.:
        raw = np.linalg.solve(weighted.T @ weighted + np.eye(6)*damping**2,
                              weighted.T @ (weights*twist))
    else:
        raw = np.linalg.lstsq(weighted, weights * twist, rcond=1e-4)[0]
    scale = 1.
    for value, low, high in zip(raw, lower, upper):
        if value > 1e-12:
            scale = min(scale, max(0., high/value))
        elif value < -1e-12:
            scale = min(scale, max(0., low/value))
    # A single scalar also enforces acceleration without bending the TCP path.
    alpha_lo, alpha_hi = 0., scale
    feasible = True
    for value, low, high in zip(raw, lo, hi):
        if abs(value) < 1e-12:
            feasible = feasible and low <= 0. <= high
        else:
            a0, a1 = sorted((low/value, high/value))
            alpha_lo = max(alpha_lo, a0)
            alpha_hi = min(alpha_hi, a1)
    if feasible and alpha_lo <= alpha_hi:
        return raw * max(0., alpha_hi)
    # A changing Jacobian can make the requested ray acceleration-infeasible.
    # Decelerate the preceding command as a whole before resuming that ray.
    factor = min(1., float(np.min(acceleration*dt/np.maximum(np.abs(previous), 1e-12))))
    return np.clip(previous*(1.-factor), lo, hi)



class ArmIK:
    FRAME = "arm_tcp"

    def __init__(self, robot, joints, limits):
        self.robot, self.joints = robot, joints[:6]
        self.base = robot.find_bodies("arm_base")[0][0]
        self.fingers = [robot.find_bodies(n)[0][0] for n in ("arm_link7", "arm_link8")]
        self.limits = np.asarray(limits, dtype=np.float64).copy()
        physical = robot.data.joint_pos_limits[0, self.joints].cpu().numpy()
        self.limits[:, 0] = np.maximum(self.limits[:, 0], physical[:, 0])
        self.limits[:, 1] = np.minimum(self.limits[:, 1], physical[:, 1])
        config = Path(__file__).with_name("lula")
        self._model_dir = tempfile.TemporaryDirectory(prefix="piper_lula_")
        urdf = ET.parse(config / "piper.urdf")
        self._origins = []
        for joint in urdf.getroot().findall("joint"):
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            self._origins.append(transform)
        for name, bounds in zip([robot.joint_names[i] for i in self.joints], self.limits):
            limit = urdf.find(f"joint[@name='{name}']/limit")
            limit.set("lower", str(bounds[0]))
            limit.set("upper", str(bounds[1]))
        active_urdf = Path(self._model_dir.name) / "piper.urdf"
        urdf.write(active_urdf, encoding="utf-8", xml_declaration=True)
        self.solver = LulaKinematicsSolver(str(config / "piper_descriptor.yaml"), str(active_urdf))
        if self.solver.get_joint_names() != [robot.joint_names[i] for i in self.joints]:
            raise ValueError("Lula joint order mismatch")
        self.max_velocity = np.asarray(self.solver.get_cspace_velocity_limits()).copy()
        physical_velocity = robot.data.joint_vel_limits[0, self.joints].cpu().numpy()
        self.max_velocity = np.minimum(self.max_velocity, physical_velocity)
        self.max_acceleration = np.full(6, 8.)
        self.translation_step = .01
        self.rotation_step = math.radians(5.)
        self.translation_speed = .15
        self.rotation_speed = math.radians(30.)
        self.max_position_lead = .015
        self.max_rotation_lead = math.radians(15.)
        self.hold_delay = .18
        self.held_keys = frozenset()
        self.held_time = 0.
        self.target = None
        self.active = self.settling = self.blocked = False
        self.status = "idle"
        self.path_kind = "cartesian"
        self.velocity = torch.zeros(6, device=robot.device)
        self.last_output = None
        self.tracking_time = 0.
        self.stall_time = 0.
        print("[CARTESIAN] position=0.15 m/s, rotation=30 deg/s; taps=1 cm/5 deg; "
              "rotation tap-only; joint velocity limits=", self.max_velocity.tolist(),
              "acceleration<=8 rad/s^2", flush=True)

    def position(self):
        d = self.robot.data
        p = d.body_pos_w[0, self.fingers].mean(0) - d.body_pos_w[0, self.base]
        return quat_apply_inverse(d.body_quat_w[0, self.base], p)

    def pose(self):
        d = self.robot.data
        orientation = quat_mul(quat_conjugate(d.body_quat_w[0, self.base]),
                               d.body_quat_w[0, self.fingers[0]])
        return self.position().unsqueeze(0).cpu(), orientation.unsqueeze(0).cpu()

    def jacobian(self):
        fixed = self.robot.is_fixed_base
        bodies = [i-1 if fixed else i for i in self.fingers]
        cols = [i if fixed else i+6 for i in self.joints]
        full = self.robot.root_physx_view.get_jacobians()[0]
        first = full[bodies[0]][:, cols]
        second = full[bodies[1]][:, cols]
        # PhysX reports the COM Jacobian; our TCP uses the finger actor origins.
        offsets = (self.robot.data.body_pos_w-self.robot.data.body_com_pos_w)[0]
        linear = []
        for body, jac in zip(self.fingers, (first, second)):
            linear.append(jac[:3] + torch.linalg.cross(
                jac[3:].T, offsets[body].expand(6, 3), dim=1).T)
        world = torch.cat(((linear[0]+linear[1])*.5, first[3:]), dim=0)
        rotation = matrix_from_quat(self.robot.data.body_quat_w[0, self.base]).T
        local = torch.cat((rotation @ world[:3], rotation @ world[3:]), dim=0)
        return local.cpu().numpy().astype(np.float64)

    def model_jacobian(self, q):
        """Model Jacobian for a short lookahead of joint-space curvature."""
        transform = np.eye(4)
        positions, axes = [], []
        for i, origin in enumerate(self._origins):
            transform = transform @ origin
            if i < 6:
                positions.append(transform[:3, 3].copy())
                axes.append(transform[:3, 2].copy())
                rotation = np.eye(4)
                c, sn = math.cos(q[i]), math.sin(q[i])
                rotation[:2, :2] = ((c, -sn), (sn, c))
                transform = transform @ rotation
        return np.array([np.r_[np.cross(a, transform[:3, 3]-p), a]
                         for p, a in zip(positions, axes)]).T

    def curvature_scale(self, q, j, error, twist):
        # Reserve most acceleration for starting/stopping and feedback. A
        # first-order lookahead prevents carrying high joint momentum into a
        # sharply turning IK path near a singular configuration.
        raw = np.linalg.lstsq(j, twist, rcond=1e-4)[0]
        eps = 1e-4/max(1., np.linalg.norm(raw))
        model = self.model_jacobian(q)
        derivative = (self.model_jacobian(q+eps*raw)-model)/eps
        curvature = np.linalg.lstsq(model, derivative @ raw, rcond=1e-4)[0]
        conditioning = min(1., float(np.linalg.svd(model, compute_uv=False)[-1])/.05)
        curvature_budget = self.max_acceleration*(.125+.375*conditioning**2)
        scale = min(1., float(np.sqrt(
            curvature_budget/np.maximum(np.abs(curvature), 1e-12)).min()))
        joint_error = np.linalg.lstsq(j, error, rcond=1e-4)[0]
        braking = np.sqrt(self.max_acceleration*.5*np.abs(joint_error))
        ratios = np.divide(braking, np.abs(raw), out=np.full(6, np.inf), where=np.abs(raw) > 1e-6)
        return min(scale, float(ratios.min()))

    def cancel(self):
        if self.settling:
            self.status = "cancelled"
        self.active = self.settling = False
        self.held_keys = frozenset()
        self.held_time = self.tracking_time = self.stall_time = 0.
        self.velocity.zero_()

    def stop(self):
        self.cancel()
        self.last_output = self.robot.data.joint_pos[0, self.joints].clone()
        return self.last_output

    def plan(self, position, orientation):
        p = torch.as_tensor(position).reshape(1, 3).cpu().clone()
        q = torch.as_tensor(orientation).reshape(1, 4).cpu().clone()
        if not bool(torch.isfinite(p).all() and torch.isfinite(q).all()) or q.norm() < 1e-9:
            self.status, self.blocked = "invalid_target", True
            return False
        self.target = (p, q/q.norm())
        self.active = self.settling = True
        self.blocked = False
        self.tracking_time = 0.
        self.status = "executing"
        return True

    def _delta(self, keys, translation, rotation):
        delta = torch.zeros(1, 6)
        for key in keys:
            axis, sign = IK_KEYS[key]
            delta[0, axis] += sign
        for begin, magnitude in ((0, translation), (3, rotation)):
            part = delta[:, begin:begin+3]
            part.mul_(magnitude / max(float(part.norm()), 1.))
        return delta

    @staticmethod
    def _cap(vector, limit):
        return vector * min(1., limit/max(np.linalg.norm(vector), 1e-12))

    def _advance(self, dt, held):
        if not self.active or self.target is None:
            self.velocity.zero_()
            return held
        ep, er = compute_pose_error(*self.pose(), *self.target, rot_error_type="axis_angle")
        error = np.r_[ep.numpy().reshape(3), er.numpy().reshape(3)].astype(np.float64)
        q = self.robot.data.joint_pos[0, self.joints].cpu().numpy().astype(np.float64)
        measured_v = self.robot.data.joint_vel[0, self.joints].cpu().numpy()
        # A sub-millimetre deadband prevents chasing amplified numerical noise
        # near singularities. Only stop when both measured and commanded joint
        # speeds are low enough to remove the remaining command in one step.
        command_stop = np.minimum(self.max_acceleration*dt, .1)
        if (np.linalg.norm(error[:3]) < .0005
                and np.linalg.norm(error[3:]) < math.radians(.2)
                and np.max(np.abs(measured_v)) < .1
                and np.all(np.abs(self.velocity.cpu().numpy()) <= command_stop)):
            self.active = self.settling = False
            self.status = "reached"
            self.velocity.zero_()
            return held
        twist = np.r_[self._cap(20.*error[:3], self.translation_speed),
                      self._cap(15.*error[3:], self.rotation_speed)]
        try:
            j = self.jacobian()
            twist *= self.curvature_scale(q, j, error, twist)
            # Dampen poorly observable directions near the goal so tiny
            # pose errors do not excite large opposing wrist velocities.
            proximity = max(np.linalg.norm(error[:3])/.005,
                            np.linalg.norm(error[3:])/math.radians(2.))
            damping = .005*max(0., 1.-proximity)
            v = bounded_velocity(j, twist, q, self.velocity.cpu().numpy(),
                                 self.limits, self.max_velocity, self.max_acceleration, dt, damping)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            output = self.stop()
            self.status, self.blocked = "servo_failed", True
            print("[CARTESIAN]", exc, flush=True)
            return output
        commanded = np.clip(np.clip(q, self.limits[:, 0], self.limits[:, 1])+v*dt,
                            self.limits[:, 0], self.limits[:, 1])
        self.velocity = torch.as_tensor(v, device=held.device, dtype=held.dtype)
        self.last_output = torch.as_tensor(commanded, device=held.device, dtype=held.dtype)
        # Measure motion toward the target, rather than error change across
        # moving targets; steady long holds must not be mistaken for a stall.
        actual_twist = j @ measured_v
        progress = (np.dot(error[:3], actual_twist[:3])/max(np.linalg.norm(error[:3]), 1e-9)
                    + .15*np.dot(error[3:], actual_twist[3:])/max(np.linalg.norm(error[3:]), 1e-9))
        self.stall_time = 0. if progress > 1e-4 else self.stall_time+dt
        self.tracking_time += dt
        if self.stall_time > 1.:
            output = self.stop()
            self.status, self.blocked = "unreachable_or_limited", True
            return output
        return self.last_output

    def step(self, keys, dt, held, taps=()):
        if dt <= 0:
            return held
        # Rotation is discrete. OS key repeats must not cause continuous rotation.
        keys = frozenset(k for k in keys if k not in ROTATION_KEYS)
        if keys != self.held_keys:
            self.held_time = 0.
        self.held_keys = keys
        if taps:
            if any(k not in ROTATION_KEYS for k in taps):
                self.held_time = 0.
            delta = self._delta(taps, self.translation_step, self.rotation_step)
            origin = self.target if self.active and not self.blocked else self.pose()
            candidate = apply_delta_pose(*origin, delta)
            ep, er = compute_pose_error(*self.pose(), *candidate, rot_error_type="axis_angle")
            if ep.norm() <= self.max_position_lead+1e-6 and er.norm() <= self.max_rotation_lead+1e-6:
                self.plan(*candidate)
        elif keys and not self.blocked:
            self.held_time += dt
            if self.held_time >= self.hold_delay:
                delta = self._delta(keys, self.translation_speed*dt, 0.)
                origin = self.target if self.active else self.pose()
                candidate = apply_delta_pose(*origin, delta)
                ep, er = compute_pose_error(*self.pose(), *candidate, rot_error_type="axis_angle")
                if ep.norm() <= self.max_position_lead and er.norm() <= self.max_rotation_lead:
                    self.plan(*candidate)
        return self._advance(dt, held)
