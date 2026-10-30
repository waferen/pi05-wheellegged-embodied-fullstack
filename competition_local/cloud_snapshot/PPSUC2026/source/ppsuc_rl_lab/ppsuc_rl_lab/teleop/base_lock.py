"""Lock leg/wheel articulation coordinates without attaching the body to the world."""
import math
import torch


class LowPoseLock:
    TIMEOUT = 1.0
    HOLD_TIME = .5
    HEIGHT_ERROR = .005
    TILT_DEG = 5.

    def __init__(self, robot, arm_joint_ids, ground_height=None):
        self.robot = robot
        self.joints = [i for i in range(robot.num_joints) if i not in arm_joint_ids]
        self.ground_height = ground_height or self._ground_height
        self.constraint_path = None
        self.phase = 'off'
        self.elapsed = self.stable_time = 0.
        self.height_error = self.HEIGHT_ERROR
        self.lock_reason = None
        self.pose = self.positions = None
        self.target_height = None
        self.reasons = []
        self.metrics = {}
        self.next_report = 5.

    @property
    def locked(self):
        return self.phase == 'locked'

    def _ground_height(self, position):
        # Query actual collision geometry, excluding this articulation. No
        # hardcoded world-Z offset: different locations may have different floors.
        import omni.physx
        root = self.robot.root_physx_view.prim_paths[0]
        hits = []
        def hit(result):
            if not (result.collision == root or result.collision.startswith(root + '/')):
                z = float(result.position[2])
                if z <= position[2] + .001 and result.normal[2] > .5:
                    hits.append(z)
            return True
        omni.physx.get_physx_scene_query_interface().raycast_all(
            (position[0], position[1], position[2] + .01), (0., 0., -1.), 3., hit)
        return max(hits) if hits else None

    def request(self, target_height, height_error=.005):
        self.release()
        self.target_height = float(target_height)
        self.height_error = float(height_error)
        self.phase = 'lowering'
        print(f'[BASE LOCK] positioning: height_cmd={self.target_height:.3f}m, tolerance={self.height_error*1000:g}mm, tilt<={self.TILT_DEG:g}deg; hold={self.HOLD_TIME:g}s, force lock at {self.TIMEOUT:g}s', flush=True)

    @torch.inference_mode()
    def lock_current(self):
        """Immediately lock current leg/wheel coordinates, leaving the arm free."""
        self.release()
        d = self.robot.data
        pose = torch.cat((d.root_pos_w, d.root_quat_w), dim=-1).clone()
        positions = d.joint_pos[:, self.joints].clone()
        if not (bool(torch.isfinite(pose).all())
                and bool(torch.isfinite(d.joint_pos).all())
                and bool((d.root_quat_w.norm(dim=-1) >= 1e-8).all())):
            self.phase = 'failed'
            raise RuntimeError('Cannot immediately lock invalid pose or joint state')
        self.pose = pose
        self.positions = positions
        self._attach_internal_constraints()
        self.lock_reason = 'immediate'
        self.phase = 'locked'
        self.enforce()
        print('[LOCK] reason=immediate; current leg/wheel pose locked; arm free', flush=True)

    def release(self):
        if self.constraint_path is not None:
            import omni.usd
            omni.usd.get_context().get_stage().RemovePrim(self.constraint_path)
            self.constraint_path = None
        if self.phase != 'off':
            print('[BASE LOCK] released', flush=True)
        self.phase = 'off'
        self.elapsed = self.stable_time = 0.
        self.lock_reason = None
        self.pose = self.positions = None
        self.reasons = []
        self.metrics = {}
        self.next_report = 5.

    @torch.inference_mode()
    def update(self, dt):
        if self.phase != 'lowering':
            return
        self.elapsed += dt
        d = self.robot.data
        pose = torch.cat((d.root_pos_w, d.root_quat_w), dim=-1).clone()
        valid = bool(torch.isfinite(pose).all() and torch.isfinite(d.joint_pos).all())
        reasons = []
        self.metrics = {}
        if valid:
            sample = pose[0].cpu().double()
            norm = sample[3:].norm().item()
            valid = norm >= 1e-8
        if not valid:
            reasons.append('invalid pose or joint state')
        else:
            sample[3:] /= norm
            try:
                floor = self.ground_height(sample[:3].tolist())
            except Exception:
                floor = None
            if floor is None or not math.isfinite(floor):
                reasons.append('ground height unavailable')
            else:
                error = sample[2].item()-floor-self.target_height
                self.metrics.update(height_error_mm=error*1000, ground_z=floor)
                if abs(error) > self.height_error:
                    reasons.append(f'height error {error*1000:+.2f}mm > +/-{self.height_error*1000:g}mm')
            q = sample[3:]
            tilt = math.degrees(math.acos(max(-1., min(1., 1-2*(q[1]**2+q[2]**2).item()))))
            self.metrics['tilt_deg'] = tilt
            if tilt > self.TILT_DEG:
                reasons.append(f'tilt {tilt:.3f}deg > {self.TILT_DEG:g}deg')
        self.stable_time = self.stable_time + dt if not reasons else 0.
        self.reasons = reasons
        qualified = self.stable_time >= self.HOLD_TIME-1e-9
        deadline = self.elapsed >= self.TIMEOUT-1e-9
        if valid and (qualified or deadline):
            self.pose = pose
            self.positions = d.joint_pos[:, self.joints].clone()
            self.lock_reason = 'criteria' if qualified else 'timeout'
            self._attach_internal_constraints()
            self.phase = 'locked'
            height = self.target_height + self.metrics.get('height_error_mm', float('nan')) / 1000
            tilt = self.metrics.get('tilt_deg', float('nan'))
            print(f'[LOCK] t={self.elapsed:.2f}s reason={self.lock_reason} height_cmd={self.target_height:.3f} height={height:.3f} tilt={tilt:.2f}', flush=True)
        elif deadline:
            self.phase = 'failed'
            print('[BASE LOCK] cannot lock invalid pose/joints; press 4/5 to retry or 1 to walk', flush=True)

    def enforce(self):
        if not self.locked:
            return
        # Internal fixed constraints hold the links; drives must not fight them.
        self.robot.set_joint_position_target(self.positions, joint_ids=self.joints)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.positions), joint_ids=self.joints)
        self.robot.write_data_to_sim()

    def _attach_internal_constraints(self):
        import omni.usd
        from pxr import Gf, Usd, UsdPhysics
        from isaaclab.utils.math import subtract_frame_transforms
        if self.robot.num_instances != 1:
            raise RuntimeError('Teleop joint lock supports one robot only')
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self.robot.root_physx_view.prim_paths[0])
        names = {self.robot.joint_names[j] for j in self.joints}
        while root:
            joints = {p.GetName(): UsdPhysics.RevoluteJoint(p)
                      for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.RevoluteJoint)}
            if names <= joints.keys():
                break
            root = root.GetParent()
        if not root:
            raise RuntimeError(f'Cannot locate leg/wheel joints: {names - joints.keys()}')
        bodies = {name: i for i, name in enumerate(self.robot.body_names)}
        frames = []
        d = self.robot.data
        for j in self.joints:
            name = self.robot.joint_names[j]
            joint = joints[name]
            parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(parent) != 1 or len(child) != 1:
                raise RuntimeError(f'{name}: both lock endpoints must be robot bodies')
            i0, i1 = bodies[parent[0].name], bodies[child[0].name]
            pos, quat = subtract_frame_transforms(
                d.body_pos_w[:, i1], d.body_quat_w[:, i1],
                d.body_pos_w[:, i0], d.body_quat_w[:, i0])
            frames.append((name, parent, child, pos[0].tolist(), quat[0].tolist()))
        path = str(root.GetPath()) + '/TeleopLegWheelLocks'
        if stage.GetPrimAtPath(path):
            raise RuntimeError(f'Lock path already exists: {path}')
        self.constraint_path = path
        try:
            stage.DefinePrim(path, 'Scope')
            for name, parent, child, pos, quat in frames:
                fixed = UsdPhysics.FixedJoint.Define(stage, path + '/' + name)
                fixed.CreateBody0Rel().SetTargets(parent)
                fixed.CreateBody1Rel().SetTargets(child)
                fixed.CreateLocalPos0Attr(Gf.Vec3f(0.))
                fixed.CreateLocalRot0Attr(Gf.Quatf(1.))
                fixed.CreateLocalPos1Attr(Gf.Vec3f(*pos))
                fixed.CreateLocalRot1Attr(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
                fixed.CreateExcludeFromArticulationAttr(True)
            print(f'[BASE LOCK] {len(frames)} internal fixed constraints; no world endpoint', flush=True)
        except Exception:
            stage.RemovePrim(path)
            self.constraint_path = None
            raise
