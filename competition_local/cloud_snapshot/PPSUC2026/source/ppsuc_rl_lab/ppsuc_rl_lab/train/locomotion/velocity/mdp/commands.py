# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

import math
from dataclasses import MISSING

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class RecoveryThenForwardVelocityCommand(UniformThresholdVelocityCommand):
    """Velocity command that starts each episode with a standing recovery window.

    During the recovery window the command is forced to zero, allowing the policy to stabilize
    posture in place. When recovery ends, a fresh forward command is sampled immediately so the
    robot does not keep standing on the zero command until the next regular resampling interval.
    """

    cfg: RecoveryThenForwardVelocityCommandCfg  # type: ignore

    def __init__(self, cfg: "RecoveryThenForwardVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._recovery_age_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_episode_length = env.episode_length_buf.detach().clone()
        self._was_recovering = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_command(self):
        super()._update_command()

        current_len = self._env.episode_length_buf.detach().clone()
        reset = current_len < self._prev_episode_length
        advanced = current_len > self._prev_episode_length
        self._recovery_age_s[reset] = 0.0
        self._recovery_age_s[advanced] += (
            current_len[advanced] - self._prev_episode_length[advanced]
        ).to(dtype=torch.float32) * self._env.step_dt
        self._prev_episode_length = current_len

        recovering = self._recovery_age_s < self.cfg.recovery_time_s
        just_finished_recovery = self._was_recovering & ~recovering
        if just_finished_recovery.any():
            self._resample_command(torch.where(just_finished_recovery)[0])

        if recovering.any():
            self.vel_command_b[recovering, :] = 0.0
            if self.cfg.heading_command:
                self.heading_target[recovering] = 0.0

        self._was_recovering = recovering





@configclass
class RecoveryThenForwardVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for recovery-then-forward velocity commands."""

    class_type: type = RecoveryThenForwardVelocityCommand
    recovery_time_s: float = 1.0

class BodyHeightCommand(CommandTerm):
    """Command generator that samples a target body height for squat/stand transitions.

    The command is a scalar target height (in m) sampled uniformly from the configured
    range, resampled on a fixed schedule so the robot learns to transition between
    standing and squatting postures.
    """

    cfg: BodyHeightCommandCfg  # type: ignore

    def __init__(self, cfg: "BodyHeightCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.height_command = torch.full(
            (self.num_envs, 1), float(cfg.ranges.height[1]), dtype=torch.float32, device=self.device
        )
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return the target body height command. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        robot = self._env.scene["robot"]
        self.metrics["height_error"] = torch.abs(robot.data.root_pos_w[:, 2] - self.height_command[:, 0])

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new target height for the given environments."""
        if self.cfg.discrete_values is not None:
            values = torch.as_tensor(self.cfg.discrete_values, dtype=torch.float32, device=self.device)
            sample_ids = torch.randint(values.numel(), (len(env_ids),), device=self.device)
            self.height_command[env_ids, 0] = values[sample_ids]
            return
        # NOTE: advanced indexing height_command[env_ids, 0] returns a COPY, so
        # in-place uniform_() would silently not update the buffer. Use assignment.
        lo, hi = self.cfg.ranges.height
        self.height_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)

    def _update_command(self):
        if self.cfg.zero_base_below_height is None:
            return
        base_term = self._env.command_manager._terms.get(self.cfg.base_command_name)
        if base_term is None or not hasattr(base_term, "vel_command_b"):
            return
        low_height = self.height_command[:, 0] <= self.cfg.zero_base_below_height
        if low_height.any():
            base_term.vel_command_b[low_height, :] = 0.0
            if hasattr(base_term, "heading_target"):
                base_term.heading_target[low_height] = 0.0
            if hasattr(base_term, "is_heading_env"):
                base_term.is_heading_env[low_height] = False
            if hasattr(base_term, "is_standing_env"):
                base_term.is_standing_env[low_height] = True


@configclass
class BodyHeightCommandCfg(CommandTermCfg):
    """Configuration for the body-height (squat/stand) command generator."""

    class_type: type = BodyHeightCommand

    @configclass
    class Ranges:
        """Uniform distribution range for the target body height (in m)."""

        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    """Distribution ranges for the target body height."""

    discrete_values: list[float] | None = None
    """Optional discrete body-height targets. When set, these values override uniform range sampling."""

    zero_base_below_height: float | None = None
    """If set, force the base velocity command to zero when the target height is below this value."""

    base_command_name: str = "base_velocity"
    """Name of the velocity command term to zero for low-height commands."""


class ArmSineCommand(CommandTerm):
    """Drive the manipulator arm joints on a sinusoidal trajectory each step.

    The arm is NOT part of the policy action space. Instead this command term
    writes sinusoidal joint targets to the arm every physics step, acting as a
    disturbance the leg policy must reject while keeping the body stable.
    """

    cfg: ArmSineCommandCfg  # type: ignore

    def __init__(self, cfg: "ArmSineCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        robot = env.scene[cfg.asset_name]
        self._arm_joint_ids, self._arm_joint_names = robot.find_joints(cfg.arm_joint_names)
        self._init_pos = robot.data.joint_pos[:, self._arm_joint_ids].detach().clone()
        self._freq = cfg.frequency_hz
        self._amp = cfg.amplitude
        self._phase = cfg.phase

    @property
    def command(self) -> torch.Tensor:
        """The arm drive has no policy-visible command; return zeros."""
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        """Set the arm joints to a sinusoidal position each step."""
        robot = self._env.scene[self.cfg.asset_name]
        t = self._env.episode_length_buf * self._env.step_dt  # (num_envs,)
        t = t.unsqueeze(1)
        # target = init + amp * sin(2*pi*f*t + phase)
        phase = torch.as_tensor(self._phase, dtype=torch.float32, device=self.device).view(1, -1)
        wave = torch.sin(2.0 * torch.pi * self._freq * t + phase)
        target = self._init_pos + self._amp * wave
        robot.set_joint_position_target(target, joint_ids=self._arm_joint_ids)


@configclass
class ArmSineCommandCfg(CommandTermCfg):
    """Configuration for the arm sine-wave command generator."""

    class_type: type = ArmSineCommand

    def __post_init__(self):
        super().__post_init__()
        # The sine drive never resamples; set the range to infinity.
        self.resampling_time_range = (1e9, 1e9)

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    arm_joint_names: list[str] = [
        "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
        "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8",
    ]
    """Names of the arm joints to drive."""

    frequency_hz: float = 0.5
    """Swing frequency in Hz."""

    amplitude: float = 0.3
    """Swing amplitude in radians (applied to all arm joints)."""

    phase: list[float] = [
        0.0, 0.5, 1.0, 1.5,
        2.0, 2.5, 3.0, 3.5,
    ]
    """Per-joint phase offsets (radians) to desynchronize the swing."""
