# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math
import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, euler_xyz_from_quat
from isaaclab.sensors import ImuCfg, Imu
from .bike_isaac_env_cfg import BikeIsaacEnvCfg

def debug_print(env, *args, **kwargs):
    if env.cfg.debug:
        print(*args, **kwargs)

class BikeIsaacEnv(DirectRLEnv):
    cfg: BikeIsaacEnvCfg
    IMU_SITE_PATH = "/World/envs/env_.*/Robot/bike_V3_mjcf/main_body/main_body/sites/imu_site"

    def __init__(self, cfg: BikeIsaacEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._fork_dof_idx, _ = self.robot.find_joints(self.cfg.fork_dof_name)
        self._back_tire_dof_idx, _ = self.robot.find_joints(self.cfg.back_tire_dof_name)
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.calculated_roll = torch.zeros(self.num_envs, device=self.device)
        self.calculated_roll_vel = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.imu_cfg = ImuCfg(
            prim_path=self.IMU_SITE_PATH,
            update_period=1 / 200,
            history_length=0.0,
            debug_vis=True,
        )
        self.imu = Imu(self.imu_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["imu_site"] = self.imu
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        target_angle = torch.full((self.num_envs, 1), math.pi / 3, device=self.device)
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self._back_tire_dof_idx)
        self.robot.set_joint_position_target(target_angle, joint_ids=self._fork_dof_idx)
        debug_print(self, "actual output:", self.actions[:, 0])
        debug_print(self, "actual velocity:", self.robot.data.joint_vel[0, self._back_tire_dof_idx[0]])
        debug_print(self, "applied torque:", self.robot.data.applied_torque[0, self._back_tire_dof_idx[0]])

    def _get_observations(self) -> dict:
        compute_imu_data(self, self.scene["imu_site"].data)
        obs = torch.cat(
            (
                self.calculated_roll.unsqueeze(dim=1),
                self.calculated_roll_vel.unsqueeze(dim=1),
                self.actions[:, 0].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reset_terminated = self._get_dones()[0]
        total_reward, rew_termination, rew_roll_angle, rew_roll_vel, rew_stable = compute_rewards(
            rew_scale_terminated=self.cfg.rew_scale_terminated,
            rew_scale_roll_angle=self.cfg.rew_scale_roll_angle,
            rew_scale_roll_vel=self.cfg.rew_scale_roll_vel,
            rew_scale_stable=self.cfg.rew_scale_stable,
            roll_angle=self.calculated_roll,
            roll_vel=self.calculated_roll_vel,
            input_back_tire=self.actions[:, 0],
            reset_terminated=reset_terminated,
        )
        self.extras["log"] = {
            "Reward/termination": rew_termination.mean(),
            "Reward/roll_angle": rew_roll_angle.mean(),
            "Reward/roll_vel": rew_roll_vel.mean(),
            "Reward/stable": rew_stable.mean(),
            "Reward/total": total_reward.mean(),
        }
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.calculated_roll.abs() > math.radians(10)
        if self.cfg.debug:
            time_out = torch.zeros_like(time_out)
            terminated = torch.zeros_like(terminated)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.calculated_roll[env_ids] = 0.0
        self.calculated_roll_vel[env_ids] = 0.0
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

@torch.jit.script
def compute_rewards(
    rew_scale_terminated: float,
    rew_scale_roll_angle: float,
    rew_scale_roll_vel: float,
    rew_scale_stable: float,
    roll_angle: torch.Tensor,
    roll_vel: torch.Tensor,
    input_back_tire: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_roll_angle = rew_scale_roll_angle * torch.abs(roll_angle)
    rew_roll_vel = rew_scale_roll_vel * torch.abs(roll_vel)
    rew_stable = rew_scale_stable * torch.abs(input_back_tire)
    total_reward = rew_termination + rew_roll_angle + rew_roll_vel + rew_stable
    return total_reward, rew_termination, rew_roll_angle, rew_roll_vel, rew_stable

def compute_imu_data(self, imu_data) -> None:
    roll, pitch, yaw = euler_xyz_from_quat(imu_data.quat_w)
    self.calculated_roll = roll
    self.calculated_roll_vel = imu_data.ang_vel_b[:, 0]