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
from isaaclab.utils.math import sample_uniform, euler_xyz_from_quat  # euler_xyz_from_quat を追加
from isaaclab.sensors import ImuCfg, Imu  # Imu クラスを追加
from .bike_isaac_env_cfg import BikeIsaacEnvCfg


class BikeIsaacEnv(DirectRLEnv):
    cfg: BikeIsaacEnvCfg
    ENV_REGEX_NS = "/World/envs/env_.*/Robot"

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

        # 修正: Cfgをインスタンス変数として保持（_setup_sceneで実体を生成するため）
        self.imu_cfg = ImuCfg(prim_path=f"{self.ENV_REGEX_NS}/main_body/main_body/sites/imu_site", 
                              update_period=1 / 200, history_length=0.0, debug_vis=True)

        self.imu = Imu(self.imu_cfg)


        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
            
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # 修正: IMUセンサーをsceneに登録
        self.scene.sensors["imu_site"] = self.imu
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 修正: 構文エラーの修正 (torch.tensor -> torch.full)
        target_angle = torch.full((self.num_envs, 1), math.pi/3, device=self.device)
        
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self._back_tire_dof_idx)
        # 修正: メソッド名の間違いを修正 (set_position_velocity_target -> set_joint_position_target)
        self.robot.set_joint_position_target(target_angle, joint_ids=self._fork_dof_idx)

        # self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        compute_imu_data(self, self.scene["imu_site"].data)
        # print("imu_data:", self.scene["imu_site"].data)
        
        # 修正: 重複していた calculated_roll_vel.unsqueeze(dim=1) を1つ削除
        obs = torch.cat(
            (
                self.calculated_roll.unsqueeze(dim=1),
                self.calculated_roll_vel.unsqueeze(dim=1),
                self.actions[:, 0].unsqueeze(dim=1)
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reset_terminated = self._get_dones()[0]
        # 修正: 関数の引数定義の順番と合わせるため、キーワード引数で明示的に指定
        total_reward = compute_rewards(
            rew_scale_terminated=self.cfg.rew_scale_terminated,
            rew_scale_roll_angle=self.cfg.rew_scale_roll_angle,
            rew_scale_roll_vel=self.cfg.rew_scale_roll_vel,
            rew_scale_stable=self.cfg.rew_scale_stable,
            roll_angle=self.calculated_roll,
            roll_vel=self.calculated_roll_vel,
            input_back_tire=self.actions[:, 0],
            reset_terminated=reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.calculated_roll.abs() > math.radians(15)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset calculated roll and roll velocity
        self.calculated_roll[env_ids] = 0.0
        self.calculated_roll_vel[env_ids] = 0.0

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
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
    # 修正: Isaac Labの報酬は1次元 [num_envs] である必要があるため、unsqueezeを削除
    rew_roll_angle = rew_scale_roll_angle * torch.abs(roll_angle)
    rew_roll_vel = rew_scale_roll_vel * torch.abs(roll_vel)
    rew_stable = rew_scale_stable * torch.abs(input_back_tire)
    total_reward = rew_termination + rew_roll_angle + rew_roll_vel + rew_stable
    return total_reward

# 修正: クラス外部からselfを受け取り、内部処理を実装
def compute_imu_data(self, imu_data) -> None:
    # IMUのクォータニオンからオイラー角を計算してロール角を取得
    # プロパティ名を orientation_w から quat_w に変更
    roll, pitch, yaw = euler_xyz_from_quat(imu_data.quat_w)
    
    self.calculated_roll = roll
    self.calculated_roll_vel = imu_data.ang_vel_b[:, 0]  # ローカルのX軸角速度