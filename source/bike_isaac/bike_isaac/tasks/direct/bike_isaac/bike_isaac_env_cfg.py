# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib, math

# bike_cfg_module = importlib.import_module(
#     "/home/shin-linux/bike_isaac/source/bike_isaac/bike_isaac/bike_cfg"
# )
# BIKE_CFG = bike_cfg_module.BIKE_CFG

from bike_isaac.bike_cfg import BIKE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
import isaaclab.envs.mdp as mdp

roll_range = math.radians(3)  # ±10度の範囲でランダム化

@configclass
class EventCfg:
    """Configuration for randomization."""
    # 1. 物理パラメータ：質量・重心・慣性モーメントのランダム化（エピソードリセットごとに再抽選）
    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset", # "startup" から "reset" に変更
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="main_body"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # 2. アクチュエータパラメータ（PDゲイン）のランダム化（エピソードリセットごとに再抽選）
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset", # "startup" から "reset" に変更
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="back_tire_pitch"),
            # "stiffness_distribution_params": (0.9, 1.1),
            # "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 3. アクチュエータパラメータ（）のランダム化（エピソードリセットごとに再抽選）
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset", # "startup" から "reset" に変更
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="back_tire_pitch"),
            # "armature_distribution_params": (0.9, 1.1),
            "friction_distribution_params": (0.8, 1.2), # 静と動の摩擦係数のランダム化
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_root_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "roll": (-roll_range, roll_range),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

@configclass
class BikeIsaacEnvCfg(DirectRLEnvCfg):
    debug: bool = False

    # env
    decimation = 1      # rendering frequency with frame
    episode_length_s = 30.0  # maximum episode length in seconds
    action_space = 1     # - spaces definition
    observation_space = 2  # - spaces definition
    state_space = 0     # 保持すべき内部状態の数

    # simulation. recommended is 1/120
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # robot(s)
    # /World/envs/env_.*/bike_V3_mjcf というプリムパスでは、
    # シーンのすべてのコピーに bike_V3_mjcf という名前のロボットが存在することを暗黙的に示しています。
    robot_cfg: ArticulationCfg = BIKE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0, operation="abs"),
    )

    # custom parameters/scales
    # - controllable joint
    fork_dof_name = "fork_yaw"
    back_tire_dof_name = "back_tire_pitch"

    # - action scale. now don't use torque control, so this is not used
    # action_scale = 0.0  # [N]

    # - reward scales
        # reward
    rew_scale_roll_angle = 1.0
        # penalty
    rew_scale_roll_vel = -0.01
    rew_scale_stable = -0.00
    rew_scale_terminated = -5.0

    # - reset states/conditions
    initial_roll_angle_range = [-0.25, 0.25]  # roll angle sample range on reset [rad]
    max_roll_angle = 3.0  # reset if cart exceeds this position [m]