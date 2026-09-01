# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the bike robots designed by hossyan and shimbei
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path
import math

# 1. 現在のファイルがあるディレクトリを基準にする
current_dir = Path(__file__).resolve().parent
##
# Configuration
##

BIKE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/shin-linux/bike_isaac/assets/bike_V3_mjcf/bike_V3_mjcf_revised.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,           # Important for accurate dynamics
            solver_position_iteration_count=8,       # Balance accuracy vs performance
            solver_velocity_iteration_count=1,            
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        # Contact properties
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset = 0.005,       # 5mm contact detection distance
            rest_offset = - 0.003,       #負の値でどれだけめり込めるか、つまりゴムによってどれだけ沈み込むかを定義
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Body joints
            "fork_yaw": math.pi / 3,
            "back_tire_pitch": 0.0,},
    ),
    actuators={
        # Body lift and torso actuators
        "fork_yaw": ImplicitActuatorCfg(
            joint_names_expr=["fork_yaw"],
            # effort_limit_sim=10000.0,
            # velocity_limit_sim=2.61,
            stiffness=10000000.0,
            damping=200.0,
        ),
        # Head actuators
        "back_tire_pitch": ImplicitActuatorCfg(
            joint_names_expr=["back_tire_pitch"],
            # effort_limit_sim=50.0,
            velocity_limit_sim=210.0,
            stiffness=100.0,
            damping=0.1,
            # aramature=0.01,
            # friction=0.1,
            # dynamic_friction=0.1,
        ),
    },
    # soft_joint_pos_limit_factor=1.0,
)