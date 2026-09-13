# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
look here https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.schemas.html#module-isaaclab.sim.schemas
Configuration for the bike robots designed by hossyan and shimbei
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# 1. 現在のファイルがあるディレクトリを基準にする
current_dir = Path(__file__).resolve().parent


##
# Configuration
##

BIKE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        activate_contact_sensors=True,

        # usd_path="/home/shin-linux/bike_isaac/assets/bike_v3/bike_v3_readable.usda",
        usd_path="/home/shin-linux/bike_isaac/assets/bike_v3/bike_v3.usd",

        # ============================================================
        # Rigid body properties
        #
        # MuJoCo:
        # <option iterations="30"/>
        # ============================================================
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=2.0,  # めり込んだときの最大押し返し速度
            enable_gyroscopic_forces=True,  # ジャイロ効果を考慮するか

            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),

        # ============================================================
        # Articulation properties
        # ============================================================
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=False,
            fix_root_link=False,

            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),

        # ============================================================
        # Contact properties
        #
        # MuJoCo:
        # geom solref / solimp
        #
        # ※ PhysXと1:1対応ではない。
        # ============================================================
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=-0.0, 
            torsional_patch_radius=0.001,
            min_torsional_patch_radius=0.001,
        ),
    ),

    # ================================================================
    # Initial state
    # ================================================================
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "fork_yaw": math.pi / 3,
            "back_tire_pitch": 0.0,
        },
    ),

    actuators={
        "fork_yaw": ImplicitActuatorCfg(
            joint_names_expr=["fork_yaw"],

            # 現在のIsaac Lab側のPD設定
            stiffness=10000000.0,
            damping=200.0,
            # friction=0.3,
            # dynamic_friction=0.161745,
            # viscous_friction=0.0,
        ),

        "back_tire_pitch": ImplicitActuatorCfg(
            joint_names_expr=["back_tire_pitch"],

            # 元のIsaac Lab設定
            effort_limit_sim=12.0,
            velocity_limit_sim=20.0,

            # Motor actuatorなのでPD stiffnessは0
            stiffness=0.0,
            damping=10,
            friction=0.27675, # feering.
            dynamic_friction=0.1, # from MuJoCo: friction="0.5", dynamic_friction="0.27675"
            # --------------------------------------------------------
            # Joint viscous friction
            #
            # MuJoCo:
            # damping="5.0e-5"
            # --------------------------------------------------------
            viscous_friction=5.0e-5,
            armature=0.0105,
        ),
    },

    # soft_joint_pos_limit_factor=1.0,
)