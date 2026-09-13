# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--debug", action="store_true", help="Enable debug print messages")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from pathlib import Path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import bike_isaac.tasks  # noqa: F401


def main():
    """Random actions agent with Isaac Lab environment."""

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    env_cfg.debug = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()

    # ------------------------------------------------------------------
    # CSV settings
    # ------------------------------------------------------------------
    base_csv_path = Path("bike_velocity_data.csv")

    # If the file already exists, add _1, _2, _3, ...
    csv_path = base_csv_path
    counter = 1

    while csv_path.exists():
        csv_path = base_csv_path.with_name(
            f"{base_csv_path.stem}_{counter}{base_csv_path.suffix}"
        )
        counter += 1
    # get unwrapped environment
    unwrapped_env = env.unwrapped

    # get back tire actuator
    back_tire_actuator = unwrapped_env.robot.actuators["back_tire_pitch"]

    # get velocity limit used by the actuator
    velocity_limit = back_tire_actuator.velocity_limit_sim[:, 0]

    print(f"[INFO]: Back tire velocity limit: {velocity_limit}")

    # ------------------------------------------------------------------
    # Open CSV file
    # ------------------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # CSV header
        writer.writerow([
            "velocity_target",
            "actual_velocity",
            "applied_torque",
        ])

        # simulate environment
        while simulation_app.is_running():

            # run everything in inference mode
            with torch.inference_mode():

                # ------------------------------------------------------
                # Action
                # ------------------------------------------------------
                ACTION_VALUE = 0.03

                actions = torch.full(
                    env.action_space.shape,
                    ACTION_VALUE,
                    device=unwrapped_env.device,
                )

                # ------------------------------------------------------
                # Apply action
                # ------------------------------------------------------
                env.step(actions)

                # ------------------------------------------------------
                # Velocity target
                #
                # action [-1, 1]
                #       ×
                # velocity_limit_sim [rad/s]
                #
                # Example:
                #   0.03 × 20 = 0.6 rad/s
                # ------------------------------------------------------
                velocity_target = (
                    actions[:, 0] * velocity_limit
                ).item()

                # ------------------------------------------------------
                # Actual joint velocity
                # ------------------------------------------------------
                actual_velocity = (
                    unwrapped_env.robot.data.joint_vel[
                        0,
                        unwrapped_env._back_tire_dof_idx[0],
                    ]
                ).item()

                # ------------------------------------------------------
                # Applied torque
                # ------------------------------------------------------
                applied_torque = (
                    unwrapped_env.robot.data.applied_torque[
                        0,
                        unwrapped_env._back_tire_dof_idx[0],
                    ]
                ).item()

                # ------------------------------------------------------
                # Write CSV
                # ------------------------------------------------------
                writer.writerow([
                    velocity_target,
                    actual_velocity,
                    applied_torque,
                ])

    # close environment
    env.close()

    print(f"[INFO]: CSV saved to {csv_path}")


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()