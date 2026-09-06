# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--env_spacing", type=float, default=None, help="Number of spaces to simulate.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable physics debug output.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
import omni.usd
from pxr import Usd, UsdPhysics

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import bike_isaac.tasks  # noqa: F401


def debug_print_physics(env):
    robot = env.unwrapped.robot
    stage = omni.usd.get_context().get_stage()
    print("\n" + "=" * 100)
    print("BIKE PHYSICS DEBUG")
    print("=" * 100)
    print("\n[1] JOINT INFORMATION")
    found_joint = False
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "back_tire_pitch" not in path:
            continue
        found_joint = True
        print(f"Joint Prim : {path}")
        print(f"Prim Type  : {prim.GetTypeName()}")
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            axis = joint.GetAxisAttr().Get()
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            print("Joint Type : RevoluteJoint")
            print(f"Joint Axis : {axis}")
            print(f"Body0      : {body0}")
            print(f"Body1      : {body1}")
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
            axis = joint.GetAxisAttr().Get()
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            print("Joint Type : PrismaticJoint")
            print(f"Joint Axis : {axis}")
            print(f"Body0      : {body0}")
            print(f"Body1      : {body1}")
        else:
            print(f"Joint Type : {prim.GetTypeName()}")
    if not found_joint:
        print("back_tire_pitch joint prim was not found by USD path search.")
    print("\n[2] BACK TIRE DOF INFORMATION")
    back_tire_ids, _ = robot.find_joints("back_tire_pitch")
    print(f"DOF indices : {back_tire_ids}")
    if len(back_tire_ids) > 0:
        joint_id = back_tire_ids[0]
        print(f"Joint name       : {robot.data.joint_names[joint_id]}")
        print(f"Joint position   : {robot.data.joint_pos[0, joint_id].item():.6f} rad")
        print(f"Joint velocity   : {robot.data.joint_vel[0, joint_id].item():.6f} rad/s")
        if hasattr(robot.data, "applied_torque"):
            print(f"Applied torque   : {robot.data.applied_torque[0, joint_id].item():.6f} N*m")
        if hasattr(robot.data, "computed_torque"):
            print(f"Computed torque  : {robot.data.computed_torque[0, joint_id].item():.6f} N*m")
    else:
        print("back_tire_pitch DOF was not found.")
    print("\n[3] COLLISION PRIMS")
    collision_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
            print(f"Collision : {prim.GetPath()}")
    print(f"Total collision prims : {collision_count}")
    print("\n[4] PHYSICS MATERIAL / FRICTION")
    material_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            material_count += 1
            material = UsdPhysics.MaterialAPI(prim)
            static_friction = material.GetStaticFrictionAttr().Get()
            dynamic_friction = material.GetDynamicFrictionAttr().Get()
            restitution = material.GetRestitutionAttr().Get()
            print(f"Material : {prim.GetPath()}")
            print(f"  Static friction  : {static_friction}")
            print(f"  Dynamic friction : {dynamic_friction}")
            print(f"  Restitution      : {restitution}")
    if material_count == 0:
        print("No UsdPhysics.MaterialAPI prim was found.")
    print(f"Total physics materials : {material_count}")
    print("\n[5] BODY HEIGHT INFORMATION")
    body_names = robot.data.body_names
    body_pos = robot.data.body_pos_w[0]
    for i, name in enumerate(body_names):
        z = body_pos[i, 2].item()
        print(f"{i:3d} : {name:40s} Z = {z:.5f} m")
    print("\n[6] ROOT INFORMATION")
    print(f"Root position       : {robot.data.root_pos_w[0].detach().cpu().numpy()}")
    print(f"Root linear velocity: {robot.data.root_lin_vel_w[0].detach().cpu().numpy()}")
    print(f"Root angular velocity: {robot.data.root_ang_vel_w[0].detach().cpu().numpy()}")
    print("\n[7] ENVIRONMENT INFORMATION")
    print(f"Environment spacing : {env.unwrapped.env_spacing}")
    print(f"Number of envs      : {env.unwrapped.num_envs}")
    print(f"Environment device  : {env.unwrapped.device}")
    print("=" * 100)
    print()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing if args_cli.env_spacing is not None else env_cfg.scene.env_spacing
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.debug:
        debug_print_physics(env)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()