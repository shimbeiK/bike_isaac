from __future__ import annotations

from isaacsim import SimulationApp
# ヘッドレスモード（GUIなし）でIsaac Simをバックグラウンド起動し、パスを通す
simulation_app = SimulationApp({"headless": True})
from omni.isaac.kit import SimulationApp
import omni.usd
from pxr import Usd
from pathlib import Path
import sys

def print_prim_tree(prim, indent=0):
    print("  " * indent + prim.GetName())
    for child in prim.GetChildren():
        print_prim_tree(child, indent + 1)

def print_prim_details(prim):
    print("=" * 100)
    print("PRIM DETAILS")
    print("=" * 100)
    print(f"Path       : {prim.GetPath()}")
    print(f"Name       : {prim.GetName()}")
    print(f"Type       : {prim.GetTypeName()}")
    print(f"Valid      : {prim.IsValid()}")
    print(f"Active     : {prim.IsActive()}")
    print(f"Instance   : {prim.IsInstance()}")
    print(f"Prototype  : {prim.GetPrototype() if prim.IsInstance() else None}")
    print(f"Has API    : {prim.HasAPI(Usd.Prim)}")
    print("=" * 100)

def main():
    usd_path = "/home/shin-linux/bike_isaac/assets/bike_v3/bike_v3.usd"
    if not Path(usd_path).is_file():
        print(f"ERROR: USD file not found:")
        print(usd_path)
        return
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.DefinePrim("/World", "Xform")
    envs_prim = stage.DefinePrim("/World/envs", "Xform")
    env_prim = stage.DefinePrim("/World/envs/env_0", "Xform")
    robot_prim = stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    bike_prim = stage.DefinePrim("/World/envs/env_0/Robot/bike_V3_mjcf", "Xform")
    bike_prim.GetReferences().AddReference(usd_path)
    simulation_app.update()
    print("")
    print("=" * 100)
    print("BIKE PRIM TREE")
    print("=" * 100)
    print(f"USD       : {usd_path}")
    print(f"Root Prim : {bike_prim.GetPath()}")
    print("=" * 100)
    print("")
    print_prim_tree(bike_prim)
    print("")
    target_paths = [
        "/World/envs/env_0/Robot/bike_V3_mjcf",
        "/World/envs/env_0/Robot/bike_V3_mjcf/main_body",
        "/World/envs/env_0/Robot/bike_V3_mjcf/main_body/main_body",
        "/World/envs/env_0/Robot/bike_V3_mjcf/main_body/main_body/sites",
        "/World/envs/env_0/Robot/bike_V3_mjcf/main_body/main_body/sites/imu_site",
    ]
    print("")
    print("=" * 100)
    print("IMPORTANT PRIMS")
    print("=" * 100)
    for path in target_paths:
        prim = stage.GetPrimAtPath(path)
        print("")
        print_prim_details(prim)
    print("")
    print("=" * 100)
    print("ALL PRIMS")
    print("=" * 100)
    for prim in stage.Traverse():
        print(prim.GetPath())
    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()