from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import omni.usd
from pxr import Usd, UsdPhysics
from bike_isaac.tasks.direct.bike_isaac.bike_isaac_env import BikeIsaacEnv
from bike_isaac.tasks.direct.bike_isaac.bike_isaac_env_cfg import BikeIsaacEnvCfg
print("=" * 100)
print("RUNTIME COLLISION DETAIL INSPECTION")
print("=" * 100)
env_cfg = BikeIsaacEnvCfg()
env = BikeIsaacEnv(cfg=env_cfg)
env.reset()
stage = omni.usd.get_context().get_stage()
root_path = "/World/envs/env_0/Robot"
print(f"Root: {root_path}")
print("=" * 100)
collision_roots = [
    f"{root_path}/bike_V3_mjcf/main_body/main_body/collisions",
    f"{root_path}/bike_V3_mjcf/main_body/back_tire/collisions",
    f"{root_path}/bike_V3_mjcf/main_body/fork/collisions",
    f"{root_path}/bike_V3_mjcf/main_body/front_tire/collisions",
]
for collision_root_path in collision_roots:
    print()
    print("=" * 100)
    print(f"COLLISION ROOT")
    print(f"{collision_root_path}")
    print("=" * 100)
    prim = stage.GetPrimAtPath(collision_root_path)
    if not prim.IsValid():
        print("[ERROR] Collision root does not exist")
        continue
    print(f"Valid      : {prim.IsValid()}")
    print(f"Type       : {prim.GetTypeName()}")
    print(f"IsInstance : {prim.IsInstance()}")
    if not prim.IsInstance():
        print("[WARNING] This collision root is not an instance")
        continue
    prototype = prim.GetPrototype()
    print(f"Prototype  : {prototype.GetPath()}")
    for p in Usd.PrimRange(prototype):
        if p.GetTypeName() != "Mesh":
            continue
        has_collision = p.HasAPI(UsdPhysics.CollisionAPI)
        has_mesh_collision = p.HasAPI(UsdPhysics.MeshCollisionAPI)
        if not has_collision and not has_mesh_collision:
            continue
        print()
        print("-" * 80)
        print(f"Collision Mesh : {p.GetPath()}")
        print(f"Type           : {p.GetTypeName()}")
        print(f"CollisionAPI   : {has_collision}")
        print(f"MeshCollision  : {has_mesh_collision}")
        if has_collision:
            collision_api = UsdPhysics.CollisionAPI(p)
            enabled_attr = collision_api.GetCollisionEnabledAttr()
            print(f"CollisionEnabled: {enabled_attr.Get() if enabled_attr else None}")
        contact_attr = p.GetAttribute("physxCollision:contactOffset")
        rest_attr = p.GetAttribute("physxCollision:restOffset")
        print(f"ContactOffset  : {contact_attr.Get() if contact_attr and contact_attr.HasAuthoredValue() else None}")
        print(f"RestOffset     : {rest_attr.Get() if rest_attr and rest_attr.HasAuthoredValue() else None}")
        approximation_names = [
            "physxMeshCollision:approximation",
            "physics:approximation",
        ]
        found_approximation = False
        for attr_name in approximation_names:
            attr = p.GetAttribute(attr_name)
            if attr:
                print(f"{attr_name}: {attr.Get()}")
                found_approximation = True
        if not found_approximation:
            print("CollisionApproximation: None")
        print()
        print("AUTHORED ATTRIBUTES")
        for attr in p.GetAttributes():
            name = attr.GetName()
            if "collision" in name.lower() or "friction" in name.lower() or "contact" in name.lower() or "rest" in name.lower():
                try:
                    value = attr.Get()
                except Exception:
                    value = "<unreadable>"
                print(f"  {name} = {value}")
print()
print("=" * 100)
print("MATERIAL / FRICTION INSPECTION")
print("=" * 100)
for prim in stage.Traverse():
    if not prim.IsValid():
        continue
    if prim.GetTypeName() != "Material":
        continue
    path = str(prim.GetPath())
    if "bike" not in path.lower() and "ground" not in path.lower():
        continue
    print(f"\nMaterial: {path}")
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if "friction" in name.lower() or "restitution" in name.lower() or "physics" in name.lower():
            try:
                print(f"  {name} = {attr.Get()}")
            except Exception:
                print(f"  {name} = <unreadable>")
print()
print("=" * 100)
print("DONE")
print("=" * 100)
env.close()
simulation_app.close()