import argparse

from isaaclab.app import AppLauncher


# ================================================================
# Isaac Lab application
# ================================================================

parser = argparse.ArgumentParser(
    description="Inspect bike USD structure."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# IMPORTANT:
# pxr must be imported after Isaac Sim has started.
from pxr import Usd, UsdPhysics


# ================================================================
# USD path
# ================================================================

USD_PATH = "/home/shin-linux/bike_isaac/assets/bike.usd"


# ================================================================
# Main
# ================================================================

def main():

    # ------------------------------------------------------------
    # Open USD
    # ------------------------------------------------------------

    stage = Usd.Stage.Open(USD_PATH)

    if stage is None:
        raise RuntimeError(
            f"Failed to open USD:\n{USD_PATH}"
        )

    # ============================================================
    # ALL PRIMS
    # ============================================================

    print()
    print("=" * 80)
    print("ALL PRIMS")
    print("=" * 80)

    for prim in stage.Traverse():
        print(prim.GetPath())

    # ============================================================
    # ARTICULATION ROOTS
    # ============================================================

    print()
    print("=" * 80)
    print("ARTICULATION ROOTS")
    print("=" * 80)

    articulation_count = 0

    for prim in stage.Traverse():

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):

            articulation_count += 1

            print(
                f"[{articulation_count}] "
                f"{prim.GetPath()}"
            )

    print()
    print(
        f"Total articulation roots: "
        f"{articulation_count}"
    )

    # ============================================================
    # ARTICULATION ROOT DETAILS
    # ============================================================

    print()
    print("=" * 80)
    print("ARTICULATION ROOT DETAILS")
    print("=" * 80)

    for prim in stage.Traverse():

        if not prim.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            continue

        print()
        print(f"Prim: {prim.GetPath()}")

        print(
            f"  Type: "
            f"{prim.GetTypeName()}"
        )

        parent = prim.GetParent()

        print(
            f"  Parent: "
            f"{parent.GetPath()}"
        )

        print("  Children:")

        children = prim.GetChildren()

        if len(children) == 0:
            print("    (none)")
        else:
            for child in children:
                print(
                    f"    {child.GetPath()}"
                )

    # ============================================================
    # JOINTS
    # ============================================================

    print()
    print("=" * 80)
    print("JOINTS")
    print("=" * 80)

    joint_count = 0

    for prim in stage.Traverse():

        if prim.IsA(UsdPhysics.Joint):

            joint_count += 1

            print(
                f"[{joint_count}] "
                f"{prim.GetPath()}"
            )

    print()
    print(
        f"Total joints: "
        f"{joint_count}"
    )

    # ============================================================
    # JOINT DETAILS
    # ============================================================

    print()
    print("=" * 80)
    print("JOINT DETAILS")
    print("=" * 80)

    for prim in stage.Traverse():

        if not prim.IsA(UsdPhysics.Joint):
            continue

        print()
        print(f"Joint: {prim.GetPath()}")

        # --------------------------------------------------------
        # body0
        # --------------------------------------------------------

        body0_rel = prim.GetRelationship(
            "physics:body0"
        )

        print("  body0:")

        if body0_rel:
            targets = body0_rel.GetTargets()

            if targets:
                for target in targets:
                    print(
                        f"    {target}"
                    )
            else:
                print("    (none)")
        else:
            print("    (relationship not found)")

        # --------------------------------------------------------
        # body1
        # --------------------------------------------------------

        body1_rel = prim.GetRelationship(
            "physics:body1"
        )

        print("  body1:")

        if body1_rel:
            targets = body1_rel.GetTargets()

            if targets:
                for target in targets:
                    print(
                        f"    {target}"
                    )
            else:
                print("    (none)")
        else:
            print("    (relationship not found)")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print()


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()