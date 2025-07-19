#!/usr/bin/env python3
"""
Usage: ./python.sh scene_loader_from_yaml.py scene.yaml
"""

from isaacsim import SimulationApp
kit = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

import omni.usd, omni.kit.commands as okc
from pxr import Gf, Sdf, UsdPhysics, UsdLux, PhysxSchema
from isaacsim.core.prims import RigidPrim
import pathlib, random, yaml, numpy as np

PARTNET_ROOT = pathlib.Path("/home/szwang/synthesis/data/dataset")
OBJAVERSE_ROOT = pathlib.Path("/home/szwang/synthesis/data/objaverse/data/obj")

### -------------------------------------------------------------- ###
def load_partnet(id_str: str, stage) -> str:
    """Returns the prim path after importing PartNet URDF."""
    urdf = PARTNET_ROOT / id_str / "mobility.urdf"
    if not urdf.exists():
        raise FileNotFoundError(f"PartNet URDF not found: {urdf}")
    # Parse and import
    _, prim_path = okc.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf),
        import_config=okc.execute("URDFCreateImportConfig")[1],
        get_articulation_root=True,
    )
    return prim_path

def load_objaverse(uid_list, stage) -> str:
    """Returns the prim path after importing Objaverse URDF."""
    uid = random.choice(uid_list)
    urdf = OBJAVERSE_ROOT / uid / "material.urdf"
    if not urdf.exists():
        raise FileNotFoundError(f"Objaverse URDF not found: {urdf}")
    _, prim_path = okc.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf),
        import_config=okc.execute("URDFCreateImportConfig")[1],
        get_articulation_root=True,
    )
    return prim_path

### -------------------------------------------------------------- ###
def apply_transform(prim_path: str, center, size):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    # 如果 prim 是不是 Xform 类型，则转换为 Xform 类型
    if prim.GetTypeName() != "Xform":
        prim = prim.GetParent()  # 获取其父级 Xform

    xform = prim

    print(f"[INFO] Applying transform to {prim_path} with center={center}, size={size}")

    # 确保 center 是有效的三元组并传入 Gf.Vec3d
    if isinstance(center, (list, tuple)) and len(center) == 3:
        tr = Gf.Vec3d(*center)  # 传递有效的三元组
    else:
        print(f"[ERROR] Invalid center value: {center}. Using default [0, 0, 0].")
        tr = Gf.Vec3d(0.0, 0.0, 0.0)

    # Apply translation (center)
    xform.GetAttribute("xformOp:translate").Set(tr) if xform.HasAttribute("xformOp:translate") \
        else xform.AddTranslateOp().Set(tr)

    # Apply scale (size)
    if size is not None:
        sc = size if isinstance(size, (list, tuple)) else [size]*3
        xform.AddScaleOp().Set(Gf.Vec3f(*sc))

# Set the movable property of the prim
def set_movable(prim_path: str, movable: bool):
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(prim_path)
    body_api = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)
    if not body_api:
        return
    body_api.CreateKinematicEnabledAttr(not movable)

### -------------------------------------------------------------- ###
def main(scene_yaml):
    # ---------- A. Read YAML ----------
    with open(scene_yaml, "r") as f:
        data = yaml.safe_load(f)
    objects = data

    stage = omni.usd.get_context().get_stage()

    # ---------- B. Set up physics (done once) ----------
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    okc.execute("AddGroundPlaneCommand", stage=stage, planePath="/ground",
                axis="Z", size=1500, position=Gf.Vec3f(0,0,-0.5), color=Gf.Vec3f(0.5))
    UsdLux.DistantLight.Define(stage, Sdf.Path("/Sun")).CreateIntensityAttr(500)

    prim_paths = []
    # ---------- C. Import objects ----------
    for idx, obj in enumerate(objects):
        if "reward_asset_path" in obj:                    # PartNet
            prim_path = load_partnet(str(obj["reward_asset_path"]), stage)
        elif "uid" in obj:                            # Objaverse
            prim_path = load_objaverse(obj["uid"], stage)
        elif "all_uid" in obj:                            # Objaverse
            prim_path = load_objaverse(obj["all_uid"], stage)
        else:
            print(f"[WARN] object {idx} missing id, skip")
            continue
        prim_paths.append(prim_path)

        # Store attributes for later application
        obj["_prim_path"] = prim_path

    # ---------- D. Run first frame to initialize physics (SimView generated) ----------
    omni.timeline.get_timeline_interface().play()
    kit.update()

    # ---------- E. Apply transforms and set movement properties ----------
    for obj in objects:
        prim_path = obj.get("_prim_path")
        if not prim_path:
            continue
        apply_transform(prim_path,
                        center=obj.get("center", [0,0,0]),
                        size=obj.get("size"))
        set_movable(prim_path, obj.get("movable", True))

    # ---------- F. Continue simulation until window closes ----------
    while kit.is_running():
        kit.update()

    omni.timeline.get_timeline_interface().stop()
    kit.close()

### -------------------------------------------------------------- ###
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: scene_loader_from_yaml.py scene.yaml"); sys.exit(1)
    main(sys.argv[1])
