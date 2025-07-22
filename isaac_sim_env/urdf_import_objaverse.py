#!/usr/bin/env python3
# 适用于 *仅包含一个 link* 的 URDF（无任何 joint）
# 启动：./python.sh cube_loader.py

from isaacsim import SimulationApp
kit = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

import omni.usd, omni.kit.commands as okc
from pxr import Gf, Sdf, UsdPhysics, UsdLux, PhysxSchema
from isaacsim.core.prims import RigidPrim
import pathlib
from pxr import UsdGeom

# ---------- 1. 导入 URDF ----------
cfg_status, cfg = okc.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints      = False
cfg.convex_decomp           = False
cfg.import_inertia_tensor   = True
cfg.fix_base                = False
cfg.distance_scale          = 1.0

URDF_FILE = "/home/szwang/synthesis/data/objaverse/data/obj/939bce9ccaec4d5ab3404dca172d2f45/material.urdf"
_, prim_path = okc.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_FILE,
    import_config=cfg,
    get_articulation_root=True,
)

stage = omni.usd.get_context().get_stage()

# ---------- 2. 物理场景 ----------
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))

# ground plane
okc.execute("AddGroundPlaneCommand", stage=stage, planePath="/ground",
            axis="Z", size=1500, position=Gf.Vec3f(0, 0, -0.5), color=Gf.Vec3f(0.5))

# 环境灯
UsdLux.DistantLight.Define(stage, Sdf.Path("/Sun")).CreateIntensityAttr(500)

# ---------- 3. 初始化刚体 ----------
# baseLink 可能在 prim_path 下，也可能 prim_path 本身就是 Mesh
# ① 创建 RigidPrim 但先 **不要** initialize
base_path = prim_path + "/baseLink" if stage.GetPrimAtPath(prim_path + "/baseLink").IsValid() else prim_path
rigid = RigidPrim(base_path)

# ② 开始仿真，让物理管线真正起动
omni.timeline.get_timeline_interface().play()
kit.update()          # <- 第一帧；PhysicsSimulationView 此时才生成


# ③ 初始化刚体
rigid.initialize()

# ---------- ✳️ 设置是否可动 ----------
movable = False  # ← 修改这里控制是否允许移动

prim = stage.GetPrimAtPath(base_path)
from pxr import PhysxSchema

# 设置 USD rigid 属性
rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
rigid_api.CreateRigidBodyEnabledAttr(movable)

# 设置 PhysX motion 属性
prim.CreateAttribute("physx:rigidBody:motionEnabled", Sdf.ValueTypeNames.Bool).Set(movable)


print(f"[INFO] Set object movable = {movable}")

# ---------- ✳️ 设置位置和缩放 ----------
xform = UsdGeom.Xformable(prim)

# 设置位置
for op in xform.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d(5.0, 5.0, 0.0))
        break
else:
    xform.AddTranslateOp().Set(Gf.Vec3d(5.0, 5.0, 0.0))

# 设置缩放
for op in xform.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
        op.Set(Gf.Vec3f(1.0, 5.0, 1.0))
        break
else:
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 5.0, 1.0))


# ④ 后续正常循环
while kit.is_running():
    kit.update()

