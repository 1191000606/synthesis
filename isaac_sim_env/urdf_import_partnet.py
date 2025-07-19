from isaacsim import SimulationApp

# URDF import, configuration and simulation sample
kit = SimulationApp({"renderer": "RaytracedLighting", "headless": False})
import omni.kit.commands
from omni.isaac.core.articulations import Articulation
from isaacsim.core.utils.extensions import get_extension_path_from_name
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
import omni.usd
import numpy as np


# Setting up import configuration:
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.distance_scale = 1.0


# Get path to extension data:
extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
# Import URDF, prim_path contains the path the path to the usd prim in the stage.
status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    # urdf_path=extension_path + "/data/urdf/robots/carter/urdf/carter.urdf",
    # urdf_path = "/home/szwang/synthesis/data/dataset/4564/mobility.urdf",
    urdf_path ="/home/szwang/synthesis/data/objaverse/data/obj/939bce9ccaec4d5ab3404dca172d2f45/material.urdf",
    import_config=import_config,
    get_articulation_root=True,

)
# Get stage handle
stage = omni.usd.get_context().get_stage()

# Enable physics
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
# Set gravity
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
# Set solver settings
PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")

# Add ground plane
omni.kit.commands.execute(
    "AddGroundPlaneCommand",
    stage=stage,
    planePath="/groundPlane",
    axis="Z",
    size=1500.0,
    position=Gf.Vec3f(0, 0, -0.50),
    color=Gf.Vec3f(0.5),
)

# Add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(500)


omni.timeline.get_timeline_interface().play()
# perform one simulation step so physics is loaded and dynamic control works.
kit.update()


# art = Articulation(prim_path)
# art.initialize()

from isaacsim.core.prims import SingleArticulation   # 更推荐的新接口
art = SingleArticulation(prim_path)
art.initialize()

print("dof_names :", art.dof_names)            # 打印所有自由度（关节）名字



if not art.is_valid():
    print(f"{prim_path} is not an articulation")
else:
    print(f"Got articulation ({prim_path})")


art.set_joint_velocities(np.array([20.0], dtype=np.float32))

# 设置速度目标后，先跑一帧
kit.update()                           # ← 第二次 physics step，SimView 已生成

# 之后再开始你的可视化 / 打印循环
while kit.is_running():
    kit.update()
    q  = art.get_joint_positions()     # 现在不再是 None
    dq = art.get_joint_velocities()
    print(f"\rθ={q[0]: .3f}  ω={dq[0]: .3f}", end="")

    print(f"\rθ={q[0]: .3f}  ω={dq[0]: .3f}", end="")



# Shutdown and exit
omni.timeline.get_timeline_interface().stop()
kit.close()



# TODO: 处理urdf文件，使其符合isaac sim的要求，即路引用文件路径中不能有-。
# 实现对整个场景文件的读取。