from isaacsim import SimulationApp
# simulation app参数可见/home/chenyifan/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py
simulation_app = SimulationApp({"headless": False})

# debug下可以看这些包中的代码，例如：
# /home/chenyifan/isaacsim/exts/isaacsim.core.utils
# /home/chenyifan/isaacsim/exts/isaacsim.core.prims
import omni.kit.commands
import omni.usd
from pxr import Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators.examples.franka import Franka

import numpy as np
from scipy.spatial.transform import Rotation


def import_urdf(urdf_path, position, orientation, scale, fix_base):
    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.convex_decomp = True
    cfg.import_inertia_tensor = True
    cfg.fix_base = fix_base
    cfg.distance_scale = 1.0

    # 这样物体才能导入到 /World 下，而不是上一个物体的子节点下
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)

    # 有一个参数get_articulation_root会获取到partnet_{uid}/base_{uid}，这样会导致后面xform变化的时候，作用在base上，而不是整个物体上
    success, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=cfg
    )

    if not success:
        print(f"Error: Failed to import URDF from {urdf_path}")
        return None

    XFormPrim(
        prim_paths_expr=prim_path,  # 使用正则表达式匹配所有符合条件的prim
        positions=np.array(position).reshape(1, 3),  # 有一个需要区分的参数是transition，transition也是平移，但是transition是相对于父节点的，而position是相对于世界坐标系的
        orientations=Rotation.from_euler("XYZ", orientation, degrees=True).as_quat(scalar_first=True).reshape(1, 4),  # 四元数表示的旋转，scalar_first=True参数是得到wxyz顺序的四元数，注意from_euler中“XYZ”大写、小写还有区别
        # orientations=euler_angles_to_quat(np.array([30, 45, 60]), degrees=True, extrinsic=False).reshape(1, 4) 效果相同
        scales=np.array(scale).reshape(1, 3),
    )


# 创建 World 实例
world = World(stage_units_in_meters=1.0)

# 重置世界
world.reset()

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(3000)

# 设置相机视角
set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")

# 导入 Franka 机械臂
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
XFormPrim(
    prim_paths_expr=franka.prim_path,
    positions=np.array([2, 0, 0]).reshape(1, 3),
    orientations=Rotation.from_euler("XYZ", [(0, 0, 0)], degrees=True).as_quat(scalar_first=True).reshape(1, 4),
    scales=np.array([1, 1, 1]).reshape(1, 3),
)

# 导入 objaverse 物体
import_urdf("/home/chenyifan/Projects/synthesis/data/objaverse/dataset/939bce9ccaec4d5ab3404dca172d2f45/material.urdf", (-5, -2.5, 0), (-90, 60, 180), (2, 2, 2), False)
import_urdf("/home/chenyifan/Projects/synthesis/data/objaverse/dataset/197785a17620486e9f5d8e5d07ad88f9/material.urdf", (-2, -2, 0), (-90, 60, 180), (2, 2, 2), False)
import_urdf("/home/chenyifan/Projects/synthesis/data/objaverse/dataset/83492c3bf9b5496aa16306098b188a41/material.urdf", (-5, 2, 0), (-90, 60, 180), (2, 2, 2), False)

# 注意：对于物体部件消失的问题，可以尝试修改 obj文件 对应的 mtl 文件中的 d 参数为 1，表示不透明
# 导入 PartNet 物体
import_urdf("/home/chenyifan/Projects/synthesis/data/partnet/dataset/4564/mobility.urdf", [2, 3.6, 2.1], [0, 0, 45], [1.0, 1.0, 1.0], False)


# 运行仿真主循环
world.play()  # 启动物理引擎

while simulation_app.is_running():
    world.step(render=True)

# 关闭应用程序
simulation_app.close()
