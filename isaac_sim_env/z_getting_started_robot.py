# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# 下面两行是必须写在开头的，用于启动仿真，可选有头或者无头。
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys
import carb
import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.asset.importer.urdf import _urdf   # 注意下划线

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

# Add Franka
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")  # add robot to stage
arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")  # create an articulation object


# set the initial poses of the arm and the car so they don't collide BEFORE the simulation starts
arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())














user_urdf_path = "/home/szwang/synthesis/data/dataset/148/mobility1.urdf"   # <<< 修改这里

# ========= 新的导入函数：更稳、更易调试 =========
def import_urdf(
    urdf_path: str,
    prim_root: str = "/World/URDF_Robot",
    fix_base: bool = True,
) -> Articulation:
    """
    解析 URDF ➜ 生成中间 USD ➜ reference 到当前 Stage
    返回 Articulation 句柄；若失败则抛异常
    """
    from pathlib import Path
    urdf_path = os.path.expanduser(urdf_path)
    if not Path(urdf_path).is_file():
        raise FileNotFoundError(urdf_path)

    # ---------- 配置 ----------
    cfg = _urdf.ImportConfig()
    cfg.fix_base          = fix_base
    cfg.make_default_prim = False
    cfg.convex_decomp     = False
    cfg.self_collision    = False
    # 也可以手动指定 scale / density 等

    # 中间 USD 保存到同级目录（.usd）
    dest_path = str(Path(urdf_path).with_suffix(".usd"))

    # ---------- 一步到位：解析 + 导入 ----------
    # 返回值：(success, prim_path)
    success, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path = urdf_path,
        import_config = cfg,
        dest_path = dest_path,          # 产生的 usd 文件
    )
    if not success:
        raise RuntimeError("URDFParseAndImportFile failed")

    # ---------- 如需换父路径，用 MovePrim ----------
    if prim_root and prim_root != prim_path:
        omni.kit.commands.execute(
            "MovePrimCommand", path_from=prim_path, path_to=prim_root
        )
        prim_path = prim_root

    # ---------- 生成 Articulation 句柄 ----------
    return Articulation(prim_paths_expr=prim_path, name="user_robot")


# -------- 如果路径有效就导入 ----------
import os
if os.path.isfile(user_urdf_path):
    try:
        user_robot = import_urdf(user_urdf_path)
        user_robot.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())
        carb.log_info(f"URDF robot loaded under {user_robot.prim_path}")
    except Exception as e:
        carb.log_error(f"URDF import failed: {e}")
else:
    carb.log_warn(f"URDF path '{user_urdf_path}' not found — skipping import")



















# initialize the world
my_world.reset()

k  = np.zeros(9)
dk = np.ones(9) * 0.01

for i in range(4):
    print("running cycle: ", i)
    if i == 1 or i == 3:
        # move the arm
        arm.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])
    if i == 2:
        # reset the arm
        arm.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    for j in range(100):
        # step the simulation, both rendering and physics
        target = (k + dk * j *i /2).tolist()
        arm.set_joint_positions([target])
        my_world.step(render=True)
        # print the joint positions of the car at every physics step

simulation_app.close()