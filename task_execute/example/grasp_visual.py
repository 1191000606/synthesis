import json
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from grasp_init import init_robot, init_world, import_urdf, set_visuals_collision_instance
from grasp_utils import get_point_cloud_and_normals

import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np

import omni.usd

world = init_world()
robot, robot_config = init_robot()
world.scene.add(robot)

object_dataset = "partnet"

if object_dataset == "objaverse":
    object_id = "0a814511b21942d297745cff34980ff8"
    urdf_path = f"./data/objaverse/dataset/{object_id}/material.urdf"
    object_initial_pose = (0.5, 0, 0.1, -90, 60, 180)
    scale = (0.2, 0.2, 0.2)
    fix_base = False
elif object_dataset == "partnet":
    object_id = "99e55a6a9ab18d31cc9c4c1909a0f80"
    object_index = "148"
    urdf_path = f"./data/partnet/dataset/{object_index}/mobility.urdf"
    link_name = "link_0"
    object_initial_pose = (0.6, 0, 0.2, 0, 0, 135) 
    scale = (0.4, 0.4, 0.4)
    fix_base = True

import_urdf(urdf_path, object_initial_pose[:3], object_initial_pose[3:], scale, fix_base)
set_visuals_collision_instance(object_id)

world.reset()
world.play()

for _ in range(20):
    world.step(render=True)

stage = omni.usd.get_context().get_stage()
if object_dataset == "objaverse":
    object_prim_path = f"/World/objaverse_{object_id}/baseLink_{object_id}/visuals/material_normalized/mesh"
    object_prim = stage.GetPrimAtPath(object_prim_path)
    
    object_point_cloud, _ = get_point_cloud_and_normals([object_prim], 32768)
elif object_dataset == "partnet":
    object_prim_list = []
    for prim in stage.Traverse():
        prim_str = str(prim)

        if f"/World/partnet_{object_id}" in prim_str and "visuals" in prim_str and "mesh" in prim_str:
            object_prim_list.append(prim)
    
    object_point_cloud, _ = get_point_cloud_and_normals(object_prim_list, 32768)


gripper_point_cloud_list = []
gripper_prim_path_list = [
    "/World/Franka/panda_hand/visuals",
    "/World/Franka/panda_leftfinger/visuals",
    "/World/Franka/panda_rightfinger/visuals",
]
for prim_path in gripper_prim_path_list:
    prim = stage.GetPrimAtPath(prim_path)
    prim.SetInstanceable(False)

gripper_prim_path_list = [
    "/World/Franka/panda_hand/visuals/hand/mesh",
    "/World/Franka/panda_leftfinger/visuals/finger/mesh",
    "/World/Franka/panda_rightfinger/visuals/finger/mesh",
]
gripper_prim_list = [stage.GetPrimAtPath(prim_path) for prim_path in gripper_prim_path_list]

grasp_result = json.load(open("./grasp_result.json", "r", encoding="utf-8"))
for grasp_config in grasp_result:
    if grasp_config["success"]:
        robot.set_joint_positions(np.array(grasp_config["grasp_joint_position"] + [0.04, 0.04]), list(range(9)))
        world.step(render=True)
        gripper_point_cloud, _ = get_point_cloud_and_normals(gripper_prim_list, 8192)
        gripper_point_cloud_list.append(gripper_point_cloud)


def visualize_grasps_with_index(object_point_cloud, gripper_point_cloud_list):
    # 初始化可视化窗口(GUI API)，使用O3DVisualizer以支持3D文本标签
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    
    vis = o3d.visualization.O3DVisualizer("Grasp Visualization", 1280, 960)
    vis.show_settings = True
    vis.scene.set_background([0.0, 0.0, 0.0, 1.0])
    vis.scene.set_lighting(rendering.Open3DScene.LightingProfile.DARK_SHADOWS, (0.577, -0.577, -0.577))

    # 创建点云对象并设置颜色，添加到可视化场景
    pcd_object = o3d.geometry.PointCloud()
    pcd_object.points = o3d.utility.Vector3dVector(object_point_cloud)
    pcd_object.paint_uniform_color([0.7, 0.7, 0.7]) 
    vis.add_geometry("Object Point Cloud", pcd_object)
    
    for i, gripper_point_cloud in enumerate(gripper_point_cloud_list):
        pcd_gripper = o3d.geometry.PointCloud()
        pcd_gripper.points = o3d.utility.Vector3dVector(gripper_point_cloud)
        pcd_gripper.paint_uniform_color([1.0, 0.2, 0.2])
        vis.add_geometry(f"Gripper Point Cloud {i}", pcd_gripper)

    # 重置视角以包含场景中所有元素
    vis.reset_camera_to_default()

    # 运行可视化应用
    app.add_window(vis)
    app.run()

visualize_grasps_with_index(object_point_cloud, gripper_point_cloud_list)