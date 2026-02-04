from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "width": "1920", "height": "1080"})

from grasp_init import init_robot, init_sensor, init_world, init_ik_solver, import_urdf, init_world_config, set_visuals_collision_instance
from grasp_utils import get_point_cloud_and_normals, generate_antipodal_grasp_poses, check_feasibility

from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

import numpy as np
import json

import omni.usd

world = init_world()
robot, robot_config = init_robot()

robot_config["kinematics"]["collision_sphere_buffer"] = 0.006

world.scene.add(robot)

gripper_sensors, arm_sensors = init_sensor()
all_sensors = gripper_sensors + arm_sensors

for sensor in all_sensors:
    world.scene.add(sensor)
    sensor.add_raw_contact_data_to_frame() # 没有这一行的话，sensor返回的数据字典是空的，缺少contacts这个键。

# object_dataset = "partnet"
# object_id = "99e55a6a9ab18d31cc9c4c1909a0f80"
# object_index = "148"
# urdf_path = f"./data/partnet/dataset/{object_index}/mobility.urdf"
# link_name = "link_0"
# object_initial_pose = (0.4, -0.3, 0.2, 0, 0, 135)
# scale = (0.4, 0.4, 0.4)
# fix_base = True

object_dataset = "partnet"
object_id = "6c04c2eac973936523c841f9d5051936"
object_index = "8736"
urdf_path = f"./data/partnet/dataset/{object_index}/mobility.urdf"
link_name = "link_0"
scale = (0.4, 0.4, 0.4)
object_initial_pose = (0.5, 0.0, 0.2, 0, 0, 135)
fix_base = True

assert scale[2] == object_initial_pose[2] * 2, "物体Z轴位置必须是高度的一半"
import_urdf(urdf_path, object_initial_pose[:3], object_initial_pose[3:], scale, fix_base)

set_visuals_collision_instance(object_id)

world.reset()
world.play()

for _ in range(20):
    world.step(render=True)

usd_help = UsdHelper()
usd_help.load_stage(world.stage)

world_config = init_world_config(usd_help)
tensor_args = TensorDeviceType()

ik_solver = init_ik_solver(robot_config, world_config, tensor_args)

stage = omni.usd.get_context().get_stage()

if object_dataset == "objaverse":
    prim_path = f"/World/objaverse_{object_id}/baseLink_{object_id}/visuals/material_normalized/mesh"
    prim = stage.GetPrimAtPath(prim_path)
    
    point_cloud, normals = get_point_cloud_and_normals([prim], 4096)
elif object_dataset == "partnet":
    prim_list = []
    for prim in stage.Traverse():
        prim_str = str(prim)

        if f"/World/partnet_{object_id}/{link_name}_{object_id}/visuals" in prim_str and "mesh" in prim_str:
            prim_list.append(prim)
    
    point_cloud, normals = get_point_cloud_and_normals(prim_list, 4096)

find_feasible_grasp = False

while True:
    candidate_list = generate_antipodal_grasp_poses(point_cloud, normals, 200)

    grasp_position = np.stack([i[0][0] for i in candidate_list])
    grasp_orientation = np.stack([i[0][1] for i in candidate_list])

    ik_goal = Pose(position=tensor_args.to_device(grasp_position), quaternion=tensor_args.to_device(grasp_orientation))
    grasp_ik_result = ik_solver.solve_batch(ik_goal)

    pregrasp_position = np.stack([i[1][0] for i in candidate_list])
    pregrasp_orientation = np.stack([i[1][1] for i in candidate_list])

    ik_goal = Pose(position=tensor_args.to_device(pregrasp_position), quaternion=tensor_args.to_device(pregrasp_orientation))
    pregrasp_ik_result = ik_solver.solve_batch(ik_goal)

    combined_success = grasp_ik_result.success & pregrasp_ik_result.success

    grasp_joint_position = grasp_ik_result.solution[combined_success].cpu().numpy()
    pregrasp_joint_position = pregrasp_ik_result.solution[combined_success].cpu().numpy()

    grasp_pose = np.concatenate([grasp_position, grasp_orientation], axis=1)[combined_success.flatten().cpu().numpy()]
    pregrasp_pose = np.concatenate([pregrasp_position, pregrasp_orientation], axis=1)[combined_success.flatten().cpu().numpy()]

    grasp_pose_estimation = []

    for i in range(grasp_joint_position.shape[0]):
        is_feasible, reason = check_feasibility(world, robot, grasp_joint_position[i], pregrasp_joint_position[i], object_dataset, object_id, object_initial_pose, gripper_sensors, arm_sensors)

        if is_feasible:
            find_feasible_grasp = True

        grasp_pose_estimation.append({
            "success": is_feasible,
            "reason": reason,
            "pregrasp_pose": pregrasp_pose[i].tolist(),
            "grasp_pose": grasp_pose[i].tolist(),
            "grasp_joint_position": grasp_joint_position[i].tolist(),
            "pregrasp_joint_position": pregrasp_joint_position[i].tolist(),
        })

    json.dump(grasp_pose_estimation, open("grasp_result.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

    print(f"运行了一次循环，IK求解共{len(grasp_pose_estimation)}个结果")

    if find_feasible_grasp:
        break

simulation_app.close()
