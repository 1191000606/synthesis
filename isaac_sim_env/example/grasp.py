from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": "1920", "height": "1080"})

from grasp_init import init_robot, init_sensor, init_world, init_ik_solver, import_urdf, init_world_config
from grasp_utils import get_point_cloud_and_normals, generate_antipodal_grasp_poses, check_feasibility

from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

import numpy as np

world = init_world()
robot, robot_config = init_robot()
world.scene.add(robot)

gripper_sensors, arm_sensors = init_sensor()
all_sensors = gripper_sensors + arm_sensors

for sensor in all_sensors:
    world.scene.add(sensor)
    sensor.add_raw_contact_data_to_frame() # 没有这一行的话，sensor返回的数据字典是空的，缺少contacts这个键。

object_id = "0a814511b21942d297745cff34980ff8"
urdf_path = f"./data/objaverse/dataset/{object_id}/material.urdf"
object_initial_pose = (0.5, 0, 0.1, -90, 60, 180)  # x, y, z, roll, pitch, yaw
prim_path = import_urdf(urdf_path, object_initial_pose[:3], object_initial_pose[3:], (0.2, 0.2, 0.2), False)

world.reset()
world.play()

for _ in range(20):
    world.step(render=True)

usd_help = UsdHelper()
usd_help.load_stage(world.stage)

world_config = init_world_config(usd_help, object_id)
tensor_args = TensorDeviceType()
ik_solver = init_ik_solver(robot_config, world_config, tensor_args)

visual_mesh_path = f"/World/objaverse_{object_id}/baseLink_{object_id}/visuals/material_normalized/mesh"
point_cloud, normals = get_point_cloud_and_normals(visual_mesh_path, 4096)

candidate_list = generate_antipodal_grasp_poses(point_cloud, normals, 1000)

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

for i in range(grasp_joint_position.shape[0]):
    print("===============================")
    print("Pregrasp pose:", pregrasp_pose[i].tolist()) # 为了打印出来有逗号分隔
    print("Grasp pose:", grasp_pose[i].tolist())
    
    is_feasible, reason = check_feasibility(world, robot, grasp_joint_position[i], pregrasp_joint_position[i], object_id, object_initial_pose, gripper_sensors, arm_sensors)

    if is_feasible:
        print(" Successful grasp")
    else:
        print("Failed grasp:", reason)
    
    print("===============================")

simulation_app.close()
