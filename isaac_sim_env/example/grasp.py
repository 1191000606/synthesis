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

object_dataset = "partnet"

if object_dataset == "objaverse":
    object_id = "0a814511b21942d297745cff34980ff8"
    urdf_path = f"./data/objaverse/dataset/{object_id}/material.urdf"
    object_initial_pose = (0.5, 0, 0.1, -90, 60, 180)  # x, y, z, roll, pitch, yaw，虽然这个欧拉角看起来很奇葩，但是是可以让物体稳定放在桌面上的
    scale = (0.2, 0.2, 0.2)
    fix_base = False

elif object_dataset == "partnet":
    object_id = "99e55a6a9ab18d31cc9c4c1909a0f80"
    object_index = "148"
    urdf_path = f"./data/partnet/dataset/{object_index}/mobility.urdf"
    link_name = "link_0"
    object_initial_pose = (0.6, 0, 0.2, 0, 0, 135)  # 角度设置不好物体容易倒下来，虽然world_config获取的是实时的位姿，get_point_cloud_and_normals也获取的是实时的位姿，但是物体从初始位置到倒地是需要时间的，可能20步还不够物体完全倒下来的，这时候world_config获取的位姿和get_point_cloud_and_normals获取的位姿是一致的，因为两者中间没有world.step()，但是在check_feasibility函数中会有world.step()，并且每一次测试还会把物体重置到初始位置，又会发生物体倒下的过程，并且有可能step次数不一致导致物体实时位置不一致。就算是抓取过程，也会因为多了几次step导致物体位置不一致，就是 用来初始规划器的点云、用来计算抓取位姿的点云、用来做可行性检测时候的点云、抓取时候的点云，最好是保持一致。为了简单起见，物体的初始位置应该是可以让物体保持静止的。
    scale = (0.4, 0.4, 0.4)
    fix_base = True


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
    # 为什么cuRobo可以进行避障逆运动学求解，还需要进行可行性检测
    # CuRobo与Isaac Sim的物理引擎不同，会出现cuRobo觉得没有碰撞但是Isaac Sim会有碰撞的情况，也也就导致预抓取、抓取+夹爪打开的时候有可能会碰撞，如果想要减少碰撞，那么可以设置robot_config["kinematics"]["collision_sphere_buffer"]参数，这个参数默认为0.04，大一点会让机械臂远离障碍物，但可能导致IK求解失败
    is_feasible, reason = check_feasibility(world, robot, grasp_joint_position[i], pregrasp_joint_position[i], object_dataset, object_id, object_initial_pose, gripper_sensors, arm_sensors)

    grasp_pose_estimation.append({
        "success": is_feasible,
        "reason": reason,
        "pregrasp_pose": pregrasp_pose[i].tolist(),
        "grasp_pose": grasp_pose[i].tolist(),
        "grasp_joint_position": grasp_joint_position[i].tolist(),
        "pregrasp_joint_position": pregrasp_joint_position[i].tolist(),
    })

json.dump(grasp_pose_estimation, open("grasp_result.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

simulation_app.close()


# 现在存在的两个可行性失败的情况：
# 1. 预抓取、抓取+夹爪打开 的时候出现碰撞，这个是应为cuRobo与Isaac Sim物理引擎不同导致的，可以通过增大robot_config["kinematics"]["collision_sphere_buffer"]参数来减少这种情况的发生，但是会导致IK求解失败率上升
# 2. 夹爪收紧的时候，两个夹爪传感器均接触到了物体，但是一个接触力为0，一个接触力不为0，这个可能是因为一个夹爪到顶，还有一种可能是step次数不够。调大robot_config["kinematics"]["collision_sphere_buffer"]可能会使这种情况更严重