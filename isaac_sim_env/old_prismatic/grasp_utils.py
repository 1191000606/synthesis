# Isaac Sim
from isaacsim.core.prims import XFormPrim
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import UsdGeom, Usd

# Standard Library
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

def get_point_cloud_and_normals(prim_list, point_num):
    meshes = []
    world_transforms = []
    total_area = 0.0

    for prim in prim_list:
        # 获取世界坐标系的变换矩阵
        xformable = UsdGeom.Xformable(prim)
        world_transform = np.array(xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        world_transforms.append(world_transform)

        # 获取网格数据
        mesh = UsdGeom.Mesh(prim)

        face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        if None in face_counts:
            continue
        assert (face_counts == 3).all()

        vertices = np.array(mesh.GetPointsAttr().Get())
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        meshes.append(mesh)
        total_area += mesh.area

    sampled_points_list = []
    sampled_normals_list = []

    for mesh, world_transform in zip(meshes, world_transforms):
        sample_count = int((mesh.area / total_area) * point_num)

        points, face_idx = trimesh.sample.sample_surface(mesh, count=sample_count)
        normals = mesh.face_normals[face_idx]

        # 点云转换到世界坐标系
        points_homo = np.hstack((points, np.ones((len(points), 1))))
        world_points = points_homo @ world_transform
        world_points = world_points[:, :3] / world_points[:, 3:4]

        # 法线变换
        try:
            m_3x3 = world_transform[:3, :3]
            normal_transform = np.linalg.inv(m_3x3).T
            world_normals = normals @ normal_transform

            norms = np.linalg.norm(world_normals, axis=1, keepdims=True)
            norms = np.where(norms < 1e-6, 1e-6, norms)
            world_normals = world_normals / norms
        except np.linalg.LinAlgError:
            print(f"警告: {prim.GetPath()} 变换矩阵奇异，使用原始法线。")
            world_normals = normals
            
        sampled_points_list.append(world_points)
        sampled_normals_list.append(world_normals)

    sampled_points_list = np.vstack(sampled_points_list)
    sampled_normals_list = np.vstack(sampled_normals_list)
    
    return sampled_points_list, sampled_normals_list


def generate_antipodal_grasp_poses(points, normals, num_candidates):
    rng = np.random.default_rng()
    
    max_attempts = num_candidates * 5000 # 最大循环次数防止死锁

    gripper_max_width=0.08 # 夹爪最大宽度
    gripper_min_width=0.001
    friction_cone_threshold=0.8

    pregrasp_distance=0.10, # 预抓取距离 10cm
    finger_hand_distance = 0.10  # 法兰盘和手指之间的距离，根据URDF文件为10cm

    tree = KDTree(points) # KDTree可以高效计算点云中某一点和其他点的距离
    
    results = []
    attempts = 0
    
    while len(results) < num_candidates and attempts < max_attempts:
        attempts += 1
        
        idx1 = rng.integers(0, points.shape[0])
        p1 = points[idx1]
        n1 = normals[idx1]
        
        nearby_indices = tree.query_ball_point(p1, gripper_max_width)
        
        if len(nearby_indices) == 1:
            continue
            
        if len(nearby_indices) > 10:
            candidate_idx2 = rng.choice(nearby_indices, size=10, replace=False)
        else:
            candidate_idx2 = nearby_indices
        
        for idx2 in candidate_idx2:
            if idx1 == idx2:
                continue
                
            p2 = points[idx2]
            n2 = normals[idx2]
            
            # 距离筛选
            grasp_width = np.linalg.norm(p2 - p1)
            if grasp_width < gripper_min_width or grasp_width > gripper_max_width:
                continue
                
            grasp_axis = (p2 - p1) / grasp_width
            
            # 法线对立性筛选
            if np.dot(n1, n2) > -0.5:
                continue
                
            # 摩擦锥筛选
            if np.abs(np.dot(n1, grasp_axis)) < friction_cone_threshold:
                continue
            if np.abs(np.dot(n2, grasp_axis)) < friction_cone_threshold:
                continue
            
            # 生成位姿数据
            # 计算 4x4 矩阵
            pose_mat = compute_frame_from_contact_pair(p1, p2, n1, n2)
            
            # 提取位置
            grasp_pos = pose_mat[:3, 3]
            
            # 提取旋转并转为四元数 [w, x, y, z]
            r = R.from_matrix(pose_mat[:3, :3])
            grasp_orn = r.as_quat(scalar_first=True)
            
            # 计算 Pre-grasp
            # Pre-grasp 是沿着 Z 轴 (接近方向) 后退, pose_mat[:3, 2] 是 Z 轴向量
            z_axis = pose_mat[:3, 2]
            pregrasp_pos = grasp_pos - (z_axis * pregrasp_distance)
            pregrasp_orn = grasp_orn.copy() # 姿态保持一致

            # 调整抓取和预抓取位置，考虑手指到手腕的偏移
            grasp_pos = grasp_pos - (z_axis * finger_hand_distance)
            pregrasp_pos = pregrasp_pos - (z_axis * finger_hand_distance)

            grasp_pose = (grasp_pos, grasp_orn)
            pregrasp_pose = (pregrasp_pos, pregrasp_orn)
            
            results.append((grasp_pose, pregrasp_pose))

            # 找到一个 P2 后，立即跳出内层循环，去寻找新的 P1
            # 这样可以保证采样的多样性，而不是在一个点周围生成好几个类似的抓取
            break
            
    return results

def compute_frame_from_contact_pair(p1, p2, n1, n2):
    """(保持不变) 构建旋转矩阵: Z轴接近, Y轴连线"""
    center = (p1 + p2) / 2.0
    y_axis = (p2 - p1)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 投影 n1 到垂直于 y 的平面，作为 Z 轴
    z_raw = -n1 
    z_axis = z_raw - np.dot(z_raw, y_axis) * y_axis
    
    # 防止 n1 和 y 平行导致的零向量 (虽然摩擦锥检测已经过滤了大部分)
    if np.linalg.norm(z_axis) < 1e-6:
        z_axis = np.cross(y_axis, np.array([1, 0, 0]))
        if np.linalg.norm(z_axis) < 1e-6:
            z_axis = np.cross(y_axis, np.array([0, 1, 0]))
            
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = center
    
    return pose

def is_collision_any(sensors) -> bool:
    """检查传感器列表中的任何一个是否发生了碰撞"""
    for sensor in sensors:
        data = sensor.get_current_frame()
        # get_current_frame() 返回当前物理步的接触信息
        if data["in_contact"] and data["force"] > 0.0:
            return True
    return False

def check_dual_finger_contact(gripper_sensors, target_root_path):
    """
    检查是否 两个手指 都接触到了目标。
    """
    finger_contacts = [False, False]
    touching_env = False

    for i, sensor in enumerate(gripper_sensors[:2]):
        data = sensor.get_current_frame()
        if data["in_contact"] and data["force"] > 0.0:
            for contact in data["contacts"]:
                if contact["body0"].startswith(target_root_path) or contact["body1"].startswith(target_root_path):
                    finger_contacts[i] = True
                else:
                    touching_env = True
        elif data["in_contact"] and data["force"] == 0.0:
            print("夹爪传感器检测到接触，但接触力为0")

    # 成功条件：两指都接触目标，且没有接触环境
    success = finger_contacts[0] and finger_contacts[1] and not touching_env
    return success

# 求解Collision-free IK的时候，夹爪是打开的，具体可以看https://curobo.org/tutorials/1_robot_configuration.html中Additional Configurations。然后Franka夹爪两个关节的最大张开是0.04m。
# Collision-free IK不会控制夹爪，但会将夹爪作为碰撞检测的一部分

def check_feasibility(world, franka, grasp_pose, pregrasp_pose, object_dataset, object_id, object_initial_pose, gripper_sensors, arm_sensors):
    """
    执行 Pre-grasp -> Grasp -> Close 的完整检测流水线。
    """
    original_joint_positions = franka.get_joint_positions()
    gripper_open = np.array([0.04, 0.04])
    gripper_closed = np.array([0.0, 0.0])

    all_robot_sensors = gripper_sensors + arm_sensors
    
    if object_dataset == "objaverse":
        target_object = XFormPrim(f"/World/objaverse_{object_id}")
        rigid_target_object= RigidPrim(f"/World/objaverse_{object_id}/baseLink_{object_id}")
        target_object_path = f"/World/objaverse_{object_id}"
    elif object_dataset == "partnet":
        target_object = XFormPrim(f"/World/partnet_{object_id}")
        rigid_target_object= RigidPrim(f"/World/partnet_{object_id}/base_{object_id}")
        target_object_path = f"/World/partnet_{object_id}"

    try:
        # 预抓取姿态，夹爪打开，不碰到物体与环境
        franka.set_joint_positions(np.concatenate((pregrasp_pose, gripper_open)), [i for i in range(9)])

        for _ in range(20):
            world.step(render=True)
        
        if is_collision_any(all_robot_sensors):
            return False, "预抓取姿态出现了碰撞"

        # 抓取姿态，夹爪打开，不碰到物体和环境
        franka.set_joint_positions(np.concatenate((grasp_pose, gripper_open)), [i for i in range(9)])
                
        for _ in range(20):
            world.step(render=True)

        if is_collision_any(all_robot_sensors):
            return False, "抓取姿态并且夹爪打开时，出现了碰撞"

        object_position, object_orientation = target_object.get_world_poses()
        object_position = object_position[0]
        object_orientation = object_orientation[0]

        if np.linalg.norm(object_position - np.array(object_initial_pose[:3])) > 0.001:
            return False, "抓取姿态并且夹爪打开时，物体位置发生了变化"

        # 抓取姿态，夹爪闭合，夹爪需要与目标碰撞
        franka.set_joint_positions(gripper_closed, [7, 8])
        
        for _ in range(60):
            world.step(render=True)

        if not check_dual_finger_contact(gripper_sensors, target_object_path):
            return False, "夹爪闭合后，双指没有同时接触到目标物体"

        return True, "抓取动作可行"
    finally:
        franka.set_joint_positions(original_joint_positions)
        
        rigid_target_object.set_world_poses(
            positions=np.array([object_initial_pose[:3]]), 
            orientations=euler_angles_to_quat(np.array(object_initial_pose[3:]), degrees=True, extrinsic=False).reshape(1, 4)
        )

        rigid_target_object.set_linear_velocities(velocities=np.array([[0.0, 0.0, 0.0]]))
        rigid_target_object.set_angular_velocities(velocities=np.array([[0.0, 0.0, 0.0]]))
        
        franka.set_joint_positions(original_joint_positions)

        for _ in range(20):
            world.step(render=True)
