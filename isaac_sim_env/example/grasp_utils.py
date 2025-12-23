# Isaac Sim
from isaacsim.core.prims import XFormPrim
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import UsdGeom, Usd
import omni.usd

# Standard Library
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

def get_point_cloud_and_normals(prim_path, num_points):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    mesh = UsdGeom.Mesh(prim)

    local_vertices = np.array(mesh.GetPointsAttr().Get())
    face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    
    face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    assert (face_counts == 3).all()

    faces = face_indices.reshape(-1, 3)
    mesh_trimesh = trimesh.Trimesh(vertices=local_vertices, faces=faces)

    local_sampled_points, mesh_face_indices = trimesh.sample.sample_surface(mesh_trimesh, count=num_points)
    local_sampled_points = np.array(local_sampled_points)

    local_normals = mesh_trimesh.face_normals[mesh_face_indices]

    # 获取从局部到世界的变换矩阵
    xformable = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    world_transform = xformable.ComputeLocalToWorldTransform(time)
    world_transform_np = np.array(world_transform)
    
    num_sampled = local_sampled_points.shape[0]
    local_points_homogeneous = np.hstack((local_sampled_points, np.ones((num_sampled, 1))))

    world_points_homogeneous = local_points_homogeneous @ world_transform_np
    
    world_points = world_points_homogeneous[:, :3] / world_points_homogeneous[:, 3:4]
    
    try:
        inv_transpose_transform = np.linalg.inv(world_transform_np).T
    except np.linalg.LinAlgError:
        print(f"警告: 变换矩阵 {prim.GetPath()} 是奇异的 (Singular)，法线可能不正确。")
        inv_transpose_transform = np.eye(4)
    
    local_norm_h = np.hstack((local_normals, np.zeros((num_points, 1)))) 
    
    world_norm_h = local_norm_h @ inv_transpose_transform
    world_norms = world_norm_h[:, :3]
    
    norms_length = np.linalg.norm(world_norms, axis=1, keepdims=True)
    
    safe_norms_length = np.where(norms_length == 0, 1e-6, norms_length)
    world_norms_normalized = world_norms / safe_norms_length

    return world_points, world_norms_normalized


def generate_antipodal_grasp_poses(
    points, 
    normals, 
    num_candidates, 
    gripper_max_width=0.10, 
    gripper_min_width=0.005, 
    friction_cone_threshold=0.8,
    pregrasp_distance=0.10, # 新增：预抓取距离 10cm
):

    rng = np.random.default_rng()

    num_points = points.shape[0]
    results = []
    
    # 1. 构建 KD-Tree
    tree = KDTree(points)
    
    # 最大循环次数防止死锁
    max_attempts = num_candidates * 50 
    attempts = 0
    
    while len(results) < num_candidates and attempts < max_attempts:
        attempts += 1
        
        # 2. 随机采样第一个接触点 P1
        idx1 = rng.integers(0, num_points)
        p1 = points[idx1]
        n1 = normals[idx1]
        
        # 3. 在 P1 附近搜索 P2
        # 注意：这里 query_ball_point 返回的是球内 **所有** 点的索引
        nearby_indices = tree.query_ball_point(p1, gripper_max_width)
        
        if len(nearby_indices) < 2:
            continue
            
        # --- 关于抽取 10 个点的解释 ---
        # 这里的逻辑不是“抽取最近的10个”，而是“随机抽取10个邻居”。
        # 目的：为了性能。如果球内有500个点，我们不需要两两遍历，
        # 只要随机测几个，如果没有符合的，说明P1这块区域几何不好，不如直接换下一个P1。
        check_limit = 10
        if len(nearby_indices) > check_limit:
            candidate_idx2 = rng.choice(nearby_indices, size=check_limit, replace=False)
        else:
            candidate_idx2 = nearby_indices
        
        for idx2 in candidate_idx2:
            if idx1 == idx2:
                continue
                
            p2 = points[idx2]
            n2 = normals[idx2]
            
            # --- 4. 物理几何筛选 ---
            
            # 4.1 距离筛选
            grasp_width = np.linalg.norm(p2 - p1)
            if grasp_width < gripper_min_width or grasp_width > gripper_max_width:
                continue
                
            grasp_axis = (p2 - p1) / grasp_width
            
            # 4.2 法线对立性筛选
            if np.dot(n1, n2) > -0.5:
                continue
                
            # 4.3 摩擦锥筛选
            if np.abs(np.dot(n1, grasp_axis)) < friction_cone_threshold:
                continue
            if np.abs(np.dot(n2, grasp_axis)) < friction_cone_threshold:
                continue
            
            # --- 5. 生成位姿数据 ---
            # 计算 4x4 矩阵
            pose_mat = compute_frame_from_contact_pair(p1, p2, n1, n2)
            
            # 提取位置
            grasp_pos = pose_mat[:3, 3]
            
            # 提取旋转并转为四元数 [w, x, y, z]
            r = R.from_matrix(pose_mat[:3, :3])
            grasp_orn = r.as_quat(scalar_first=True)
            
            # 计算 Pre-grasp
            # Pre-grasp 是沿着 Z 轴 (接近方向) 后退
            # pose_mat[:3, 2] 是 Z 轴向量
            z_axis = pose_mat[:3, 2]
            pregrasp_pos = grasp_pos - (z_axis * pregrasp_distance)
            pregrasp_orn = grasp_orn.copy() # 姿态保持一致

            # 调整抓取和预抓取位置，考虑手指到手腕的偏移
            FINGER_HAND_DISTANCE = 0.10  # panda hand和right gripper的距离，根据URDF文件为10cm
            grasp_pos = grasp_pos - (z_axis * FINGER_HAND_DISTANCE)
            pregrasp_pos = pregrasp_pos - (z_axis * FINGER_HAND_DISTANCE)

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
        # get_current_frame() 返回当前物理步的接触信息
        if sensor.get_current_frame()["in_contact"]:
            return True
    return False

def is_collision_env(sensors, target_root_path) -> bool:
    """
    检查是否与环境（除目标物体以外的任何东西）发生碰撞。
    """
    for sensor in sensors:
        data = sensor.get_current_frame()
        if data["in_contact"]:
            for contact in data["contacts"]:
                # contact["body1"] 是碰撞对象的 Prim Path
                # 如果它不包含目标物体的路径前缀，说明撞到了别的东西
                if not contact["body1"].startswith(target_root_path):
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
        if data["in_contact"]:
            for contact in data["contacts"]:
                if contact["body1"].startswith(target_root_path):
                    finger_contacts[i] = True
                else:
                    touching_env = True
    
    # 成功条件：两指都接触目标，且没有接触环境
    success = finger_contacts[0] and finger_contacts[1] and not touching_env
    return success

def get_interpolated_position(end_position, start_position, t):
    alpha = 0.5 * (1 - np.cos(t * np.pi))
    interpolated_position = (1 - alpha) * start_position + alpha * end_position
    return interpolated_position

# 求解Collision-free IK的时候，夹爪是打开的，具体可以看https://curobo.org/tutorials/1_robot_configuration.html中Additional Configurations。然后Franka夹爪两个关节的最大张开是0.04m。
# Collsion-free IK不会控制夹爪，但会将夹爪作为碰撞检测的一部分

def check_feasibility(world, franka, grasp_pose, pregrasp_pose, object_id, object_initial_pose, gripper_sensors, arm_sensors):
    """
    执行 Pre-grasp -> Grasp -> Close 的完整检测流水线。
    """
    original_joint_positions = franka.get_joint_positions()
    gripper_open = np.array([0.04, 0.04])
    gripper_closed = np.array([0.0, 0.0])

    all_robot_sensors = gripper_sensors + arm_sensors
    
    target_object = XFormPrim(f"/World/objaverse_{object_id}")
    rigid_target_object= RigidPrim(f"/World/objaverse_{object_id}/baseLink_{object_id}")
    target_object_path = f"/World/objaverse_{object_id}"

    try:
        # 预抓取姿态，夹爪打开，不碰到物体与环境
        franka.set_joint_positions(np.concatenate((pregrasp_pose, gripper_open)), [i for i in range(9)])

        for _ in range(2):
            world.step(render=True)
        
        if is_collision_any(all_robot_sensors):
            return False, "预抓取姿态出现了碰撞"

        # 抓取姿态，夹爪打开，理想情况下，不碰到物体和环境。非严格要求的话，夹爪可以轻微接触物体
        franka.set_joint_positions(np.concatenate((grasp_pose, gripper_open)), [i for i in range(9)])
                
        for _ in range(2):
            world.step(render=True)

        if is_collision_any(arm_sensors):
            return False, "抓取姿态并且夹爪打开时，机械臂出现了碰撞"

        if is_collision_env(gripper_sensors, target_object_path): 
            return False, "抓取姿态并且夹爪打开时，夹爪与环境发生了碰撞"
        
        object_position, object_orientation = target_object.get_world_poses()
        object_position = object_position[0]
        object_orientation = object_orientation[0]

        if np.linalg.norm(object_position - np.array(object_initial_pose[:3])) > 0.001:
            return False, "抓取姿态并且夹爪打开时，物体位置发生了变化"

        # 抓取姿态，夹爪闭合，夹爪需要与目标碰撞
        franka.set_joint_positions(gripper_closed, [7, 8])
        
        for _ in range(10):
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

        for _ in range(10):
            world.step(render=True)
