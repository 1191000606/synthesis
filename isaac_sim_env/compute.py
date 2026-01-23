# Isaac Sim
import omni.usd
from pxr import Usd, UsdGeom

# CuRobo
from curobo.types.math import Pose

# Standard Library
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

from isaac_sim import IsaacSimCollision

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

def get_grasp_pose(isaac_sim_vars ,curobo_vars, object_id, object_initial_pose, link_name):
    tensor_args = curobo_vars.tensor_args
    ik_solver = curobo_vars.ik_solver

    stage = omni.usd.get_context().get_stage()

    prim_list = []
    for prim in stage.Traverse():
        prim_str = str(prim)

        if f"/World/partnet_{object_id}/{link_name}_{object_id}/visuals" in prim_str and "mesh" in prim_str:
            prim_list.append(prim)
    
    point_cloud, normals = get_point_cloud_and_normals(prim_list, 4096)

    candidate_list = generate_antipodal_grasp_poses(point_cloud, normals, 300)

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
        is_feasible, _ = IsaacSimCollision.check_feasibility(isaac_sim_vars, grasp_joint_position[i], pregrasp_joint_position[i], "partnet", object_id, object_initial_pose)

        if is_feasible:
            return pregrasp_pose[i], grasp_pose[i]
    
    return None, None

def compute_arc_trajectory(joint_origin, joint_axis, ee_start_position, ee_start_orientation, rotation_degrees):
    # 数学公式：
    # goal_pose = joint_origin + Rotation * (start_pose - joint_origin) 
    # 首先将点（start_pose）平移到旋转中心（关节原点，joint_origin），然后进行旋转（绕关节轴 joint_axis），然后平移回去
    # 写成矩阵形式就是（矩阵计算从左到右，矩阵理解含义是从右到左）：
    # P_goal = T_from_origin * R_axis * T_to_origin * P_start
    
    # 起始位置
    P_start = np.eye(4)
    P_start[:3, :3] = R.from_quat(ee_start_orientation, scalar_first=True).as_matrix()
    P_start[:3, 3] = ee_start_position

    # 平移到原点
    T_to_origin = np.eye(4)
    T_to_origin[:3, 3] = -joint_origin
    
    # 从原点平移回去
    T_from_origin = np.eye(4)
    T_from_origin[:3, 3] = joint_origin

    trajectory_points = []

    degree_list = list(range(0, int(rotation_degrees), 3))
    if degree_list[-1] != rotation_degrees:
        degree_list.append(rotation_degrees)
    
    for angle_degree in degree_list:

        angle_radian = np.radians(angle_degree)
        
        # 这里的 joint_axis 必须是归一化的
        rotation_vector = joint_axis * angle_radian
        rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
        
        R_step = np.eye(4)
        R_step[:3, :3] = rotation_matrix
        
        P_goal = T_from_origin @ R_step @ T_to_origin @ P_start

        trajectory_point = (P_goal[:3, 3], R.from_matrix(P_goal[:3, :3]).as_quat(scalar_first=True))

        trajectory_points.append(trajectory_point)
        
    return trajectory_points
