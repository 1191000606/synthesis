# Isaac Sim
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import omni.usd

# CuRobo
from curobo.types.state import JointState
from curobo.geom.sphere_fit import SphereFitType

# Standard Library
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

def parse_topology_map(urdf_path):
    link_parent_joint_map = {}
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    for joint in root.findall('joint'):
        child = joint.find('child')
        if child is not None:
            child_link = child.get('link')
            joint_name = joint.get('name')
            # 保存映射
            link_parent_joint_map[child_link] = joint_name

    return link_parent_joint_map

def get_joint_info(object_id, joint_name):
    stage = omni.usd.get_context().get_stage()

    joint_prim = stage.GetPrimAtPath(f"/World/partnet_{object_id}/joints/{joint_name}")
    joint = UsdPhysics.RevoluteJoint(joint_prim)

    lower_degree = joint.GetLowerLimitAttr().Get()
    upper_degree = joint.GetUpperLimitAttr().Get()
    state_api = PhysxSchema.JointStateAPI.Get(joint_prim, "angular")
    current_degree = state_api.GetPositionAttr().Get()   # degrees (angular)

    # 取joint在body0坐标系的位置、朝向，body1也可以，然后转化为世界坐标系下的位置、朝向
    body0_path = str(joint.GetBody0Rel().GetTargets()[0])
    body0_prim = stage.GetPrimAtPath(body0_path)
    body0_xf = UsdGeom.Xformable(body0_prim)

    # 本地坐标系到世界坐标系
    world_transform = body0_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # 本地坐标系下的位置、朝向、变换矩阵
    joint_position_in_body0 = joint.GetLocalPos0Attr().Get()
    joint_rotation_in_body0 = joint.GetLocalRot0Attr().Get()
    joint_matrix_in_body0 = Gf.Matrix4d(Gf.Rotation(joint_rotation_in_body0), Gf.Vec3d(joint_position_in_body0))

    # 世界坐标系下的位置、朝向、变换矩阵
    joint_matrix_in_world = joint_matrix_in_body0 * world_transform 

    joint_origin_in_world = np.array(joint_matrix_in_world.ExtractTranslation())

    axis_token = joint.GetAxisAttr().Get().upper()
    if axis_token == "X":
        axis_local = Gf.Vec3d(1.0, 0.0, 0.0)
    elif axis_token == "Y":
        axis_local = Gf.Vec3d(0.0, 1.0, 0.0)
    elif axis_token == "Z":
        axis_local = Gf.Vec3d(0.0, 0.0, 1.0)

    joint_axis_in_world = np.array(joint_matrix_in_world.TransformDir(axis_local))

    axis_length = np.linalg.norm(joint_axis_in_world) + 1e-12
    joint_axis_in_world = joint_axis_in_world / axis_length

    return {
        "upper_limit": upper_degree,
        "lower_limit": lower_degree,
        "current_value": current_degree,
        "joint_origin": joint_origin_in_world,
        "joint_axis": joint_axis_in_world
    }

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

    step_degree = 3 if rotation_degrees >= 0 else -3
    degree_list = list(range(0, int(rotation_degrees), step_degree))
    
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

def get_robot_joint_state(robot, tensor_args, motion_gen, velocity_zero=False):
    robot_joint_state = robot.get_joints_state()

    ratio = 0.0 if velocity_zero else 1.0

    cuRobo_joint_state = JointState(
        position=tensor_args.to_device(robot_joint_state.positions),
        velocity=tensor_args.to_device(robot_joint_state.velocities) * ratio,
        acceleration=tensor_args.to_device(robot_joint_state.velocities) * 0.0,
        jerk=tensor_args.to_device(robot_joint_state.velocities) * 0.0,
        joint_names=robot.dof_names,
    )

    # 由九个关节变成七个关节
    cuRobo_joint_state = cuRobo_joint_state.get_ordered_joint_state(motion_gen.kinematics.joint_names)

    return cuRobo_joint_state

def soft_joint_drive(object_id, joint_name):
    stage = omni.usd.get_context().get_stage()
    joint_prim = stage.GetPrimAtPath(f"/World/partnet_{object_id}/joints/{joint_name}")
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    drive.GetStiffnessAttr().Set(0)
    drive.GetDampingAttr().Set(0.5)
    drive.GetMaxForceAttr().Set(5)

def ik_solver_attach_object(ik_solver, robot_config, world_config, tensor_args, ee_pose, object_names):
    link_name = "attached_object"
    surface_sphere_radius = 0.001

    spheres_num = robot_config["kinematics"]["extra_collision_spheres"][link_name]    
    spheres_num_per_object = spheres_num // len(object_names)

    sphere_list = []
    sphere_tensor = torch.zeros((spheres_num, 4))
    sphere_tensor[:, 3] = -10.0

    for object_name in object_names:
        obstacle = world_config.get_obstacle(object_name)

        sphere = obstacle.get_bounding_spheres(
            spheres_num_per_object,
            surface_sphere_radius,
            pre_transform_pose=ee_pose.inverse(),
            tensor_args=tensor_args,
            fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            voxelize_method="ray",
        )

        sphere_list += [s.position + [s.radius] for s in sphere]

        ik_solver.world_coll_checker.enable_obstacle(enable=False, name=object_name)

    spheres = tensor_args.to_device(torch.as_tensor(sphere_list))
    sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

    ik_solver.attach_object_to_robot(surface_sphere_radius, sphere_tensor, link_name)
