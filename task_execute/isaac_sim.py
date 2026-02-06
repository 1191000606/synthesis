# Isaac Sim
from isaacsim.core.api.world import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import XFormPrim
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.physics import ContactSensor
from isaacsim.asset.importer.urdf import _urdf
from pxr import Sdf, UsdLux, Usd, UsdGeom, UsdPhysics
import omni.kit.commands
import omni.usd

# CuRobo
from curobo.util_file import get_assets_path

# Standard Library
import os
import numpy as np
import xml.etree.ElementTree as ET

class IsaacSimVariables:
    def __init__(self, world, robot, gripper_sensors, arm_sensors, link_parent_joint_map):
        self.world = world
        self.robot = robot
        self.gripper_sensors = gripper_sensors
        self.arm_sensors = arm_sensors
        self.all_sensors = gripper_sensors + arm_sensors
        self.link_parent_joint_map = link_parent_joint_map
        self.articulation_controller = robot.get_articulation_controller()


class IsaacSimUtils:
    def init_world():
        world = World(stage_units_in_meters=1.0)

        world.scene.add_default_ground_plane()

        stage = omni.usd.get_context().get_stage()
        distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(3000)

        set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")

        return world

    def import_urdf(urdf_path, position, orientation, scale, fix_base):
        _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        cfg.merge_fixed_joints = False
        cfg.convex_decomp = True
        cfg.import_inertia_tensor = True
        cfg.fix_base = fix_base
        cfg.distance_scale = 1.0

        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        stage.SetDefaultPrim(world_prim)

        success, prim_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf_path, import_config=cfg)

        if not success:
            print(f"Error: Failed to import URDF from {urdf_path}")
            return None

        XFormPrim(
            prim_paths_expr=prim_path,
            positions=np.array(position).reshape(1, 3),
            orientations=euler_angles_to_quat(np.array(orientation), degrees=True, extrinsic=False).reshape(1, 4),
            scales=np.array(scale).reshape(1, 3),
        )

        return prim_path

    def get_franka_usd():
        urdf_path = get_assets_path() + "/robot/franka_description/franka_panda.urdf"
        dest_path = get_assets_path() + "/robot/franka_description/cuRobo_franka.usd"

        if os.path.exists(dest_path):
            return dest_path

        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = True
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = True
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.distance_scale = 1
        import_config.density = 0.0

        result, _ = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
            dest_path=dest_path,
        )

        if result:
            print("Successfully generated franka usd")
        else:
            print("Failed to generate franka usd")
            dest_path = None
        
        return dest_path

    def init_robot():
        robot_usd_path = IsaacSimUtils.get_franka_usd()
        robot_prim_path="/World/Franka"
        add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        franka = Robot(prim_path=robot_prim_path, name="franka")

        return franka

    def init_sensor():
        gripper_sensors, arm_sensors = [], []

        gripper_sensor_paths = [
            "/World/Franka/panda_leftfinger",
            "/World/Franka/panda_rightfinger",
            "/World/Franka/panda_hand"
        ]
        for sensor_path in gripper_sensor_paths:
            gripper_sensors.append(ContactSensor(
                prim_path=f"{sensor_path}/contact_sensor",
                name=f"{sensor_path.split('/')[-1]}_sensor",
                min_threshold=0
            ))

        arm_sensor_paths = [f"/World/Franka/panda_link{i}" for i in range(1, 8)]
        for sensor_path in arm_sensor_paths:
            arm_sensors.append(ContactSensor(
                prim_path=f"{sensor_path}/contact_sensor",
                name=f"{sensor_path.split('/')[-1]}_sensor",
                min_threshold=0
            ))

        return gripper_sensors, arm_sensors

    def set_visuals_collision_instance(object_id):
        stage = omni.usd.get_context().get_stage()

        for prim in stage.Traverse():
            prim_str = str(prim)

            if "World" in prim_str and object_id in prim_str and ("collisions" in prim_str or "visuals" in prim_str):
                prim.SetInstanceable(False)

    def parse_topology_map(urdf_path):
        link_parent_joint_map = {}
        
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        for joint in root.findall('joint'):
            child = joint.find('child')
            if child is not None:
                child_link = child.get('link')
                joint_name = joint.get('name')
                link_parent_joint_map[child_link] = joint_name

        return link_parent_joint_map
    
    def accurate_physics_simulation(world, robot):
        robot.set_solver_velocity_iteration_count(4)
        robot.set_solver_position_iteration_count(124)
        world._physics_context.set_solver_type("TGS")

    def increase_gripper_gains(robot):
        articulation_controller = robot.get_articulation_controller()
        kps, kds = articulation_controller.get_gains()
        kps[-2:] *= 180 / np.pi
        kds[-2:] *= 180 / np.pi
        articulation_controller.set_gains(kps, kds)

    def get_joint_info(object_id ,joint_name, target_object):
        stage = omni.usd.get_context().get_stage()
        prim_path = f"/World/partnet_{object_id}/joints/{joint_name}"
        prim = stage.GetPrimAtPath(prim_path)
        
        info = {"name": joint_name, "prim_path": prim_path}

        xform = UsdGeom.Xformable(prim)
        local_to_world = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        
        info["origin"] = local_to_world[:3, 3]

        world_rot_mat = local_to_world[:3, :3]

        # 判断关节类型并获取局部轴向
        local_axis = np.array([0, 0, 1]) # 默认为 Z
        joint_type = "fixed" # 默认类型

        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "revolute"
            # 获取定义的轴 (X, Y, or Z)
            axis_attr = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            if axis_attr == "X": 
                local_axis = np.array([1, 0, 0])
            elif axis_attr == "Y": 
                local_axis = np.array([0, 1, 0])
            
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "prismatic"
            axis_attr = UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()
            if axis_attr == "X": 
                local_axis = np.array([1, 0, 0])
            elif axis_attr == "Y": 
                local_axis = np.array([0, 1, 0])

        info["type"] = joint_type

        # 将局部轴旋转到世界坐标系
        world_axis = world_rot_mat @ local_axis
        # 归一化 (非常重要，防止缩放影响)
        info["axis"] = world_axis / np.linalg.norm(world_axis)

        # --- 3. 获取 Articulation 状态 (获取角度和限位) ---
        # 尝试从 self.target_object (Articulation类) 中获取动态数据
        # 如果是 Fixed Joint，这一步会失败或返回 None
        dof_index = target_object.get_dof_index(joint_name)
        
        # 获取限位 (Lower, Upper)
        limits = target_object.get_dof_limits()[dof_index][0]
        info["lower_limit"] = limits[0]
        info["upper_limit"] = limits[1]
        
        # 获取当前角度/位置
        current_pos = target_object.get_joint_positions()[dof_index]
        info["current_value"] = current_pos
        info["dof_index"] = dof_index

        return info

class IsaacSimCollision:
    def is_collision(sensors) -> bool:
        for sensor in sensors:
            data = sensor.get_current_frame()
            if data["in_contact"] and data["force"] > 0.0:
                return True
        return False

    def check_dual_finger_contact(gripper_sensors, target_root_path):
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

    def check_feasibility(cls, isaac_sim_vars, grasp_pose, pregrasp_pose, object_dataset, object_id, object_initial_pose):
        world = isaac_sim_vars.world
        robot = isaac_sim_vars.robot
        gripper_sensors = isaac_sim_vars.gripper_sensors
        arm_sensors = isaac_sim_vars.arm_sensors

        original_joint_positions = robot.get_joint_positions()
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
            robot.set_joint_positions(np.concatenate((pregrasp_pose, gripper_open)), [i for i in range(9)])

            for _ in range(20):
                world.step(render=False)
            
            if cls.is_collision(all_robot_sensors):
                return False, "预抓取姿态出现了碰撞"

            # 抓取姿态，夹爪打开，理想情况下，不碰到物体和环境。非严格要求的话，夹爪可以轻微接触物体
            robot.set_joint_positions(np.concatenate((grasp_pose, gripper_open)), [i for i in range(9)])
                    
            for _ in range(20):
                world.step(render=False)

            if cls.is_collision(all_robot_sensors):
                return False, "抓取姿态并且夹爪打开时，出现了碰撞"
            
            object_position, object_orientation = target_object.get_world_poses()
            object_position = object_position[0]
            object_orientation = object_orientation[0]

            if np.linalg.norm(object_position - np.array(object_initial_pose[:3])) > 0.001:
                return False, "抓取姿态并且夹爪打开时，物体位置发生了变化"

            # 抓取姿态，夹爪闭合，夹爪需要与目标碰撞
            robot.set_joint_positions(gripper_closed, [7, 8])
            
            for _ in range(60):
                world.step(render=False)

            if not cls.check_dual_finger_contact(gripper_sensors, target_object_path):
                return False, "夹爪闭合后，双指没有同时接触到目标物体"

            return True, "抓取动作可行"
        finally:
            robot.set_joint_positions(original_joint_positions)
            
            rigid_target_object.set_world_poses(
                positions=np.array([object_initial_pose[:3]]), 
                orientations=euler_angles_to_quat(np.array(object_initial_pose[3:]), degrees=True, extrinsic=False).reshape(1, 4)
            )

            rigid_target_object.set_linear_velocities(velocities=np.array([[0.0, 0.0, 0.0]]))
            rigid_target_object.set_angular_velocities(velocities=np.array([[0.0, 0.0, 0.0]]))
            
            robot.set_joint_positions(original_joint_positions)

            for _ in range(20):
                world.step(render=False)