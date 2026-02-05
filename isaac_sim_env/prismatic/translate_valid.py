from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from grasp_init import init_robot, init_sensor, init_world, init_motion_gen, init_ik_solver, import_urdf, init_world_config, set_visuals_collision_instance
from articulated_utils import parse_topology_map, get_joint_info, compute_arc_trajectory, compute_linear_trajectory, get_robot_joint_state, ik_solver_attach_object, set_drive_parameters

from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

from isaacsim.core.prims import Articulation, GeometryPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.materials import PhysicsMaterial

import numpy as np


class Task:
    def __init__(self):
        self.world = init_world()

        self.robot, self.robot_config = init_robot()
        
        self.robot_config["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}

        self.robot_config["kinematics"]["collision_sphere_buffer"] = 0.006

        self.world.scene.add(self.robot)

        self.gripper_sensors, self.arm_sensors = init_sensor()
        self.all_sensors = self.gripper_sensors + self.arm_sensors

        for sensor in self.all_sensors:
            self.world.scene.add(sensor)
            sensor.add_raw_contact_data_to_frame()

        # 平移关节
        self.object_id = "6c04c2eac973936523c841f9d5051936"
        object_index = "8736"
        urdf_path = f"./data/partnet/dataset/{object_index}/mobility.urdf"
        object_pose = (0.7, 0, 0.38, 0, 0, 75)
        scale = (0.4, 0.4, 0.4)
        fix_base = True

        self.link_parent_joint_map = parse_topology_map(urdf_path)
        import_urdf(urdf_path, object_pose[:3], object_pose[3:], scale, fix_base)
        set_visuals_collision_instance(self.object_id)

        # 让抓取的渲染仿真更清晰一些
        self.robot.set_solver_velocity_iteration_count(4)
        self.robot.set_solver_position_iteration_count(124)
        self.world._physics_context.set_solver_type("TGS")

        # 调整夹爪的stiffness等参数，通过articulation_controller.set_gains的方式无法设置maxforce
        set_drive_parameters("/World/Franka/joints/panda_finger_joint1", 62500, 1, 500)
        set_drive_parameters("/World/Franka/joints/panda_finger_joint2", 62500, 1, 500)

        # 定义高摩擦材质
        self.friction_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/friction_material",
            static_friction=4.3,
            dynamic_friction=4.3,
            restitution=0.2,
        )
        
        # 绑定材质至机械臂夹爪
        GeometryPrim("/World/Franka/panda_leftfinger/collisions").apply_physics_materials(self.friction_material)
        GeometryPrim("/World/Franka/panda_rightfinger/collisions").apply_physics_materials(self.friction_material)

        self.world.reset()
        self.world.play()
        
        # 让机器人准备好
        for _ in range(20):
            self.world.step(render=True)
    
        self.articulation_controller = self.robot.get_articulation_controller()

        usd_help = UsdHelper()
        usd_help.load_stage(self.world.stage)

        self.world_config = init_world_config(usd_help)
        self.tensor_args = TensorDeviceType()

        self.motion_gen, self.plan_config = init_motion_gen(self.robot_config, self.world_config, self.tensor_args)
        self.ik_solver = init_ik_solver(self.robot_config, self.world_config, self.tensor_args)

        print("Initialization complete.")

    def task_plan(self):
        # 平移关节，针对partnet物体8736，位置为(0.7, 0, 0.38, 0, 0, 75)，大小为（0.4，0.4，0.4）
        pregrasp_pose = [
            0.4601010850457249,
            -0.02325492042558976,
            0.8592412552919655,
            0.2523485587625774,
            -0.6786435923620786,
            0.6808788176374697,
            -0.11030464690548723
        ]
        grasp_pose = [
            0.5094363510763157,
            -0.004024793449369388,
            0.7744106373396643,
            0.2523485587625774,
            -0.6786435923620786,
            0.6808788176374697,
            -0.11030464690548723
        ]

        self.event_list = [
            {
                "event_type": "gripper_action",
                "event_params": {"action": "open"},
            },
            {
                "event_type": "wait_robot_inertia_settle",
                "event_params": {},
            },
            {
                "event_type": "move_end_effector",
                "event_params": {
                    "target_position": np.array(pregrasp_pose[:3]),
                    "target_orientation": np.array(pregrasp_pose[3:])
                },
            },
            {
                "event_type": "wait_robot_inertia_settle",
                "event_params": {},
            },
            {
                "event_type": "move_end_effector",
                "event_params": {
                    "target_position": np.array(grasp_pose[:3]),
                    "target_orientation": np.array(grasp_pose[3:])
                },
            },
            {
                "event_type": "wait_robot_inertia_settle",
                "event_params": {},
            },
            {
                "event_type": "gripper_action",
                "event_params": {"action": "close"},
            },
            {
                "event_type": "wait_robot_inertia_settle",
                "event_params": {},
            },
            {
                "event_type": "translate_link",
                "event_params": {
                    "link_name": "link_0",
                },
            }
        ]

    def run_task(self):
        for event in self.event_list:
            # input(f"Next event: {event['event_type']}")

            if event["event_type"] == "gripper_action":
                self.gripper_action(**event["event_params"])
            elif event["event_type"] == "move_end_effector":
                self.move_end_effector(**event["event_params"])
            elif event["event_type"] == "wait_robot_inertia_settle":
                self.wait_robot_inertia_settle()
            elif event["event_type"] == "attach_grasped_object":
                self.attach_grasped_object()
            elif event["event_type"] == "print_contact_info":
                self.print_contact_info()
            elif event["event_type"] == "rotate_link":
                self.rotate_link(**event["event_params"])
            elif event["event_type"] == "translate_link":
                self.translate_link(**event["event_params"])
            elif event["event_type"] == "endless_run":
                self.endless_run()
            elif event["event_type"] == "wait":
                self.wait(**event["event_params"])

        self.world.pause()

    def wait(self, steps):
        for _ in range(steps):
            self.world.step(render=True)

    def endless_run(self):
        while True:
            self.world.step(render=True)

    def print_contact_info(self):
        for sensor in self.all_sensors:
            data = sensor.get_current_frame()
            if data["in_contact"]:
                print(f"Step {data['physics_step']} Contact force: {data['force']}, number of contacts: {data['number_of_contacts']}")
                for contact in data["contacts"]:
                    print(f"body0: {contact['body0']}, body1: {contact['body1']}")

    def gripper_action(self, action="open"):
        if action == "open":
            self.robot.apply_action(ArticulationAction(joint_positions=[0.04, 0.04], joint_indices=[7, 8]))
        elif action == "close":
            self.robot.apply_action(ArticulationAction(joint_positions=[0.0, 0.0], joint_indices=[7, 8]))
        
        for _ in range(20):
            self.world.step(render=True)

    def wait_robot_inertia_settle(self):
        for _ in range(20):
            self.world.step(render=True)

    def move_end_effector(self, target_position, target_orientation):
        ik_goal = Pose(
            position=self.tensor_args.to_device(target_position),
            quaternion=self.tensor_args.to_device(target_orientation),
        )

        curobo_joint_state = get_robot_joint_state(self.robot, self.tensor_args, self.motion_gen, velocity_zero=True)

        result = self.motion_gen.plan_single(curobo_joint_state.unsqueeze(0), ik_goal, self.plan_config)

        success = result.success.item()

        if not success:
            print("Motion generation failed.")
            exit(1)
        else:
            print("Motion generation succeeded.")

        cmd_plan = result.get_interpolated_plan() # N * 7，包括位置、速度、加速度、jerk

        for i in range(len(cmd_plan.position)):
            cmd_state = cmd_plan[i]

            articulation_action = ArticulationAction(
                joint_positions=cmd_state.position.cpu().numpy(),
                joint_velocities=None,
                joint_indices=[k for k in range(7)]
            )

            self.articulation_controller.apply_action(articulation_action)

            self.world.step(render=True)

    def translate_link(self, link_name):
        # 两种方案，任选其一
        # self.translate_link_by_seq_seed_ik(link_name)
        self.translate_link_by_receding_horizon(link_name)

    def translate_link_by_seq_seed_ik(self, link_name):
        joint_name = self.link_parent_joint_map[f"{link_name}_{self.object_id}"]
        joint_info = get_joint_info(self.object_id, joint_name)

        # 降低物体关节的刚度，提高link的摩擦力
        set_drive_parameters(f"/World/partnet_{self.object_id}/joints/{joint_name}", 0, 0.5, 0.1)
        GeometryPrim(f"/World/partnet_{self.object_id}/{link_name}_{self.object_id}/collisions").apply_physics_materials(self.friction_material)

        curobo_joint_state = get_robot_joint_state(self.robot, self.tensor_args, self.motion_gen, velocity_zero=True)
        fk_result = self.ik_solver.fk(curobo_joint_state.position)
        ee_position = fk_result.ee_position[0].detach().cpu().numpy()
        ee_orientation = fk_result.ee_quaternion[0].detach().cpu().numpy()

        trajectory_points = compute_linear_trajectory(
            joint_axis=joint_info["joint_axis"],
            ee_start_position=ee_position,
            ee_start_orientation=ee_orientation,
            translation_distance=joint_info["upper_limit"] - joint_info["current_value"]
        )

        # 取消碰撞体，避免IK规划的时候被夹爪上物体干扰
        obstacle_names = []
        for mesh in self.world_config.mesh:
            if self.object_id in mesh.name and link_name in mesh.name:
                obstacle_names.append(mesh.name)
        
        for name in obstacle_names:
            self.ik_solver.world_coll_checker.enable_obstacle(enable=False, name=name)

        seed_position = self.tensor_args.to_device(curobo_joint_state.position)

        solved_joint_positions = []

        for i, (pos, quat) in enumerate(trajectory_points):
            # 构造目标 Pose
            ik_goal = Pose(
                position=self.tensor_args.to_device(pos).unsqueeze(0),
                quaternion=self.tensor_args.to_device(quat).unsqueeze(0),
            )
            
            # 调用 IK
            result = self.ik_solver.solve_single(
                goal_pose=ik_goal,
                retract_config=seed_position.unsqueeze(1),              # 形状 (1, dof)，用于正则化，拉住解不让它乱跑
                seed_config=seed_position.unsqueeze(1).unsqueeze(1),    # 形状 (1, 1, dof)，作为优化的初始猜测值
                num_seeds=1,                                            # 只使用这1个种子，不额外生成随机种子
                return_seeds=1
            )

            if not result.success.item():
                print(f"IK failed at point {i}")
                # 简单的错误处理：复用上一个点，或者报错停止
                break
            
            # 更新 seed 为当前解，供下一次循环使用
            seed_position = result.solution[0][0]
            
            solved_joint_positions.append(seed_position.cpu().numpy())

        for joint_target in solved_joint_positions:
            # 因为ik_solver不接受频率参数，所以需要自己控制每个轨迹点执行的步数
            steps_per_waypoint = 3
            
            action = ArticulationAction(
                joint_positions=np.concatenate([joint_target, np.array([0.0, 0.0])]),
                joint_indices=[k for k in range(9)]
            )
            
            for _ in range(steps_per_waypoint):
                self.articulation_controller.apply_action(action)
                self.world.step(render=True)

    def translate_link_by_receding_horizon(self, link_name):
        joint_name = self.link_parent_joint_map[f"{link_name}_{self.object_id}"]
        joint_info = get_joint_info(self.object_id, joint_name)

        # 降低物体关节的刚度，提高link的摩擦力
        set_drive_parameters(f"/World/partnet_{self.object_id}/joints/{joint_name}", 0, 0.5, 0.1)
        GeometryPrim(f"/World/partnet_{self.object_id}/{link_name}_{self.object_id}/collisions").apply_physics_materials(self.friction_material)

        curobo_joint_state = get_robot_joint_state(self.robot, self.tensor_args, self.motion_gen, velocity_zero=True)
        fk_result = self.ik_solver.fk(curobo_joint_state.position)
        ee_position = fk_result.ee_position[0].detach().cpu().numpy()
        ee_orientation = fk_result.ee_quaternion[0].detach().cpu().numpy()

        trajectory_points = compute_linear_trajectory(
            joint_axis=joint_info["joint_axis"],
            ee_start_position=ee_position,
            ee_start_orientation=ee_orientation,
            translation_distance=joint_info["upper_limit"] - joint_info["current_value"]
        )

        # # motion_gen的attach函数内有一条语句是取消attach的物体link的碰撞体
        # object_names = []
        # for mesh in self.world_config.mesh:
        #     if self.object_id in mesh.name and link_name in mesh.name:
        #         object_names.append(mesh.name)

        # attach_success = self.motion_gen.attach_objects_to_robot(
        #     joint_state=curobo_joint_state,
        #     object_names=object_names,
        #     surface_sphere_radius=0.001,
        #     link_name="attached_object",
        #     remove_obstacles_from_world_config=False,
        # )

        # if not attach_success:
        #     assert False, "Failed to attach object to motion_gen for rotate_link"
        
        # # 由于把一部分link attach到机器人上了，而attach link与物体其他部分距离太近，所以也得取消attach
        # obstacle_name = []
        # for mesh in self.world_config.mesh:
        #     if self.object_id in mesh.name and link_name not in mesh.name:
        #         obstacle_name.append(mesh.name)

        # for name in obstacle_name:
        #     self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=name)

        # 上面的代码是先将待操作link attach到机器人上（motion gen attach函数实现中有一行是取消link的碰撞体），然后由于attach的物体link和
        # 物体其他部分距离太近，会导致碰撞（修改collision_sphere_buffer也不起作用），因此物体剩下的部分也要取消碰撞体。
        # 因此，一种更直接的方式是直接取消待操作物体link的碰撞体，不attach到motion gen上，以免进一步取消其他部分的碰撞体
        for mesh in self.world_config.mesh:
            if self.object_id in mesh.name and link_name in mesh.name:
                self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=mesh.name)

        start_value = joint_info["current_value"]

        target_object = Articulation(f"/World/partnet_{self.object_id}")
        joint_index = target_object.get_dof_index(joint_name)

        points_num = len(trajectory_points)

        point_index = 0
        finish_all = False

        while point_index < points_num - 1:
            print(f"Progress {point_index}/{points_num}")

            curobo_joint_state = get_robot_joint_state(self.robot, self.tensor_args, self.motion_gen, velocity_zero=False)

            result = None
            for window_size in [18, 12, 6, 4, 2, 1]: # 8736物体关节的平移上下限只有0.04，按0.001的步长进行采样，如果不设置大窗口，机械臂运动之后还是接近step 0
                goal_index = min(point_index + window_size, points_num - 1)
                pos, quat = trajectory_points[goal_index]
                ik_goal = Pose(
                    position=self.tensor_args.to_device(pos).unsqueeze(0),
                    quaternion=self.tensor_args.to_device(quat).unsqueeze(0),
                )

                r = self.motion_gen.plan_single(curobo_joint_state.unsqueeze(0), ik_goal, self.plan_config)

                if r.success.item():
                    result = r
                    if goal_index == points_num - 1:
                        finish_all = True
                    break

            if result is None:
                if point_index == 0:
                    print("Linear trajectory control with receding_horizon planning failed at the beginning.")
                else:
                    print(f"Linear trajectory control with receding_horizon planning stopped at index={point_index}")
                    for step_rest in range(step_index, len(cmd_plan.position)):
                        cmd_state = cmd_plan[step_rest]
                        action = ArticulationAction(
                            joint_positions=cmd_state.position.cpu().numpy(),
                            joint_velocities=None,
                            joint_indices=list(range(7)),
                        )
                        self.articulation_controller.apply_action(action)
                        self.world.step(render=True)
                    return

            cmd_plan = result.get_interpolated_plan()

            if not finish_all:
                step_num = max(1, int(len(cmd_plan.position) * 0.8))
            else:
                step_num = len(cmd_plan.position)

            for step_index in range(step_num):
                cmd_state = cmd_plan[step_index]
                action = ArticulationAction(
                    joint_positions=cmd_state.position.cpu().numpy(),
                    joint_velocities=None,
                    joint_indices=list(range(7)),
                )
                self.articulation_controller.apply_action(action)
                self.world.step(render=True)

            if finish_all:
                print("Finished all trajectory points.")
                return

            current_value = target_object.get_joint_positions()[joint_index].item()
            point_index = round((current_value - start_value) / 0.001)

        print("Finished all trajectory points.")
        return

task = Task()
task.task_plan()
task.run_task()


