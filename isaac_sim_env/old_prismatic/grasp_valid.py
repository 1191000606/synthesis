from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from grasp_init import init_robot, init_sensor, init_world, init_motion_gen, import_urdf, init_world_config, set_visuals_collision_instance

from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState

from isaacsim.core.utils.types import ArticulationAction

import numpy as np

class Task:
    def __init__(self):
        self.world = init_world()

        self.robot, robot_config = init_robot()
        
        robot_config["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}

        robot_config["kinematics"]["collision_sphere_buffer"] = 0.006

        self.world.scene.add(self.robot)

        self.gripper_sensors, self.arm_sensors = init_sensor()
        self.all_sensors = self.gripper_sensors + self.arm_sensors

        for sensor in self.all_sensors:
            self.world.scene.add(sensor)
            sensor.add_raw_contact_data_to_frame()

        object_dataset = "partnet"

        if object_dataset == "objaverse":
            self.object_id = "0a814511b21942d297745cff34980ff8"
            urdf_path = f"./data/objaverse/dataset/{self.object_id}/material.urdf"
            import_urdf(urdf_path, (0.5, 0, 0.1), (-90, 60, 180), (0.2, 0.2, 0.2), False)
        elif object_dataset == "partnet":
            self.object_id = "99e55a6a9ab18d31cc9c4c1909a0f80"
            object_index = "148"
            urdf_path = f"./data/partnet/dataset/{object_index}/mobility.urdf"
            import_urdf(urdf_path, (0.6, 0, 0.2), (0, 0, 135), (0.4, 0.4, 0.4), True)
            set_visuals_collision_instance(self.object_id)

        # 让抓取的渲染仿真更清晰一些
        self.robot.set_solver_velocity_iteration_count(4)
        self.robot.set_solver_position_iteration_count(124)
        self.world._physics_context.set_solver_type("TGS")
 
        self.world.reset()
        self.world.play()
        
        # 让机器人准备好
        for _ in range(20):
            self.world.step(render=True)

        usd_help = UsdHelper()
        usd_help.load_stage(self.world.stage)

        self.world_config = init_world_config(usd_help)
        self.tensor_args = TensorDeviceType()

        self.motion_gen, self.plan_config = init_motion_gen(robot_config, self.world_config, self.tensor_args)

        self.articulation_controller = self.robot.get_articulation_controller()

        # 调整夹爪的PID参数
        kps, kds = self.articulation_controller.get_gains()
        kps[-2:] *= 180 / np.pi
        kds[-2:] *= 180 / np.pi
        self.articulation_controller.set_gains(kps, kds)

    def task_plan(self):
        pregrasp_pose = [0.5269468797724133, -0.013517422630286505, 0.5177520006511356, -0.15078891027186506, 0.978497653035146, -0.12547612047485215, 0.06372433392601977]
        grasp_pose = [0.5432017835042898, 0.014392719890486731, 0.42311161789019386, -0.15078891027186506, 0.978497653035146, -0.12547612047485215, 0.06372433392601977]

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
                "event_type": "print_contact_info",
                "event_params": {},
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
                "event_type": "print_contact_info",
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
                "event_type": "print_contact_info",
                "event_params": {},
            },
            {
                "event_type": "attach_grasped_object",
                "event_params": {},
            }
        ]

    def run_task(self):
        for event in self.event_list:
            input(f"Next event: {event['event_type']}")

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

        self.world.pause()

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

    def attach_grasped_object(self):
        robot_joint_state = self.robot.get_joints_state()
        
        cuRobo_joint_state = JointState(
            position=self.tensor_args.to_device(robot_joint_state.positions),
            velocity=self.tensor_args.to_device(robot_joint_state.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(robot_joint_state.velocities) * 0.0,
            jerk=self.tensor_args.to_device(robot_joint_state.velocities) * 0.0,
            joint_names=self.robot.dof_names,
        )

        cuRobo_joint_state = cuRobo_joint_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        # world_config里面的物体name就是prim_path
        object_names = []
        for mesh in self.world_config.mesh:
            if self.object_id in mesh.name:
                object_names.append(mesh.name)

        success = self.motion_gen.attach_objects_to_robot(
            joint_state=cuRobo_joint_state,
            object_names=object_names,
            surface_sphere_radius=0.001,
            link_name="attached_object",
            remove_obstacles_from_world_config=False,
        )

        if success:
            print("Object attached successfully!")
        else:
            exit(1)
            print("Failed to attach object.")

    def move_end_effector(self, target_position, target_orientation):
        ik_goal = Pose(
            position=self.tensor_args.to_device(target_position),
            quaternion=self.tensor_args.to_device(target_orientation),
        )

        robot_joint_state = self.robot.get_joints_state()

        cuRobo_joint_state = JointState(
            position=self.tensor_args.to_device(robot_joint_state.positions),
            velocity=self.tensor_args.to_device(robot_joint_state.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(robot_joint_state.velocities) * 0.0,
            jerk=self.tensor_args.to_device(robot_joint_state.velocities) * 0.0,
            joint_names=self.robot.dof_names,
        )

        # 由九个关节变成七个关节
        cuRobo_joint_state = cuRobo_joint_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        result = self.motion_gen.plan_single(cuRobo_joint_state.unsqueeze(0), ik_goal, self.plan_config)

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

task = Task()
task.task_plan()
task.run_task()
