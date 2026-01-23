from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaac_sim import IsaacSimUtils, IsaacSimVariables
from cuRobo import CuRoboUtils, CuRoboVariables

import numpy as np


class Task:
    def __init__(self):
        # Isaac Sim Initialization
        world = IsaacSimUtils.init_world()
        robot = IsaacSimUtils.init_robot()
        world.scene.add(robot)

        gripper_sensors, arm_sensors = IsaacSimUtils.init_sensor()
        all_sensors = gripper_sensors + arm_sensors

        for sensor in all_sensors:
            world.scene.add(sensor)
            sensor.add_raw_contact_data_to_frame()

        object_id = "99e55a6a9ab18d31cc9c4c1909a0f80"
        urdf_path = f"./data/partnet/dataset/148/mobility.urdf"
        object_initial_pose = (0.6, 0, 0.2, 0, 0, 135)
        link_name = "link_0"
        IsaacSimUtils.import_urdf(urdf_path, object_initial_pose[:3], object_initial_pose[3:], (0.4, 0.4, 0.4), True)

        IsaacSimUtils.set_visuals_collision_instance(object_id)

        link_parent_joint_map = IsaacSimUtils.parse_topology_map(urdf_path)

        IsaacSimUtils.accurate_physics_simulation(world, robot)

        world.reset()
        world.play()
        
        for _ in range(20):
            world.step(render=True)

        IsaacSimUtils.increase_gripper_gains(robot)

        self.issac_sim_vars = IsaacSimVariables(world, robot, gripper_sensors, arm_sensors, link_parent_joint_map)

        # CuRobo Initialization
        robot_config = CuRoboUtils.init_robot_config()
        world_config = CuRoboUtils.init_world_config(world)
        tensor_args = CuRoboUtils.init_tensor_args()

        ik_solver = CuRoboUtils.init_ik_solver(robot_config, world_config, tensor_args)
        motion_gen, plan_config = CuRoboUtils.init_motion_gen(robot_config, world_config, tensor_args)
        
        self.curobo_vars = CuRoboVariables(robot_config, world_config, tensor_args, ik_solver, motion_gen, plan_config)



    def task_plan(self):
        pregrasp_pose = [0.5020130398866539, 0.0027723793904265714, 0.3702974717255742, 0.30881144159915797, 0.9261992289167138, 0.1742311405257189, -0.12819513080707065]
        grasp_pose = [0.48902710756121925, -0.05889892119760516, 0.29265717133061264, 0.30881144159915797, 0.9261992289167138, 0.1742311405257189, -0.12819513080707065]

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
                "event_type": "attach_grasped_object",
                "event_params": {
                    "object_id": "99e55a6a9ab18d31cc9c4c1909a0f80",
                    "link_name": "link_0"
                },
            },
            {
                "event_type": "move_end_effector",
                "event_params": {
                    "target_position": np.array(grasp_pose[:3])+ np.array([0.0, 0.0, 0.5]),
                    "target_orientation": np.array(grasp_pose[3:])
                }
            }
        ]

    def run_task(self):
        for event in self.event_list:
            if event["event_type"] == "gripper_action":
                self.gripper_action(**event["event_params"])
            elif event["event_type"] == "move_end_effector":
                self.move_end_effector(**event["event_params"])
            elif event["event_type"] == "wait_robot_inertia_settle":
                self.wait_robot_inertia_settle()
            elif event["event_type"] == "attach_grasped_object":
                self.attach_grasped_object()

        self.world.pause()

task = Task()
task.task_plan()
task.run_task()
