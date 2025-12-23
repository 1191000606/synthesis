# /home/chenyifan/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka文件夹里面config.json、config_cortex.json、config_no_feedback.json三个文件中的end_effector_frame_name，由“right_gripper”改成“panda_hand”
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from task_utils import (
    init_world,
    init_robot,
    init_controller,
    get_interpolated_position, 
    get_interval, init_slerp, 
    quaternion_angular_distance, 
    get_interpolated_orientation
)

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat

import numpy as np


class Task:
    def __init__(self):
        self.world = init_world()
        self.franka, self.end_effector = init_robot()
        self.world.scene.add(self.franka)
        self.controller = init_controller(self.franka)
        
        self.init_scene()
        self.init_task_config()

        self.world.reset()
        self.controller.reset()

        self.world.add_physics_callback("sim_step", callback_fn=self.physics_step)

    def init_scene(self):
        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube", 
                position=np.array([0.3, 0.3, 0.3]), 
                color=np.array([0.0, 0.0, 1.0]), 
                size=0.05
            )
        )

    def init_task_config(self):
        self.event_index = 0
        self.t = 0.0
        self.start_position = None
        self.start_orientation = None
        self.end_position = None
        self.end_orientation = None
        self.slerp = None
        self.t_increment = None

    def task_plan(self):
        cube_current_position, _ = self.cube.get_world_pose()
        cube_goal_position = np.array([0.6, 0.0, 0.025])

        self.event_list = [
            {
                "event_type": "gripper_action",
                "event_params": {"action": "open"},
            },
            {
                "event_type": "move_end_effector",
                "event_params": {
                    "target_position": cube_current_position + np.array([0.0, 0.0, 0.25]),
                    "target_orientation": euler_angles_to_quat([0.0, np.pi, 0.0])
                },
            },
            {
                "event_type": "move_end_effector",
                "event_params": {
                    "target_position": cube_current_position + np.array([0.0, 0.0, 0.1]),  # hand本身有一定高度
                    "target_orientation": None
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
                "event_type": "move_end_effector",
                "event_params": {
                    "target_position": cube_current_position + np.array([0.0, 0.0, 0.25]),
                    "target_orientation": None
                }
            },
            {
                "event_type": "move_end_effector",
                "event_params": {
                    "target_position": cube_goal_position + np.array([0.0, 0.0, 0.1]), # hand本身有一定高度
                    "target_orientation": np.array([0.0, 0.8, 0.6, 0.0])
                },
            },
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
                    "target_position": cube_current_position + np.array([0.0, 0.0, 0.2]),
                    "target_orientation": None
                },
            },
            {
                "event_type": "move_end_effector_default",
                "event_params": {},
            }
        ]

    def run_task(self):
        self.world.play()
        while simulation_app.is_running():
            self.world.step(render=True)
            simulation_app.update()

    def event_refresh(self):
        self.t = 0.0
        self.event_index += 1
        self.start_position = None
        self.start_orientation = None
        self.end_position = None
        self.end_orientation = None
        self.slerp = None
        self.t_increment = None
        # input(f"Press Enter to continue to event {self.event_index}...")

    def gripper_action(self, action="open"):
        if action == "open":
            self.franka.apply_action(ArticulationAction(joint_positions=[None] * 7 + [0.04, 0.04]))
        elif action == "close":
            self.franka.apply_action(ArticulationAction(joint_positions=[None] * 7 + [0.0, 0.0]))
        self.t += 0.1

    def wait_robot_inertia_settle(self):
        self.franka.apply_action(ArticulationAction(joint_positions=[None] * 9))
        self.t += 0.1

    def move_end_effector_default(self):
        self.move_end_effector(np.array([0.3, 0.0, 0.3]), euler_angles_to_quat([0.0, np.pi, 0.0]))

    def move_end_effector(self, target_position, target_orientation=None):
        if self.t == 0.0:
            self.start_position, self.start_orientation = self.end_effector.get_world_poses()
            self.start_position, self.start_orientation = self.start_position[0], self.start_orientation[0]
            self.end_position, self.end_orientation = target_position, target_orientation

            if self.end_orientation is None:
                self.end_orientation = self.start_orientation

            if quaternion_angular_distance(self.start_orientation, self.end_orientation) > 5:
                print(f"event = {self.event_index}: using slerp for orientation interpolation")
                self.slerp = init_slerp(self.end_orientation, self.start_orientation)

            self.t_increment = get_interval(self.end_position, self.start_position, velocity=0.001)

        interpolated_position = get_interpolated_position(self.end_position, self.start_position, self.t)
        if self.slerp is not None:
            interpolated_orientation = get_interpolated_orientation(self.slerp, self.t)
            print(f"event = {self.event_index}: using slerp for orientation interpolation")
        else:
            interpolated_orientation = self.end_orientation

        robot_target_joint_positions = self.controller.forward(
            target_end_effector_position=interpolated_position,
            target_end_effector_orientation=interpolated_orientation,
        )

        self.franka.apply_action(robot_target_joint_positions)

        self.t += self.t_increment

    def physics_step(self, step_size):
        if self.t == 0.0:  # 每个event完成之后，都应该重新去做task_plan，最早的时候规划一次，这样里面的物体位置会得不到更新
            self.task_plan()

        event = self.event_list[self.event_index]
        event_type = event["event_type"]
        event_params = event["event_params"]

        print(f"event = {self.event_index}, event_type = {event_type}, t = {self.t:.3f}")

        if event_type == "gripper_action":
            self.gripper_action(**event_params)
        elif event_type == "wait_robot_inertia_settle":
            self.wait_robot_inertia_settle()
        elif event_type == "move_end_effector":
            self.move_end_effector(**event_params)
        elif event_type == "move_end_effector_default":
            self.move_end_effector_default()
        else:
            raise ValueError(f"Unknown event type: {event_type}")

        if self.t >= 1.0:
            self.event_refresh()
        
        if self.event_index >= len(self.event_list):
            print("Task completed!")
            self.world.pause()

task = Task()
task.run_task()

# 如果环境在开始的时候是不稳态的，比如物体在下落，可能需要10个step才能落地