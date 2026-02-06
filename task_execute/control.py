from cuRobo import CuRoboUtils

from isaacsim.core.utils.types import ArticulationAction
from curobo.types.math import Pose
from curobo.types.state import JointState

import numpy as np



class API:
    def __init__(self, isaac_sim_vars, curobo_vars):
        self.issac_sim_vars = isaac_sim_vars
        self.curobo_vars = curobo_vars

    def gripper_action(self, action):
        robot = self.issac_sim_vars.robot
        world = self.issac_sim_vars.world

        if action == "open":
            robot.apply_action(ArticulationAction(joint_positions=[0.04, 0.04], joint_indices=[7, 8]))
        elif action == "close":
            robot.apply_action(ArticulationAction(joint_positions=[0.0, 0.0], joint_indices=[7, 8]))
        
        world.step(render=True)

    def wait_robot_inertia_settle(self):
        world = self.issac_sim_vars.world
        
        for _ in range(10):
            world.step(render=True)

    def attach_grasped_object(self, object_id, link_name):
        robot = self.issac_sim_vars.robot
        world_config = self.curobo_vars.world_config
        tensor_args = self.curobo_vars.tensor_args
        motion_gen = self.curobo_vars.motion_gen

        curobo_joint_state = CuRoboUtils.get_robot_joint_state(robot, tensor_args, motion_gen, velocity_zero=True)
                
        object_names = []
        for mesh in world_config.mesh:
            if object_id in mesh.name and link_name in mesh.name:
                object_names.append(mesh.name)
        
        success = motion_gen.attach_objects_to_robot(
            joint_state=curobo_joint_state,
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
        robot = self.issac_sim_vars.robot
        world = self.issac_sim_vars.world
        articulation_controller = self.issac_sim_vars.articulation_controller
        tensor_args = self.curobo_vars.tensor_args
        motion_gen = self.curobo_vars.motion_gen
        plan_config = self.curobo_vars.plan_config

        ik_goal = Pose(
            position=tensor_args.to_device(target_position),
            quaternion=tensor_args.to_device(target_orientation),
        )

        curobo_joint_state = CuRoboUtils.get_robot_joint_state(robot, tensor_args, motion_gen, velocity_zero=True)

        result = motion_gen.plan_single(curobo_joint_state.unsqueeze(0), ik_goal, plan_config)

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

            articulation_controller.apply_action(articulation_action)

            world.step(render=True)

    def rotate_link_seed_IK(self, trajectory_points):
        robot = self.issac_sim_vars.robot
        articulation_controller = self.issac_sim_vars.articulation_controller
        world = self.issac_sim_vars.world
        tensor_args = self.curobo_vars.tensor_args
        motion_gen = self.curobo_vars.motion_gen

        cuRobo_joint_state = CuRoboUtils.get_robot_joint_state(robot, tensor_args, motion_gen, velocity_zero=True)

        seed_positions = tensor_args.to_device(cuRobo_joint_state.positions)

        solved_joint_positions = []

        for i, (pos, quat) in enumerate(trajectory_points):
            # 构造目标 Pose
            ik_goal = Pose(
                position=self.tensor_args.to_device(pos).unsqueeze(0),
                quaternion=self.tensor_args.to_device(quat).unsqueeze(0),
            )
            
            seed_state = JointState(
                position=seed_positions,
                joint_names=motion_gen.kinematics.joint_names
            )

            result = self.ik_solver.solve_single(ik_goal, seed_state, retries=0) 
            
            if not result.success.item():
                print(f"IK failed at point {i}")
                # 简单的错误处理：复用上一个点，或者报错停止
                break
            
            # 更新 seed 为当前解，供下一次循环使用
            seed_positions = result.solution 
            
            solved_joint_positions.append(seed_positions.cpu().numpy().flatten())

        for joint_target in solved_joint_positions:
            # 你可以控制这里的步数来控制速度
            # 比如每个点执行 3-5 个 simulation step，这取决于你采样的密度
            steps_per_waypoint = 3 
            
            action = ArticulationAction(
                joint_positions=joint_target,
                joint_indices=[k for k in range(7)] 
            )
            
            for _ in range(steps_per_waypoint):
                articulation_controller.apply_action(action)
                world.step(render=True)

    def rotate_link_receding_horizon(self, trajectory_points):
        robot = self.issac_sim_vars.robot
        articulation_controller = self.issac_sim_vars.articulation_controller
        world = self.issac_sim_vars.world
        tensor_args = self.curobo_vars.tensor_args
        motion_gen = self.curobo_vars.motion_gen
        plan_config = self.curobo_vars.plan_config

        points_num = len(trajectory_points)

        i = 0

        while i < points_num - 1:
            cuRobo_joint_state = CuRoboUtils.get_robot_joint_state(robot, tensor_args, motion_gen, velocity_zero=False)

            result = None
            for K in [4, 2, 1]:
                goal_i = min(i + K, points_num - 1)
                pos, quat = trajectory_points[goal_i]
                ik_goal = Pose(
                    position=tensor_args.to_device(pos).unsqueeze(0),
                    quaternion=tensor_args.to_device(quat).unsqueeze(0),
                )

                r = motion_gen.plan_single(cuRobo_joint_state.unsqueeze(0), ik_goal, plan_config)

                if r.success.item():
                    result = r
                    break

            if result is None:
                print(f"Arc trajectory control with receding_horizon planning failed at i={i}")
                return False

            cmd_plan = result.get_interpolated_plan()

            step_num = max(1, int(len(cmd_plan.position) * 0.5))

            for t in range(step_num):
                cmd_state = cmd_plan[t]
                action = ArticulationAction(
                    joint_positions=cmd_state.position.cpu().numpy(),
                    joint_velocities=None,
                    joint_indices=list(range(7)),
                )
                articulation_controller.apply_action(action)
                world.step(render=True)

            theta_now = float(self.target_object.get_joint_positions()[self.joint_dof_index])
            new_index =  int((theta_now - self.start_radian) / np.pi * 180.0 / 3.0)

            i = new_index
