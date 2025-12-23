from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": "1920", "height": "1080"})


# Third Party
import numpy as np
from helper import init_robot, init_scene, init_world, init_motion_gen
from isaacsim.core.utils.types import ArticulationAction

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper


def main():
    world = init_world()

    target, world_config = init_scene()
    
    usd_help = UsdHelper()
    usd_help.load_stage(world.stage)
    usd_help.add_world_to_stage(world_config, base_frame="/World")

    robot, robot_config = init_robot()

    world.scene.add(robot)
    # world.scene.add()的功能主要就是自动重置，区别于USD prim path的一套物体获取路径，以及物理视图初始化 (Physics Views)

    tensor_args = TensorDeviceType()
    motion_gen, plan_config = init_motion_gen(robot_config, world_config, tensor_args)

    world.reset()

    world.play()

    articulation_controller = robot.get_articulation_controller()

    for _ in range(20):
        world.step(render=True)

    cube_goal_position, cube_goal_orientation = target.get_world_pose()
    cube_past_position, cube_past_orientation = target.get_world_pose()
    cmd_plan = None

    while simulation_app.is_running():
        world.step(render=True)

        step_index = world.current_time_step_index

        if step_index % 50 == 0.0:
            temp = usd_help.get_obstacles_from_stage(
                only_paths=["/World"], # 仅在World路径下寻找障碍物
                reference_prim_path="/World/Franka", # 障碍物的位置都是相对于Franka的
                ignore_substring=[
                    "/World/Franka",
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ) # 获取障碍物列表
            
            obstacles = temp.get_collision_check_world() # 获取障碍物中与碰撞检测相关的信息

            motion_gen.update_world(obstacles)

        cube_current_position, cube_current_orientation = target.get_world_pose()

        robot_velocities = robot.get_joints_state().velocities
        robot_static = np.linalg.norm(robot_velocities) < 0.2

        cube_moved = np.linalg.norm(cube_current_position - cube_goal_position) > 1e-3 or \
            np.linalg.norm(cube_current_orientation - cube_goal_orientation) > 1e-3

        cube_stopped = np.linalg.norm(cube_current_position - cube_past_position) < 1e-3 and \
            np.linalg.norm(cube_current_orientation - cube_past_orientation) < 1e-3

        print(f"Robot static: {robot_static}, Cube moved: {cube_moved}, Cube stopped: {cube_stopped}")

        if robot_static and cube_moved and cube_stopped:
            ik_goal = Pose(
                position=tensor_args.to_device(cube_current_position),
                quaternion=tensor_args.to_device(cube_current_orientation),
            )

            robot_joint_state = robot.get_joints_state()

            cuRobo_joint_state = JointState(
                position=tensor_args.to_device(robot_joint_state.positions),
                velocity=tensor_args.to_device(robot_joint_state.velocities),
                acceleration=tensor_args.to_device(robot_joint_state.velocities) * 0.0,
                jerk=tensor_args.to_device(robot_joint_state.velocities) * 0.0,
                joint_names=robot.dof_names,
            )

            # motion_gen的robot_config中机器人只有7个自由度，不考虑夹爪上的两个关节。但是在避障上应该还是会考虑夹爪的
            cuRobo_joint_state = cuRobo_joint_state.get_ordered_joint_state(motion_gen.kinematics.joint_names)

            result = motion_gen.plan_single(cuRobo_joint_state.unsqueeze(0), ik_goal, plan_config)

            success = result.success.item()

            if success:
                cube_goal_position, cube_goal_orientation = cube_current_position, cube_current_orientation

                cmd_plan = result.get_interpolated_plan() # N * 7，包括位置、速度、加速度、jerk
                cmd_plan = motion_gen.get_full_js(cmd_plan) # N * 9，补全夹爪的两个关节

                cmd_idx = 0

            else:
                print("Plan did not converge to a solution: " + str(result.status))


        cube_past_position, cube_past_orientation = cube_current_position, cube_current_orientation

        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]

            articulation_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
            )

            articulation_controller.apply_action(articulation_action)

            cmd_idx += 1
            
            world.step(render=True)
            
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None

    simulation_app.close()


if __name__ == "__main__":
    main()
