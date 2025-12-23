from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from grasp_init import init_robot, init_sensor, init_world, init_motion_gen, import_urdf, init_world_config

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

        robot_config["kinematics"]["extra_collision_spheres"] = {"attached_object": 100} # 提前给被抓取物体留出碰撞球GPU空间

        self.world.scene.add(self.robot)

        self.gripper_sensors, self.arm_sensors = init_sensor()
        self.all_sensors = self.gripper_sensors + self.arm_sensors

        for sensor in self.all_sensors:
            self.world.scene.add(sensor)
            sensor.add_raw_contact_data_to_frame()

        object_id = "0a814511b21942d297745cff34980ff8"
        urdf_path = f"./data/objaverse/dataset/{object_id}/material.urdf"
        import_urdf(urdf_path, (0.5, 0, 0.1), (-90, 60, 180), (0.2, 0.2, 0.2), False)

        # 让抓取的渲染仿真更清晰一些
        self.robot.set_solver_velocity_iteration_count(4) # 在抓取物体并移动时，摩擦力是防止物体滑落的关键。更高的速度迭代能让摩擦力的模拟更加稳定，减少物体在指尖“打滑”的现象。默认为1。
        self.robot.set_solver_position_iteration_count(124) # 位置求解器，防止夹爪穿透物体。默认为32。
        self.world._physics_context.set_solver_type("TGS") # TGS 是 PhysX 4/5 引入的高级求解器。它在处理质量比差异大（例如沉重的机械臂抓取轻小的物体）时更稳定，收敛速度更快，且能更好地处理关节的刚性。PGS 是老一代默认算法，虽然速度快，但在处理高刚度关节（High Stiffness）和复杂约束时容易产生抖动或发散。对于机器人仿真，TGS 几乎是必选项，因为它能显著减少机械臂在运动过程中的“颤抖”现象。TGS就是默认选项
 
        self.world.reset()
        self.world.play()
        
        # 让机器人准备好
        for _ in range(20):
            self.world.step(render=True)

        usd_help = UsdHelper()
        usd_help.load_stage(self.world.stage)

        self.world_config = init_world_config(usd_help, object_id)
        self.tensor_args = TensorDeviceType()
        self.motion_gen, self.plan_config = init_motion_gen(robot_config, self.world_config, self.tensor_args)

        self.articulation_controller = self.robot.get_articulation_controller()

        # 调整夹爪的PID参数
        kps, kds = self.articulation_controller.get_gains()
        kps[-2:] *= 180 / np.pi
        kds[-2:] *= 180 / np.pi
        self.articulation_controller.set_gains(kps, kds)

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
                "event_params": {},
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


    def gripper_action(self, action="open"):
        if action == "open":
            self.robot.apply_action(ArticulationAction(joint_positions=[0.04, 0.04], joint_indices=[7, 8]))
        elif action == "close":
            self.robot.apply_action(ArticulationAction(joint_positions=[0.0, 0.0], joint_indices=[7, 8]))
        
        self.world.step(render=True)

    def wait_robot_inertia_settle(self):
        for _ in range(10):
            self.world.step(render=True)

    def attach_grasped_object(self):
        robot_js = self.robot.get_joints_state() # position还好，velocity由于机器人已经静止，所以非常接近0，efforts是None，没有其他属性
        
        cu_js = JointState(
            position=self.tensor_args.to_device(robot_js.positions),
            velocity=self.tensor_args.to_device(robot_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(robot_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(robot_js.velocities) * 0.0,
            joint_names=self.robot.dof_names,
        )

        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        # 这里的name是world_config里定义的物体名字，与isaac sim场景里的名字无关
        object_name = self.world_config.mesh[0].name
        
        success = self.motion_gen.attach_objects_to_robot(
            joint_state=cu_js,
            object_names=[object_name],
            surface_sphere_radius=0.001,
            link_name="attached_object", # 机械臂上预留的碰撞球组名字，URDF文件（/home/chenyifan/Projects/curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf）里面找不到，这里不能写成"panda_hand"，因为panda_hand有visual和collision的mesh，而这里的link不能有mesh。这个link在urdf文件里面没有，但是在/home/chenyifan/Projects/curobo/src/curobo/content/configs/robot/franka.yml里面有。上面robot_config["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}这里也是用的这个名字。
            remove_obstacles_from_world_config=False, # 无论这里True还是False，这个物体都不会成为环境碰撞检测中的一个障碍了。这个参数的作用是是否把物体从cuRobo world中删掉。如果删掉了，等机械臂放下物体之后，那么物体就要重新添加到cuRobo world中。如果物体没从cuRobo world中删掉，那么只需要后续调用一下detach_objects_from_robot，物体就会重新成为环境碰撞检测中的一个障碍。
        )

        if success:
            print("Object attached successfully!")
        else:
            exit(1)
            print("Failed to attach object.")
        
        # 单纯的想取消碰撞，可以用下面这行代码，不过机械臂抓着物体移动的时候，被抓住的物体应该也要和环境避障，下面这行代码没考虑到这个。
        # self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=self.world_config.mesh[0].name)

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
            # isaac sim是60Hz，也就是一次0.0166秒，而cuRobo motion_gen_config是0.01秒
            # 理论上一个isaac_sim step内是可以完成一次cmd_plan的执行的，不过这里为了保险起见，做了两次step
            # 还需要考虑的一点是，cuRobo的轨迹是有速度的，而这里我们把速度直接设为0了，所以会更慢一些
            for _ in range(2):

                cmd_state = cmd_plan[i]

                articulation_action = ArticulationAction(
                    joint_positions=cmd_state.position.cpu().numpy(),
                    joint_velocities=cmd_state.velocity.cpu().numpy() * 0.0,
                    joint_indices=[k for k in range(7)]
                )

                self.articulation_controller.apply_action(articulation_action)

                self.world.step(render=True)

task = Task()
task.task_plan()
task.run_task()
