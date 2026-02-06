# CuRobo
from curobo.util_file import get_robot_configs_path, load_yaml
from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class CuRoboUtils:
    def init_robot_config():
        robot_config = load_yaml(get_robot_configs_path() + "/franka.yml")["robot_cfg"]

        robot_config["kinematics"]["extra_collision_spheres"] = {"attached_object": 50}
        robot_config["kinematics"]["collision_sphere_buffer"] = 0.007

        return robot_config

    def init_world_config(isaac_sim_world):
        usd_help = UsdHelper()
        usd_help.load_stage(isaac_sim_world.stage)

        world_config = usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path="/World/Franka",
            ignore_substring=["/World/Franka", "visuals"]
        )

        return world_config
    
    def init_tensor_args():
        tensor_args = TensorDeviceType()
        return tensor_args

    def init_ik_solver(robot_config, world_config, tensor_args):
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_config,
            world_config,
            num_seeds=20,
            tensor_args=tensor_args,
            collision_cache={"mesh": 20},
        )
        ik_solver = IKSolver(ik_config)
        return ik_solver

    def init_motion_gen(robot_config, world_config, tensor_args):
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config,
            world_config,
            tensor_args,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=1/60,
            collision_cache={"mesh": 20},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=32,
            trim_steps=None,
        )

        motion_gen = MotionGen(motion_gen_config)

        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=4,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.5, # 放慢轨迹速度，在CuRobo和Isaac Sim频率一致的时候方便跟踪
        )

        return motion_gen, plan_config

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
    
class CuRoboVariables:
    def __init__(self, robot_config, world_config, tensor_args, ik_solver, motion_gen, plan_config):
        self.robot_config = robot_config
        self.world_config = world_config
        self.tensor_args = tensor_args
        self.ik_solver = ik_solver
        self.motion_gen = motion_gen
        self.plan_config = plan_config