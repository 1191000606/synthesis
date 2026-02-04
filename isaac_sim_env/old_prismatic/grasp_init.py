# Isaac Sim
from isaacsim.core.api.world import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.physics import ContactSensor
from isaacsim.asset.importer.urdf import _urdf
from pxr import Sdf, UsdLux
import omni.kit.commands
import omni.usd

# CuRobo
from curobo.util_file import get_assets_path, get_robot_configs_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# Standard Library
import os
import numpy as np


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
    # 最好用与controller适配的机械臂模型文件
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
    # 最好用与controller适配的机械臂模型文件
    robot_usd_path = get_franka_usd()
    robot_prim_path="/World/Franka"
    add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
    franka = Robot(prim_path=robot_prim_path, name="franka")
    
    robot_config = load_yaml(get_robot_configs_path() + "/franka.yml")["robot_cfg"]

    return franka, robot_config

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


def init_ik_solver(robot_config, world_config, tensor_args):
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_config,
        world_config,
        num_seeds=20,
        tensor_args=tensor_args,
        collision_cache={"mesh": 100},
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver

def set_visuals_collision_instance(object_id):
    stage = omni.usd.get_context().get_stage()

    for prim in stage.Traverse():
        prim_str = str(prim)

        if "World" in prim_str and object_id in prim_str and ("collisions" in prim_str or "visuals" in prim_str):
            prim.SetInstanceable(False)

def init_world_config(usd_help):
    world_config = usd_help.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path="/World/Franka",
        ignore_substring=["/World/Franka", "visuals"]
    )

    return world_config

def init_motion_gen(robot_config, world_config, tensor_args):
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_config,
        world_config,
        tensor_args,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=1/60,
        collision_cache={"mesh": 100},
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
        time_dilation_factor=0.5,
    )

    return motion_gen, plan_config