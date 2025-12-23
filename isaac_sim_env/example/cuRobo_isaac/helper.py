# Isaac Sim
from isaacsim.core.api.world import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.api.objects import cuboid

import numpy as np
from pxr import Sdf, UsdLux
import omni.kit.commands
import omni.usd

# CuRobo
from curobo.util_file import get_assets_path, get_world_configs_path, get_robot_configs_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig


# Standard Library
import os

def init_world():
    world = World(stage_units_in_meters=1.0)

    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(3000)

    # set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")
    
    return world

def init_scene():
    target = cuboid.VisualCuboid("/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    config = load_yaml(get_world_configs_path() + "/collision_table.yml") # /collision_mesh_scene.yml
    config["cuboid"]["table"]["dims"] = [0.2, 0.2, 0.005]

    world_config = WorldConfig.from_dict(config).get_mesh_world()
    world_config.mesh[0].pose = [0.5, 0.0, 0.6, 1, 0, 0, 0]
    
    return target, world_config

def get_franka_usd():
    urdf_path = get_assets_path() + "/robot/franka_description/franka_panda.urdf"
    dest_path = get_assets_path() + "/robot/franka_description/cuRobo_franka.usd"

    if os.path.exists(dest_path):
        return dest_path

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False # 凸包计算碰撞等全部由cuRobo接管，不需要isaac sim来做了
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751 # 刚度，PID中的P
    import_config.default_position_drive_damping = 52.35988 # 阻尼，PID中的D
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
    robot_usd_path = get_franka_usd()
    robot_prim_path="/World/Franka"
    add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
    franka = Robot(prim_path=robot_prim_path, name="franka")
    
    robot_config = load_yaml(get_robot_configs_path() + "/franka.yml")["robot_cfg"]

    return franka, robot_config

def init_motion_gen(robot_config, world_config, tensor_args):
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_config,
        world_config,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"mesh": 5},
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
