# /home/chenyifan/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka文件夹里面config.json、config_cortex.json、config_no_feedback.json三个文件中的end_effector_frame_name，由“right_gripper”改成“panda_hand”
import numpy as np
import omni.usd
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Rotation, Slerp
from pxr import Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.viewports import set_camera_view
import isaacsim.robot_motion.motion_generation as mg


def init_world():
    world = World(stage_units_in_meters=1.0)

    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(3000)

    set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")
    
    return world

def init_robot():
    usd_path = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
    
    prim_path="/World/Franka"

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    franka = Robot(prim_path=prim_path, name="franka")

    end_effector = XFormPrim("/World/Franka/panda_hand")
    
    return franka, end_effector

def init_controller(franka, obstacle):
    rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_flow = mg.lula.motion_policies.RmpFlow(**rmp_flow_config)

    rmp_flow.add_obstacle(obstacle)

    articulation_rmp = mg.ArticulationMotionPolicy(franka, rmp_flow)

    rmp_flow_controller = mg.MotionPolicyController(name="controller", articulation_motion_policy=articulation_rmp)

    rmp_flow_controller._motion_policy.set_robot_base_pose(
        robot_position=np.array([0.0, 0.0, 0.0]), 
        robot_orientation=np.array([1.0, 0.0, 0.0, 0.0])
    )

    return rmp_flow_controller

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


def if_end_effector_achieved(object_current_position, object_target_position):
    object_distance = np.linalg.norm(object_current_position - object_target_position)
    print(f"Object distance to target: {object_distance}")
    return object_distance < 0.1


def get_interval(end_position, start_position, velocity):
    distance = np.linalg.norm(end_position - start_position)
    total_step = distance / velocity
    t_increment = 1.0 / total_step
    return t_increment


def get_interpolated_position(end_position, start_position, t):
    alpha = 0.5 * (1 - np.cos(t * np.pi))
    interpolated_position = (1 - alpha) * start_position + alpha * end_position
    return interpolated_position


def quaternion_angular_distance(q1, q2):
    dot_product = np.dot(q1, q2)
    dot_product_abs = np.clip(abs(dot_product), -1.0, 1.0)
    radian = 2 * np.arccos(dot_product_abs)

    angle = np.rad2deg(radian)

    return angle

def init_slerp(end_orientation, start_orientation):
    key_rotations = Rotation.from_quat(
        [
            [start_orientation[1], start_orientation[2], start_orientation[3], start_orientation[0]],
            [end_orientation[1], end_orientation[2], end_orientation[3], end_orientation[0]],
        ]
    )

    key_times = [0, 1]
    slerp = Slerp(key_times, key_rotations)

    return slerp


def get_interpolated_orientation(slerp, t):
    interpolated_rotation = slerp(t)

    interpolated_orientation = interpolated_rotation.as_quat(scalar_first=True)

    return interpolated_orientation
