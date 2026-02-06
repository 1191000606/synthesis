# /home/chenyifan/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka文件夹里面config.json、config_cortex.json、config_no_feedback.json三个文件中的end_effector_frame_name，由“right_gripper”改成“panda_hand”

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.viewports import set_camera_view
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.types import ArticulationAction

import numpy as np

t = 0.0

def physics_step(step_size):
    global t

    if t < 1.0:
        robot_target_joint_positions = controller.forward(
            target_end_effector_position=np.array([0.2, 0.3, 0.4]),
            target_end_effector_orientation=np.array([0, 0.8, 0.6, 0]),
        )

        # 这里的controller不止生成关节位置，还生成速度
        franka.apply_action(robot_target_joint_positions)
        t += 0.01
    else:
        # 打开夹爪
        franka.apply_action(ArticulationAction(joint_positions=[None] * 7 + [0.04, 0.04]))

def init_world():
    world = World(stage_units_in_meters=1.0)

    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(3000)

    set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")
    
    return world

def init_robot():
    # 来自example包中的Franka类
    # from isaacsim.robot.manipulators.examples.franka import Franka
    # franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    usd_path = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
    
    prim_path="/World/Franka"

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    franka = world.scene.add(Robot(prim_path=prim_path, name="franka"))
    
    return franka

def init_controller(franka):
    rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_flow = mg.lula.motion_policies.RmpFlow(**rmp_flow_config)

    articulation_rmp = mg.ArticulationMotionPolicy(franka, rmp_flow)

    rmp_flow_controller = mg.MotionPolicyController(name="controller", articulation_motion_policy=articulation_rmp)

    rmp_flow_controller._motion_policy.set_robot_base_pose(
        robot_position=np.array([0.0, 0.0, 0.0]), 
        robot_orientation=np.array([1.0, 0.0, 0.0, 0.0])
    )

    return rmp_flow_controller


world = init_world()

franka = init_robot()

controller = init_controller(franka)

world.reset()

# 如果是以panda_rightfinger作为rmp配置文件中的end_effector_frame_name，那么最好是把夹爪设置为0.025, 0.025，这样误差最小
# 如果想达到和example包中franka的效果，那么把夹爪设置为0.010即可（仍然是以panda_rightfinger作为frame name）
# 如果以panda_hand作为rmp配置文件中的end_effector_frame_name，那么夹爪设置不影响效果，可以不设置
# franka.set_joint_positions(positions=[0.010238299, 0.010238837], joint_indices=[7,8]) 
# franka.set_joint_positions(positions=[0.025, 0.025], joint_indices=[7,8])

controller.reset()

world.add_physics_callback("sim_step", callback_fn=physics_step)

# 含义：启动物理仿真。
# 作用：这相当于你在 Isaac Sim 的图形界面（GUI）中点击了 “Play” 按钮。
# 细节：在这一行之前，物理场景是静止的（或者处于加载状态）。
# 执行这一行后，物理引擎（PhysX）开始工作，时间轴（Timeline）开始流动，场景中的物体开始受重力、碰撞等物理规律影响。
world.play()

# 只要 Isaac Sim 的应用程序窗口没有被关闭（用户没有点击窗口右上角的 X，也没有代码调用 close），循环就会一直执行。
while simulation_app.is_running():
    # 它会让物理引擎向前计算一个时间步（默认通常是 1/60 秒，取决于 physics_dt 的设置）。
    # 参数 render=True 表示在计算完物理后，同步更新图形渲染，让你可以看到物体移动了。
    # 它会把物理计算的结果（比如物体的新位置）同步回 USD 舞台，以便你可以获取最新的状态数据。
    world.step(render=True)

    # 这一行负责处理 Omniverse Kit 应用程序层面的事务。负责响应用户输入（鼠标点击、键盘按键）。负责刷新UI 界面（按钮、菜单）。
    simulation_app.update()

# 安全地关闭 Omniverse Kit 进程，释放 GPU 显存和内存资源，防止进程卡在后台。
simulation_app.close()


# motion generation中lula里面也有一个franka.urdf文件，其中引用的mesh文件可能是来自于/home/chenyifan/isaacsim/extscache/isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310/data/urdf/robots/franka_description