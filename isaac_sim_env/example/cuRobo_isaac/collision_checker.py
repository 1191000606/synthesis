# 在 Isaac Sim 场景中加载障碍物和一个可移动的球体，利用 cuRobo 库在 GPU 上实时计算球体与周围障碍物的碰撞距离（SDF, Signed Distance Field）和碰撞梯度（向量），并通过画线和改变颜色来进行可视化。

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": "1920", "height": "1080"})

from isaacsim.core.api import World
from isaacsim.core.api.materials import OmniPBR
from isaacsim.core.api.objects import VisualSphere
from isaacsim.util.debug_draw import _debug_draw

import torch
import numpy as np

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType

from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


def draw_line(start, gradient):
    # 获取 Isaac Sim 的调试绘图接口
    draw = _debug_draw.acquire_debug_draw_interface()
    # 清除上一帧画的线，防止画面混乱
    draw.clear_lines()
    
    # 定义线段起点（球心位置）
    start_list = [start]
    # 定义线段终点（起点 + 梯度向量）。这就画出了一个指示碰撞方向的箭头。
    end_list = [start + gradient]

    # 定义颜色：红色 (R=1, G=0, B=0, Alpha=0.8)
    colors = [(1, 0, 0, 0.8)]

    # 线条粗细
    sizes = [10.0]
    # 调用底层 API 绘图
    draw.draw_lines(start_list, end_list, colors, sizes)

def main():
    # 初始化 USD 辅助工具，用于在 Isaac Sim (USD) 和 cuRobo 之间转换数据
    usd_help = UsdHelper()
    # 碰撞激活距离：当物体距离障碍物小于 0.1m 时，开始计算精细碰撞
    act_distance = 0.1

    # 初始化 Isaac Sim 的物理世界，单位为米
    my_world = World(stage_units_in_meters=1.0)
    # 添加默认地面
    my_world.scene.add_default_ground_plane()

    # 获取当前 USD 舞台句柄
    stage = my_world.stage
    # 让 usd_help 工具绑定这个舞台，方便后续操作
    usd_help.load_stage(stage)
    
    # 创建 /World 根节点（Xform类型）
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    # 创建 /curobo 节点，通常用来存放 cuRobo 生成的调试可视化物体
    stage.DefinePrim("/curobo", "Xform")

    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))

    # 创建一个材质：绿色
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))

    # 定义球体半径
    radius = 0.1

    # 创建一个可视化的球体 (VisualSphere)
    target = VisualSphere(
        "/World/target",
        position=np.array([0.5, 0, 1.0]),
        orientation=np.array([1, 0, 0, 0]),
        radius=radius,
        visual_material=target_material,
    )

    # 设置日志级别
    setup_curobo_logger("warn")

    # 配置设备（默认使用 CUDA GPU）
    tensor_args = TensorDeviceType()
    
    # 指定配置文件名
    robot_file = "franka.yml" # 机器人配置文件
    world_file = ["collision_thin_walls.yml", "collision_test.yml"][-1] # 取列表最后一个，即 collision_test.yml

    # 碰撞检查类型：使用 MESH (网格)，精度比 Primitive (基本几何体) 更高，但计算量稍大
    collision_checker_type = CollisionCheckerType.MESH
    
    # 从 yaml 文件加载世界配置（包含障碍物信息）
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    # 修改第0个障碍物的高度，把它抬高 0.2米
    world_cfg.objects[0].pose[2] += 0.2
    vis_world_cfg = world_cfg

    # 【关键步骤】将 cuRobo 配置文件中的障碍物，加载到 Isaac Sim 的可见场景中
    # 如果没有这一步，你计算时有碰撞，但画面里看不到障碍物。
    usd_help.add_world_to_stage(vis_world_cfg, base_frame="/World")

    # 创建 RobotWorld 配置
    config = RobotWorldConfig.load_from_config(
        robot_file,
        world_cfg,
        collision_activation_distance=act_distance,
        collision_checker_type=collision_checker_type,
    )
    # 初始化 RobotWorld 实例（这是 cuRobo 的核心对象）
    model = RobotWorld(config)
    
    i = 0
    # 创建一个 PyTorch 张量来存储球体的位置和半径
    # 形状 (1, 1, 1, 4) 是 cuRobo 批处理计算要求的格式：(Batch, Time, Dims, x/y/z/radius)
    x_sph = torch.zeros((1, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    # 设置第 4 维为半径
    x_sph[..., 3] = radius

    my_world.play()

    # 只要仿真器还在运行
    while simulation_app.is_running():
        # 物理步进 + 渲染画面
        my_world.step(render=True)
        
        # 如果用户没点击 Isaac Sim 的 "Play" 按钮
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue # 跳过后续逻辑，等待用户点播放
            
        step_index = my_world.current_time_step_index

        # 刚开始的一帧重置一下世界
        if step_index == 0:
            my_world.reset()
        
        # 跳过前20帧，等待物理稳定
        if step_index < 20:
            continue

        # 每 1000 帧更新一次障碍物信息（避免每帧更新太卡）
        if step_index % 1000 == 0.0:
            # 从 USD 舞台中扫描所有的障碍物信息
            # 忽略掉 target（我们自己控制的球）和地面
            obstacles = usd_help.get_obstacles_from_stage(
                reference_prim_path="/World",
                ignore_substring=["/World/target", "/World/defaultGroundPlane"],
            ).get_collision_check_world()

            # 将扫描到的新障碍物信息更新给 cuRobo 计算模型
            model.update_world(obstacles)
            print("Updated World")
    
        # 获取当前球体在 Isaac Sim 中的实时位置
        sph_position, _ = target.get_local_pose()

        # 将位置数据填入 GPU 张量的前3位 (x, y, z)
        x_sph[..., :3] = tensor_args.to_device(sph_position).view(1, 1, 1, 3)

        # 【核心计算】调用 cuRobo 计算碰撞向量
        # d: 距离 (Signed Distance)，正值表示安全，0或负值表示碰撞
        # d_vec: 梯度向量，指向离开障碍物最近的方向
        d, d_vec = model.get_collision_vector(x_sph)

        # 转回 CPU 方便打印和逻辑判断
        d = d.view(-1).cpu()

        # 计算一个用于颜色显示的强度值 p
        p = d.item()
        p = max(1, p * 5) # 简单的数值放大

        # 如果距离不为0（实际上 cuRobo 返回0可能意味着没激活或者出错了，通常指有有效距离数据）
        if d.item() != 0.0:
            # 画线：从球心指向障碍物梯度的反方向（或正方向，视具体实现）
            draw_line(sph_position, d_vec[..., :3].view(3).cpu().numpy())
            print(d, d_vec)
        else:
            # 如果没数据，清除线条
            draw = _debug_draw.acquire_debug_draw_interface()
            draw.clear_lines()
            
        # --- 变色逻辑 ---
        if d.item() == 0.0:
            # 数据异常或正好在表面：绿色
            target_material.set_color(np.ravel([0, 1, 0]))
        elif d.item() <= model.contact_distance:
            # 距离非常近（小于接触阈值）：红色变调
            target_material.set_color(np.array([0, 0, p]))
        elif d.item() >= model.contact_distance:
            # 距离安全：蓝色变调
            target_material.set_color(np.array([p, 0, 0]))

if __name__ == "__main__":
    main()

