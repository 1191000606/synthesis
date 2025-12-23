# Third Party
import torch
# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

robot_file = "franka.yml"

world_config = {
    "cuboid": {
        # dims表示物体在X，Y，Z方向的尺寸。pose表示物体的位置和朝向，前3个值是位置(x,y,z)，后4个值是四元数表示的旋转(w,x,y,z)
        "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
    },
}

collision_activation_distance = 0.9

tensor_args = TensorDeviceType()
config = RobotWorldConfig.load_from_config(robot_file, world_config, collision_activation_distance=collision_activation_distance)
curobo_fn = RobotWorld(config)

custom_spheres = [
    # 1.【严重碰撞】球心 z=0.0 -> 底部 z=-0.2。深深嵌入桌子内部。
    [0.4, 0.0, 0.0, 0.2], 
    
    # 2.【轻微碰撞】球心 z=0.18 -> 底部 z=-0.02。轻微嵌入桌子 2cm。
    [0.4, 0.0, 0.18, 0.2],

    # 3.【安全但很近】球心 z=0.25 -> 底部 z=0.05。离桌子只有 5cm 间隙。
    [0.4, 0.0, 0.25, 0.2],

    # 4.【完全安全】球心 z=1.0 -> 离得非常远。
    [0.4, 0.0, 1.0, 0.2],
]

# 将列表转换为 Tensor，并调整形状为 (Batch, Horizon, N_spheres, 4)
# 这里 Batch=4 (对应上面4个点)
q_sph = torch.tensor(custom_spheres, device=tensor_args.device, dtype=tensor_args.dtype)
q_sph = q_sph.view(4, 1, 1, 4)

# 计算这些球体与环境障碍物（table, cube, mesh）之间的距离
d = curobo_fn.get_collision_distance(q_sph)

print("collision Activation Distance", collision_activation_distance)
print("Collision distances:", d)


# 这里的 d 是穿透深度，正值表示发生了碰撞，绝对值表示穿透的深度。小于等于0表示没有碰撞，但没有碰撞时为了计算更快，会返回0，而不是精确距离值
# 当设置collision_activation_distance，例如等于10cm时，当距离大于10cm时，仍然返回0，当距离在0～10cm时，返回负的距离，大于0表示碰撞