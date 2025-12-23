# Third Party
import torch
# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

robot_file = "franka.yml"

# create a world from a dictionary of objects
# cuboid: {} # dictionary of objects that are cuboids
# mesh: {} # dictionary of objects that are meshes
world_config = {
    "cuboid": {
        "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
        "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
    },
    "mesh": {
        "scene": {
            "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
            "file_path": "scene/nvblox/srl_ur10_bins.obj",
        }
    },
}
tensor_args = TensorDeviceType()
config = RobotWorldConfig.load_from_config(robot_file, world_config,
                                          collision_activation_distance=0.0)
curobo_fn = RobotWorld(config)

# create spheres with shape batch, horizon, n_spheres, 4.
q_sph = torch.randn((10, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
q_sph[...,3] = 0.2 # radius of spheres

# 计算这些球体与环境障碍物（table, cube, mesh）之间的距离
d = curobo_fn.get_collision_distance(q_sph)

# 随机采样 5 组机器人关节角度 (q_s)
# mask_valid=False 表示不需要保证采样的姿态是无碰撞的，仅仅是随机采样
q_s = curobo_fn.sample(5, mask_valid=False)

# 根据关节角度计算距离
# d_world: 机器人每个连杆(Link)到环境障碍物的最近距离
# d_self:  机器人连杆与连杆自身之间的最近距离（自碰撞检测）
d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)

# 当距离大于0，也就是没碰撞时，这时返回的距离值是0，而不是精确的距离值，这样是为了计算更快
# 当设置collision_activation_distance，例如等于10cm时，当距离大于10cm时，仍然返回0，当距离在0～10cm时，返回负的距离，大于0表示碰撞