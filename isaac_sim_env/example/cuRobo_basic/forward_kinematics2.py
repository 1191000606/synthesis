# Third Party
import torch

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml


tensor_args = TensorDeviceType()

# 区别就在这里，这里是直接一个config file，而fk.py是urdf_file, base_link, ee_link和tensor_args，看起来后面这四个都可以自己给
config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(config_file, tensor_args)

kin_model = CudaRobotModel(robot_cfg.kinematics)

# compute forward kinematics:
# torch random sampling might give values out of joint limits
q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
out = kin_model.get_state(q)