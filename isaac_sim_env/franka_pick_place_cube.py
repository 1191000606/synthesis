# 针对一直遇到的 困扰了很久的 库未发现问题
# 两种解决方法：

# 1.启动SimulationApp时，config里添加"extra_args": ["--enable", "isaacsim.examples.interactive"]

# 2.在代码中手动启用扩展：
# from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("isaacsim.examples.interactive")


# pick_place_demo.py
import isaacsim
from isaacsim.simulation_app import SimulationApp
import asyncio

# 配置启动参数
config = {
    "headless": False,
    "extra_args": [
        "--enable", "isaacsim.examples.interactive",
        "--enable", "isaacsim.robot.manipulators.examples",
        "--enable", "isaacsim.core.api",  # 确保启用任务基础API

    ]
}

simulation_app = SimulationApp(config)


from omni.isaac.core.utils.extensions import enable_extension

# 启用扩展
#enable_extension("isaacsim.examples.interactive")
#enable_extension("isaacsim.robot.manipulators")
#enable_extension("isaacsim.core.api")  # 确保启用任务基础API



# ------------ 2. 常规 Isaac API 代码 ----------
# 这个类，我目前的理解是GUI界面的交互接口（垃圾类，别用）
from isaacsim.examples.interactive.base_sample.base_sample import BaseSample
# 核心任务模块，所有任务的基类，提供任务生命周期的管理，关于场景设置、重制、观察设置、任务完成情况等。
from isaacsim.core.api.tasks.base_task import BaseTask


from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.tasks.pick_place import PickPlace
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import  PickPlaceController

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid, GroundPlane

from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view

class FrankaPlaying(BaseTask):
    def __init__(self, name, offset=None):
        super().__init__(name=name, offset=offset)
        self._goal_position = np.array([-0.3, -0.3, 0.05515/2.0])
        self._task_achieved = False
        return
    # 似乎是必有的方法
    # 设置任务的场景
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(
            prim_path="/World/random_cube",
            name="fancy_cube",
            position=np.array([0.3, 0.3, 0.3]),
            color=np.array([0.0, 0.0, 1.0]),
            size=0.05
        ))
        self._franka = scene.add(Franka(
            prim_path="/World/Fancy_Franka",
            name="fancy_franka"
        ))
        return
    #似乎是必有的方法
    # 获取观测值，架起环境状态与控制器的桥梁，使得控制器可以获取到环境状态信息
    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations
    # 每个物理步之前执行，用于检测任务完成状态，更新视觉反馈
    # 我感觉目前这个代码没有被执行，即还没有自动更新视觉状态，但由于现在的任务很简单，故不更新视觉也可以完成。
    
    # 后面再想想办法，怎么让这段代码实际起作用
    # 我找了一下，似乎是在world类中有一个step方法，或者step_async方法，其中用到了basetask的pre_step方法
    # 那么应该研究一下world的step方法。
    # 没错，只要加上demo._world.step(render=True) ，就可以了
    # 在有多步控制时，用control index索引来做多步控制策略，确定控制哪一步骤
    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()
        # print(f"Cube Position: {cube_position}, Goal Position: {np.mean(np.abs(self._goal_position - cube_position))}")
        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
            self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return



    # 用于仿真重置，任务恢复到初始状态
    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._cube.get_applied_visual_material().set_color(color=np.array([1.0, 0, 0]))
        self._task_achieved = False
        return




# 这个类，继承自BaseSample，但BaseSample有问题，里面有的方法无法使用，实际使用时就当是一个集成的类吧


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._cube = None
        return
    # 将任务添加到仿真环境中，构建场景    
    def setup_scene(self):
        self._world = World(stage_units_in_meters=1.0)
        world = self.get_world()
        world.add_task(FrankaPlaying(name="my_first_task"))
        world.reset()  # 重置世界状态
        set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")
        return
    # ---------- 任务加载完毕后 ----------
    # 场景加载完成后，启动物理仿真和控制器
    async def setup_post_load(self):
        # 初始化物理实体
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        # 控制器，传入机械臂和夹爪关节状态，从而进行逆运动学求解和路径规划
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        # 每一物理步前的回调函数，指定回调的方法，不太懂
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        # 异步起动仿真
        return
    
    async def setup_pre_reset(self):
        return
    # 仿真重置后被调用，重置控制器，重新启动仿真
    async def setup_post_reset(self):
        # 重置控制器
        self._controller.reset()
        # 重新启动仿真，异步保证在仿真开始前，完成所有重置操作，避免状态的不一致
        await self._world.play_async()
        return
    #物理步回调方法
    def physics_step(self, step_size):
        # 获取当前的观测值，正如前面定义的，返回机器人和cube的状态
        current_observations = self._world.get_observations()
        # 控制器根据当前观测值计算动作
        actions = self._controller.forward(
            picking_position=current_observations["fancy_cube"]["position"],
            placing_position=current_observations["fancy_cube"]["goal_position"],
            current_joint_positions=current_observations["fancy_franka"]["joint_positions"]
        )
    # 检查任务是否完成，如果完成则暂停仿真
        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()

        return
    def world_cleanup(self):
        self._cube = None
        return






import asyncio

if __name__ == "__main__":
    demo = HelloWorld()  # 实例化HelloWorld类
    
    # 使用 asyncio 来运行异步加载世界和设置任务
    demo.setup_scene()
    asyncio.run(demo.setup_post_load())  # 等待任务加载完成后，启动仿真

    # 等待仿真运行，直到仿真结束
    while simulation_app.is_running():
        demo._world.step(render=True)  # 每一帧渲染
        simulation_app.update()  # 更新仿真状态

    simulation_app.close()  # 关闭仿真进程












    # 使用 asyncio 来运行异步加载世界和设置任务
    # asyncio.run(demo.load_world_async())  # 异步加载世界并设置任务
    # md，想骂人，load_world_async调用到create_new_stage_async时，
    # 有行 await omni.kit.app.get_app().next_update_async()
    # 运行到这，程序就一动不动，也不报错
    # 我进行搜索，根本就没有next_update_async这个方法，倒是有next_update，不知道是不是版本更新，这个方法被删除，但仍被调用
    # fuck you nvidia
    # 但为什么不报错？
    # md，我认为不能再用BaseSample类了
    # 必须自己实现
    # 应该是可以的，创建world啥的
    # 等待仿真运行，直到仿真结束





    # from isaacsim.core.api import World
    # my_world = World(stage_units_in_meters=1.0)

    # my_task = FrankaPlaying()
    # my_world.add_task(my_task)
    # my_world.reset()
    # my_franka = my_world.scene.get_object("fancy_robot")
    # my_controller = PickPlaceController(name="generic_pd_controller", kp=my_task._pd_gains[0], kd=my_task._pd_gains[1])
    # articulation_controller = my_franka.get_articulation_controller()
    



