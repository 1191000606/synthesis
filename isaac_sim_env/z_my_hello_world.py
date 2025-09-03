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
        "--enable", "isaacsim.robot.manipulators.examples"
    ]
}

simulation_app = SimulationApp(config)


from omni.isaac.core.utils.extensions import enable_extension

# 启用扩展
#enable_extension("isaacsim.examples.interactive")
#enable_extension("isaacsim.robot.manipulators")



# ------------ 2. 常规 Isaac API 代码 ----------
from isaacsim.examples.interactive.base_sample.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka.tasks.pick_place import PickPlace
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import  PickPlaceController


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._cube = None
        return

    # ---------- 创建场景 ----------
    def setup_scene(self):
        world = self.get_world()
        world.add_task(PickPlace(name="my_first_task"))
        return

    # ---------- 任务加载完毕后 ----------
    async def setup_post_load(self):
        self._world = self.get_world()
        params = self._world.get_task("my_first_task").get_params()

        self._franka = self._world.scene.get_object(params["robot_name"]["value"])
        self._cube_name = params["cube_name"]["value"]

        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )

        # 每一步调用 physics_step
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        await self._world.play_async()        # 开始仿真
        return

    async def setup_pre_reset(self):
        pass

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()

    # ---------- 物理循环 ----------
    def physics_step(self, step_size):
        obs = self._world.get_observations()

        actions = self._controller.forward(
            picking_position=obs[self._cube_name]["position"],
            placing_position=obs[self._cube_name]["goal_position"],
            current_joint_positions=obs[self._franka.name]["joint_positions"],
        )

        self._franka.apply_action(actions)

        # 控制器完成后暂停仿真
        if self._controller.is_done():
            self._world.pause()


# 创建 run 方法，开始仿真
    async def run(self):
        await self.load_world_async()  # 确保世界已经初始化
        await self.setup_scene()  # 调用场景设置
        await self.setup_post_load()  # 加载场景后，进行其他设置

# ------------ 3. 入口 ------------
if __name__ == "__main__":
    demo = HelloWorld()  # 实例化示例
    asyncio.run(demo.run())  # 使用 asyncio 运行异步事件循环

    # 确保仿真继续运行
    while simulation_app.is_running():
        simulation_app.update()  # 更新仿真
    simulation_app.close()  # 关闭仿真进程