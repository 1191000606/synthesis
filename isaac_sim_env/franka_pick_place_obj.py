import isaacsim
from isaacsim.simulation_app import SimulationApp
import asyncio

# 配置启动参数
config = {
    "headless": False,
    "extra_args": [
        "--enable", "isaacsim.examples.interactive",
        "--enable", "isaacsim.robot.manipulators.examples",
        "--enable", "isaacsim.core.api", 

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

import omni.kit.commands
from isaacsim.core.prims import XFormPrim
from pxr import UsdPhysics, Sdf




class FrankaPlaying(BaseTask):
    def __init__(self, name, offset=None, scene_config=None):
        super().__init__(name=name, offset=offset)
        self._goal_position = np.array([-0.3, -0.3, 0.05515/2.0])
        self._task_achieved = False
        self._scene_config = scene_config
        return

    
    # 在场景中添加urdf物体的方法：
    # success, prim_path = okc.execute(
        #     "URDFParseAndImportFile",
        #     urdf_path=urdf_path,
        #     import_config=cfg,
        #     get_articulation_root=True,
            
        # )
    # 得到prim_path后，用类似
    # prims = XFormPrim(prim_paths_expr="/World" )
    # scene.add(prims)
    # 这样来在scene中添加物体
    
    # 先实现一个这样的函数：
    # 给定obj_id，那么
    # urdf_path = f"/home/szwang/synthesis/data/objaverse/data/obj/{obj_id}/material.urdf"
    
    
    
    def set_up_scene(self, scene):
        
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        # self._cube = scene.add(DynamicCuboid(
        #     prim_path="/World/random_cube",
        #     name="fancy_cube",
        #     position=np.array([0.3, 0.3, 0.3]),
        #     color=np.array([0.0, 0.0, 1.0]),
        #     size=0.05
        # ))
        self._franka = scene.add(Franka(
            prim_path="/World/Fancy_Franka",
            name="fancy_franka"
        ))
        for i in range(len(self._scene_config["obj_ids"])):
            obj_id = self._scene_config["obj_ids"][i]
            position = self._scene_config["positions"][i]
            scale = self._scene_config["scales"][i]
            movable = self._scene_config["movables"][i]
            obj = self.add_objaverse_scene(scene, obj_id, position, scale, movable)
            if i==0:
                self._cube = obj
        self.add_partnet_scene(scene, "4564", [1.3, 0.3, 0.3], [1.0, 1.0, 1.0], True)
        return
    
    def add_objaverse_scene(self, scene, obj_id, position, scale, movable):
        urdf_path = f"/home/szwang/synthesis/data/objaverse/data/obj/{obj_id}/material.urdf"
        _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        cfg.merge_fixed_joints    = False
        cfg.convex_decomp         = False
        cfg.import_inertia_tensor = True
        cfg.fix_base              = not movable  #这这个目前还有问题


        success, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=cfg,
            get_articulation_root=True,
        )



        pos = np.asarray(position, dtype=np.float32).reshape(1, 3)
        
        if np.isscalar(scale):
            scl = np.full((1, 3), float(scale), dtype=np.float32)
        else:
            scl = np.asarray(scale, dtype=np.float32).reshape(1, 3)

        prims = XFormPrim(
            prim_paths_expr=prim_path,
            name=str(obj_id),
            positions=pos,           # (1,3)
            translations=None,       # 不要和 positions 同时给
            scales=scl               # (1,3)
        )   
        print("prims")
        print(prims)
        

        obj = scene.add(prims)
        return obj
    
    def add_partnet_scene(self, scene, obj_id, position, scale, movable):
        urdf_path = f"/home/szwang/synthesis/data/dataset/{obj_id}/mobility.urdf"
        _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        cfg.merge_fixed_joints    = False
        cfg.convex_decomp         = False
        cfg.import_inertia_tensor = True
        cfg.fix_base              = not movable  #这个目前还有问题。
        cfg.distance_scale       = 1.0


        success, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=cfg,
            get_articulation_root=True,
        )



        pos = np.asarray(position, dtype=np.float32).reshape(1, 3)
        
        if np.isscalar(scale):
            scl = np.full((1, 3), float(scale), dtype=np.float32)
        else:
            scl = np.asarray(scale, dtype=np.float32).reshape(1, 3)

        prims = XFormPrim(
            prim_paths_expr=prim_path,
            name=str(obj_id),
            positions=pos,           # (1,3)
            translations=None,       # 不要和 positions 同时给
            scales=scl               # (1,3)
        )   
        print("prims")
        print(prims)
        

        obj = scene.add(prims)
        return obj
    
    
    #似乎是必有的方法
    # 获取观测值，架起环境状态与控制器的桥梁，使得控制器可以获取到环境状态信息
    def get_observations(self):
        # cube_position, _ = self._cube.get_world_poses()
        pos_batch, _ = self._cube.get_world_poses()   # (1,3), (1,4)
        cube_position = pos_batch[0]                  # → (3,)
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

    # demo._world.step(render=True) 来调用pre_step,渲染每一帧
    # 在有多步控制时，用control index索引来做多步控制策略，确定控制哪一步骤
    def pre_step(self, control_index, simulation_time):
        # cube_position, _ = self._cube.get_world_poses()
        pos_batch, _ = self._cube.get_world_poses()   # (1,3)
        cube_position = pos_batch[0]                  # (3,)
        # print(f"Cube Position: {cube_position}, Goal Position: {np.mean(np.abs(self._goal_position - cube_position))}")
        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
            # self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return



    # 用于仿真重置，任务恢复到初始状态
    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        # self._cube.get_applied_visual_material().set_color(color=np.array([1.0, 0, 0]))
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
        ####!!!!
        world.add_task(FrankaPlaying(name="my_first_task",scene_config = scene_config))
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
            picking_position=current_observations["939bce9ccaec4d5ab3404dca172d2f45"]["position"],
            placing_position=current_observations["939bce9ccaec4d5ab3404dca172d2f45"]["goal_position"],
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
    # scene_config是一个字典，包含场景的配置信息，包含的key为obj_ids、positions、scales、movables
    scene_config = {
        "obj_ids": [
            "939bce9ccaec4d5ab3404dca172d2f45",
            "3729cf312d054b9db8767c934ed13215",
            "03567b3881dc44a98ff3e6c1d449e32d",
        ],
        "positions": [
            (0.5, 0, 0),
            (-2, 2, 0),
            (0, -2, 0),
        ],
        "scales": [
            (0.05, 0.03, 0.2),
            (2, 2, 2),
            (1, 1, 1.5),
        ],
        "movables": [
            True,
            True,
            True,
        ]
    }
    demo = HelloWorld()  # 实例化HelloWorld类
    
    # 使用 asyncio 来运行异步加载世界和设置任务
    demo.setup_scene()
    asyncio.run(demo.setup_post_load())  # 等待任务加载完成后，启动仿真

    # 等待仿真运行，直到仿真结束
    while simulation_app.is_running():
        demo._world.step(render=True)  # 每一帧渲染
        simulation_app.update()  # 更新仿真状态

    simulation_app.close()  # 关闭仿真进程
