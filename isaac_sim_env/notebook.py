



--------------------
from typing import List, Optional

import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper


class PickPlaceController(manipulators_controllers.PickPlaceController):
    """[summary]

    Args:
        name (str): [description]
        gripper (ParallelGripper): [description]
        robot_articulation (SingleArticulation): [description]
        end_effector_initial_height (Optional[float], optional): [description]. Defaults to None.
        events_dt (Optional[List[float]], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [0.008, 0.005, 1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )
        return


---------------------
from typing import Optional

import isaacsim.core.api.tasks as tasks
import numpy as np
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot.manipulators.examples.franka import Franka


class PickPlace(tasks.PickPlace):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "franka_pick_place".
        cube_initial_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        cube_initial_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str = "franka_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )
        return

    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(prim_path=franka_prim_path, name=franka_robot_name)
-----------------


from typing import List, Optional

import carb
import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.storage.native import get_assets_root_path


class Franka(Robot):
    """[summary]

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "franka_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_rightfinger"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_rightfinger"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([0.05, 0.05]) / get_stage_units()
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )
        return

    @property
    def end_effector(self) -> SingleRigidPrim:
        """[summary]

        Returns:
            SingleRigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return



------------------------------
import gc
from abc import abstractmethod

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view


class BaseSample(object):
    def __init__(self) -> None:
        self._world = None
        self._current_tasks = None
        self._world_settings = {"physics_dt": 1.0 / 60.0, "stage_units_in_meters": 1.0, "rendering_dt": 1.0 / 60.0}
        # self._logging_info = ""
        return

    def get_world(self):
        return self._world

    def set_world_settings(self, physics_dt=None, stage_units_in_meters=None, rendering_dt=None):
        if physics_dt is not None:
            self._world_settings["physics_dt"] = physics_dt
        if stage_units_in_meters is not None:
            self._world_settings["stage_units_in_meters"] = stage_units_in_meters
        if rendering_dt is not None:
            self._world_settings["rendering_dt"] = rendering_dt
        return

    async def load_world_async(self):
        """Function called when clicking load buttton"""
        await create_new_stage_async()
        self._world = World(**self._world_settings)
        await self._world.initialize_simulation_context_async()
        self.setup_scene()
        set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")
        self._current_tasks = self._world.get_current_tasks()
        await self._world.reset_async()
        await self._world.pause_async()
        await self.setup_post_load()
        if len(self._current_tasks) > 0:
            self._world.add_physics_callback("tasks_step", self._world.step_async)
        return

    async def reset_async(self):
        """Function called when clicking reset buttton"""
        if self._world.is_tasks_scene_built() and len(self._current_tasks) > 0:
            self._world.remove_physics_callback("tasks_step")
        await self._world.play_async()
        await update_stage_async()
        await self.setup_pre_reset()
        await self._world.reset_async()
        await self._world.pause_async()
        await self.setup_post_reset()
        if self._world.is_tasks_scene_built() and len(self._current_tasks) > 0:
            self._world.add_physics_callback("tasks_step", self._world.step_async)
        return

    @abstractmethod
    def setup_scene(self, scene: Scene) -> None:
        """used to setup anything in the world, adding tasks happen here for instance.

        Args:
            scene (Scene): [description]
        """
        return

    @abstractmethod
    async def setup_post_load(self):
        """called after first reset of the world when pressing load,
        intializing provate variables happen here.
        """
        return

    @abstractmethod
    async def setup_pre_reset(self):
        """called in reset button before resetting the world
        to remove a physics callback for instance or a controller reset
        """
        return

    @abstractmethod
    async def setup_post_reset(self):
        """called in reset button after resetting the world which includes one step with rendering"""
        return

    @abstractmethod
    async def setup_post_clear(self):
        """called after clicking clear button
        or after creating a new stage and clearing the instance of the world with its callbacks
        """
        return

    # def log_info(self, info):
    #     self._logging_info += str(info) + "\n"
    #     return

    def _world_cleanup(self):
        self._world.stop()
        self._world.clear_all_callbacks()
        self._current_tasks = None
        self.world_cleanup()
        return

    def world_cleanup(self):
        """Function called when extension shutdowns and starts again, (hot reloading feature)"""
        return

    async def clear_async(self):
        """Function called when clicking clear buttton"""
        await create_new_stage_async()
        if self._world is not None:
            self._world_cleanup()
            self._world.clear_instance()
            self._world = None
            gc.collect()
        await self.setup_post_clear()
        return

——————————————————————————————————————————————————————————

from typing import Optional

import numpy as np
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.simulation_context import SimulationContext


class BaseTask(object):
    """This class provides a way to set up a task in a scene and modularize adding objects to stage,
    getting observations needed for the behavioral layer, calculating metrics needed about the task,
    calling certain things pre-stepping, creating multiple tasks at the same time and much more.

    Checkout the required tutorials at
    https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html

    Args:
        name (str): needs to be unique if added to the World.
        offset (Optional[np.ndarray], optional): offset applied to all assets of the task.
    """

    def __init__(self, name: str, offset: Optional[np.ndarray] = None) -> None:
        self._scene = None
        self._name = name
        self._offset = offset
        self._task_objects = dict()
        if self._offset is None:
            self._offset = np.array([0.0, 0.0, 0.0])

        if SimulationContext.instance() is not None:
            self._device = SimulationContext.instance().device
        return

    @property
    def device(self):
        return self._device

    @property
    def scene(self) -> Scene:
        """Scene of the world

        Returns:
            Scene: [description]
        """
        return self._scene

    @property
    def name(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return self._name

    def set_up_scene(self, scene: Scene) -> None:
        """Adding assets to the stage as well as adding the encapsulated objects such as SingleXFormPrim..etc
           to the task_objects happens here.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        return

    def _move_task_objects_to_their_frame(self):

        # if self._task_path:
        # TODO: assumption all task objects are under the same parent
        # Specifying a task path has many limitations atm
        # SingleXFormPrim(prim_path=self._task_path, position=self._offset)
        # for object_name, task_object in self._task_objects.items():
        #     new_prim_path = self._task_path + "/" + task_object.prim_path.split("/")[-1]
        #     task_object.change_prim_path(new_prim_path)
        #     current_position, current_orientation = task_object.get_world_pose()
        for object_name, task_object in self._task_objects.items():
            current_position, current_orientation = task_object.get_world_pose()
            task_object.set_world_pose(position=current_position + self._offset)
            task_object.set_default_state(position=current_position + self._offset)
        return

    def get_task_objects(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return self._task_objects

    def get_observations(self) -> dict:
        """Returns current observations from the objects needed for the behavioral layer.

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        raise NotImplementedError

    def calculate_metrics(self) -> dict:
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Returns True of the task is done.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def post_reset(self) -> None:
        """Calls while doing a .reset() on the world."""
        return

    def get_description(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return ""

    def cleanup(self) -> None:
        """Called before calling a reset() on the world to removed temporary objects that were added during
        simulation for instance.
        """
        return

    def set_params(self, *args, **kwargs) -> None:
        """Changes the modifiable parameters of the task

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Gets the parameters of the task.
           This is defined differently for each task in order to access the task's objects and values.
           Note that this is different from get_observations.
           Things like the robot name, block name..etc can be defined here for faster retrieval.
           should have the form of params_representation["param_name"] = {"value": param_value, "modifiable": bool}

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: defined parameters of the task.
        """
        raise NotImplementedError