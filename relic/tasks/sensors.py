import gym.spaces as spaces
import numpy as np
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes


@registry.register_sensor
class RelativeLocalizationSensor(Sensor):
    cls_uuid = "rel_localization_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return RelativeLocalizationSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        agent = self._sim.get_agent_data(self.agent_id).articulated_agent

        rel_base_pos = agent.base_pos - task.agent_start.translation
        return np.array(rel_base_pos, dtype=np.float32)


@registry.register_sensor
class OneHotTargetSensor(Sensor):
    def __init__(self, *args, task, **kwargs):
        self._task = task
        # TODO: Hard-coded for the ycb objects. Change to work with any object
        # set.
        if self._task._config.get("is_large_objs", False):
            self._all_objs = ["bed", "couch", "chair", "tv", "plant", "toilet"]
        else:
            self._all_objs = [
                "002_master_chef_can",  # 0
                "003_cracker_box",  # 1
                "004_sugar_box",  # 2
                "005_tomato_soup_can",  # 3
                "007_tuna_fish_can",  # 4
                "008_pudding_box",  # 5
                "009_gelatin_box",  # 6
                "010_potted_meat_can",  # 7
                "011_banana",  # 8
                "012_strawberry",  # 9
                "013_apple",  # 10
                "014_lemon",  # 11
                "015_peach",  # 12
                "016_pear",  # 13
                "017_orange",  # 14
                "018_plum",  # 15
                "021_bleach_cleanser",  # 16
                "024_bowl",  # 17
                "025_mug",  # 18
                "026_sponge",  # 19
            ]
        self._n_cls = len(self._all_objs)

        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "one_hot_target_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(self._n_cls,), low=0, high=1, dtype=np.float32)

    def get_observation(self, *args, **kwargs):
        cur_target = self._task.get_sampled()[0]

        # For receptacles the name will not be a class but the name directly.
        use_name = cur_target.expr_type.name
        if cur_target.name in self._all_objs:
            use_name = cur_target.name

        if use_name not in self._all_objs:
            raise ValueError(
                f"Object not found given {use_name}, {cur_target}, {self._task.get_sampled()}"
            )
        set_i = self._all_objs.index(use_name)

        obs = np.zeros((self._n_cls,))
        if use_name in self._all_objs:
            set_i = self._all_objs.index(use_name)
            obs[set_i] = 1.0

        return obs


@registry.register_sensor
class OneHotReceptacleSensor(Sensor):
    def __init__(self, *args, task, **kwargs):
        self._task = task
        # TODO: Hard-coded for the ycb objects. Change to work with any object
        # set.
        self._all_objs = [
            None,
            "couch",
            "unknown",
            "table",
            "cabinet",
            "chair",
            "bed",
            "chest_of_drawers",
            "shelves",
            "stool",
            "toilet",
            "washer_dryer",
            "bench",
            "stand",
            "counter",
            "bathtub",
            "car",
            "wardrobe",
            "sink",
            "shower",
        ]
        self._n_cls = len(self._all_objs)

        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "one_hot_receptacle_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(self._n_cls,), low=0, high=1, dtype=np.float32)

    def get_observation(self, *args, **kwargs):
        rec_name = self._task._receptacle_name

        if rec_name not in self._all_objs:
            raise ValueError(f"Receptacle not found given {rec_name}.")
        set_i = self._all_objs.index(rec_name)

        obs = np.zeros((self._n_cls,))
        if rec_name in self._all_objs:
            set_i = self._all_objs.index(rec_name)
            obs[set_i] = 1.0

        return obs


from habitat.core.simulator import (
    SemanticSensor,
    Sensor,
    VisualObservation,
)
import habitat_sim
from omegaconf import DictConfig
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

if TYPE_CHECKING:
    from torch import Tensor

from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSimSensor,
    check_sim_obs,
    HabitatSimSemanticSensor,
)


@registry.register_sensor
class MaskedSemanticSensor(Sensor):
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_uuid(self, *args, **kwargs):
        return "masked_semantic_sensor"

    def __init__(self, *args, task, **kwargs) -> None:
        self._task = task
        # TODO: Hard-coded for the ycb objects. Change to work with any object
        # set.
        self._all_objs = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
            "011_banana",
            "012_strawberry",
            "013_apple",
            "014_lemon",
            "015_peach",
            "016_pear",
            "017_orange",
            "018_plum",
            "021_bleach_cleanser",
            "024_bowl",
            "025_mug",
            "026_sponge",
        ]
        self._n_cls = len(self._all_objs)

        super().__init__(*args, **kwargs)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.config.height, self.config.width, 1),
            dtype=np.bool_,
        )

    def get_observation(self, *args, observations, **kwargs) -> VisualObservation:
        obs = observations["head_semantic"]
        ids = []
        for obj in self._task.new_entities.values():
            ids.append(self._task.object_handle2id[obj.name])

        obs_mask = np.zeros_like(obs, dtype="bool")
        for id_ in ids:
            obs_mask |= obs == id_ + self._task._sim.habitat_config.object_ids_start

        return obs_mask


@registry.register_sensor
class MaskedFlattenedSemanticSensor(Sensor):
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_uuid(self, *args, **kwargs):
        return "masked_flattened_semantic_sensor"

    def __init__(self, *args, task, **kwargs) -> None:
        self._task = task
        # TODO: Hard-coded for the ycb objects. Change to work with any object
        # set.
        self._all_objs = [
            "002_master_chef_can",  # 0
            "003_cracker_box",  # 1
            "004_sugar_box",  # 2
            "005_tomato_soup_can",  # 3
            "007_tuna_fish_can",  # 4
            "008_pudding_box",  # 5
            "009_gelatin_box",  # 6
            "010_potted_meat_can",  # 7
            "011_banana",  # 8
            "012_strawberry",  # 9
            "013_apple",  # 10
            "014_lemon",  # 11
            "015_peach",  # 12
            "016_pear",  # 13
            "017_orange",  # 14
            "018_plum",  # 15
            "021_bleach_cleanser",  # 16
            "024_bowl",  # 17
            "025_mug",  # 18
            "026_sponge",  # 19
        ]
        self._n_cls = len(self._all_objs)

        super().__init__(*args, **kwargs)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.config.n_cells * self.config.n_cells, 1),
            dtype=np.float32,
        )

    def get_observation(self, *args, observations, **kwargs) -> VisualObservation:
        obs = observations["head_semantic"]
        ids = []
        for obj in self._task.new_entities.values():
            ids.append(self._task.object_handle2id[obj.name])

        obs_mask = np.zeros_like(obs, dtype="bool")
        for id_ in ids:
            obs_mask |= obs == id_ + self._task._sim.habitat_config.object_ids_start

        obs_mask = obs_mask.astype("float32")

        STEP = obs.shape[0] // self.config.n_cells
        feats = np.zeros((self.config.n_cells * self.config.n_cells, 1))
        for i in range(self.config.n_cells):
            for j in range(self.config.n_cells):
                feats[i * self.config.n_cells + j] = obs_mask[
                    i * STEP : (i + 1) * STEP, j * STEP : (j + 1) * STEP
                ].mean()

        return feats


from habitat.tasks.nav.nav import PointGoalSensor


@registry.register_sensor(name="PointGoalWithGPSCompassSensorV3")
class IntegratedPointGoalGPSAndCompassSensorV3(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass_v3"

    def __init__(self, *args: Any, task, **kwargs: Any) -> str:
        self._task = task
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.last_goal = None
        self.noise = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        if (
            self.last_goal is None
            or (self.last_goal != self._task.random_obj_pos).any()
        ):
            self.last_goal = self._task.random_obj_pos
            self.noise = (
                np.random.randn(len(self._task.random_obj_pos))
                * self.counter
                / 1_000_000
                * self.config.get("std_noise_1m", 0)
            )

        goal_position = (self._task.random_obj_pos) + self.noise
        self.counter += 1
        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
