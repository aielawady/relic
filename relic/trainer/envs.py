from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

import gym
from habitat.core.simulator import Observations
from habitat.utils import profiling_wrapper
import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig
import gym
import habitat
from habitat import Dataset
from habitat.core.environments import RLTaskEnv
from habitat.gym.gym_wrapper import HabGymWrapper


class CustomRLTaskEnv(RLTaskEnv):
    def after_update(self):
        self._env.episode_iterator.after_update()
        task = self._env.task
        if hasattr(task, "after_update"):
            task.after_update()


@habitat.registry.register_env(name="CustomGymHabitatEnv")
class CustomGymHabitatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(self, config: "DictConfig", dataset: Optional[Dataset] = None):
        base_env = CustomRLTaskEnv(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)


from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
import magnum as mn
from habitat.datasets.rearrange.samplers.receptacle import (
    AABBReceptacle,
    find_receptacles,
)


@registry.register_simulator(name="CustomRearrangeSim-v0")
class CustomRearrangeSim(RearrangeSim):
    def _create_recep_info(
        self, scene_id: str, ignore_handles: List[str]
    ) -> Dict[str, mn.Range3D]:
        if scene_id not in self._receptacles_cache:
            receps = {}
            all_receps = find_receptacles(
                self,
                ignore_handles=ignore_handles,
            )
            for recep in all_receps:
                recep = cast(AABBReceptacle, recep)
                local_bounds = recep.bounds
                global_T = recep.get_global_transform(self)
                # Some coordinates may be flipped by the global transformation,
                # mixing the minimum and maximum bound coordinates.
                bounds = np.stack(
                    [
                        global_T.transform_point(local_bounds.min),
                        global_T.transform_point(local_bounds.max),
                    ],
                    axis=0,
                )
                receps[recep.unique_name.split("|")[0]] = mn.Range3D(
                    np.min(bounds, axis=0), np.max(bounds, axis=0)
                )
            self._receptacles_cache[scene_id] = receps
        return self._receptacles_cache[scene_id]
