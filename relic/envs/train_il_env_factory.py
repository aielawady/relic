# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
import os
import random
import signal
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Type

from habitat import ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat.config import read_write
from habitat.gym import make_gym_from_config
from habitat_baselines.common.env_factory import VectorEnvFactory
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size
import torch
from multiprocessing.connection import Connection
from habitat.core.logging import logger
from habitat.gym.gym_env_episode_count_wrapper import EnvCountEpisodeWrapper
from habitat.gym.gym_env_obs_dict_wrapper import EnvObsDictWrapper

from habitat.core.vector_env import (
    CALL_COMMAND,
    CLOSE_COMMAND,
    COUNT_EPISODES_COMMAND,
    RENDER_COMMAND,
    RESET_COMMAND,
    STEP_COMMAND,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from relic.tasks.utils import get_obj_pixel_counts

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import numpy as np
import importlib


def get_make_env_func_by_name(name):
    if "." in name:
        module_name, func_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
    else:
        func = globals()[name]
    return func


class DummyAgent:
    def __init__(self, sim, agent_id=0):
        self.sim = sim
        self.agent_id = agent_id
        self.agent = self.sim.get_agent(agent_id)

    def __getattr__(self, attr):
        return getattr(self.agent, attr)

    @property
    def state(self):
        return self.sim.get_agent_state(self.agent_id)


class Hab3ShortestPathFollower(ShortestPathFollower):
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self._init_state()

    def _build_follower(self, *args, **kwargs):
        super()._build_follower(*args, **kwargs)
        self._follower.agent = DummyAgent(self._sim)

    def _init_state(self):
        self.is_around_the_obj = False
        self.state = -1
        self.inside_state_counter = 0
        self.last_angle = None

    def after_update(self):
        self._current_scene = None
        self._init_state()

    def get_next_action(self, goal_pos):
        # l2_distance = self.task.measurements.get_metrics()["l2_distance"]
        if not self.is_around_the_obj:
            next_action = super().get_next_action(goal_pos)
            if next_action == HabitatSimActions.stop or next_action is None:
                self.is_around_the_obj = True
                self.state = 0
            else:
                return next_action

        rot_dist_to_closest_goal = self.task.measurements.get_metrics()[
            "rot_dist_to_closest_goal"
        ]
        if self.state == 0:
            next_action = HabitatSimActions.turn_left
            self.inside_state_counter += 1
            if self.last_angle is None:
                self.last_angle = rot_dist_to_closest_goal
            elif (rot_dist_to_closest_goal > self.last_angle) and (
                rot_dist_to_closest_goal < 1
            ):
                self.state += 1
                self.inside_state_counter = 0
                self.last_angle = None
        elif self.state == 1:
            next_action = HabitatSimActions.turn_right
            self.inside_state_counter += 1
            if self.last_angle is None:
                self.last_angle = rot_dist_to_closest_goal
            if rot_dist_to_closest_goal > self.last_angle:
                self.state += 1
                self.inside_state_counter = 0
        elif self.state == 2:
            next_action = 4
            self.inside_state_counter += 1
            if self.inside_state_counter > 5:
                self.state += 1
                self.inside_state_counter = 0
        elif self.state == 3:
            next_action = 5
            self.inside_state_counter += 1
            if self.inside_state_counter > 5:
                self.state += 1
                self.inside_state_counter = 0

        closest_object_id = self.task.measurements.measures[
            "rot_dist_to_closest_goal"
        ].closest_object_index
        if self.state >= 4 or (
            self.state >= 2
            and get_obj_pixel_counts(self.task, margin=5, strict=False).get(
                closest_object_id, 0
            )
            > 10
        ):
            next_action = HabitatSimActions.stop
        self.last_angle = rot_dist_to_closest_goal
        # print(self.state, get_obj_pixel_counts(self.task, margin=5, strict=False).get(closest_object_id, 0), next_action)
        return next_action


class ShortestPathVectorEnv(VectorEnv):
    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = EnvCountEpisodeWrapper(EnvObsDictWrapper(env_fn(*env_fn_args)))
        try:
            follower = Hab3ShortestPathFollower(
                env.env.env.habitat_env.sim,
                0.25,
                False,
                task=env.env.env.habitat_env._task,
            )
        except Exception:
            logger.warn("Couldn't create shortest path follower.")
            follower = None

        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    observations, reward, done, info = env.step(data)

                    if auto_reset_done and done:
                        observations = env.reset()
                        if follower is not None:
                            follower._init_state()

                    connection_write_fn((observations, reward, done, info))

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_name == "best_action":
                        task = env.env.env.habitat_env._task
                        closest_index = task.measurements.measures[
                            "rot_dist_to_closest_goal"
                        ].closest_object_index
                        if closest_index == 0:
                            start_index = 0
                            end_index = task.all_snapped_obj_pos_sizes[closest_index]
                        else:
                            start_index = task.all_snapped_obj_pos_sizes[
                                closest_index - 1
                            ]
                            end_index = task.all_snapped_obj_pos_sizes[closest_index]
                        snapped_points = np.asarray(
                            task.all_snapped_obj_pos[start_index:end_index]
                        )
                        agent_norms = np.linalg.norm(
                            (
                                snapped_points
                                - np.asarray(task.all_obj_pos[closest_index])
                            )[:, [0, 2]],
                            axis=1,
                        )
                        closest_point = agent_norms.argmin()

                        if (agent_norms < 1.5).any():
                            snapped_points = snapped_points[agent_norms < 1.5]
                            goal_point = snapped_points[
                                np.random.choice(len(snapped_points))
                            ]
                        else:
                            goal_point = snapped_points[closest_point]
                        action = follower.get_next_action(goal_point)
                        connection_write_fn(action)
                    elif function_name == "episodes" and hasattr(
                        env.env.env, "habitat_env"
                    ):
                        iterator_ = env.env.env.habitat_env.episode_iterator
                        connection_write_fn(
                            [(x.scene_id, x.episode_id) for x in iterator_.episodes]
                        )
                    else:
                        if function_name == "after_update" and follower is not None:
                            follower.after_update()

                        if function_args is None:
                            function_args = {}

                        result_or_fn = getattr(env, function_name)

                        if len(function_args) > 0 or callable(result_or_fn):
                            result = result_or_fn(**function_args)
                        else:
                            result = result_or_fn

                        connection_write_fn(result)

                elif command == COUNT_EPISODES_COMMAND:
                    connection_write_fn(len(env.episodes))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                command, data = connection_read_fn()

        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            if child_pipe is not None:
                child_pipe.close()
            env.close()


class HabitatVectorEnvFactory(VectorEnvFactory):
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
        distribute_envs_across_gpus=None,
    ) -> VectorEnv:
        r"""Create VectorEnv object with specified config and env class type.
        To allow better performance, dataset are split into small ones for
        each individual env, grouped by scenes.
        """
        if distribute_envs_across_gpus is None:
            distribute_envs_across_gpus = enforce_scenes_greater_eq_environments

        num_environments = config.habitat_baselines.num_environments
        configs = []
        make_env_func_name = config.habitat.task.get(
            "make_env_fn", "make_gym_from_config"
        )
        if make_env_func_name == "make_gym_from_config":
            dataset = make_dataset(config.habitat.dataset.type)
            scenes = list(config.habitat.dataset.content_scenes)
            if "*" in config.habitat.dataset.content_scenes:
                scenes = dataset.get_scenes_to_load(config.habitat.dataset)
                scenes = sorted(scenes)
                local_rank, world_rank, world_size = get_distrib_size()
                split_size = ceil(len(scenes) / world_size)
                orig_size = len(scenes)
                scenes = scenes[world_rank * split_size : (world_rank + 1) * split_size]
                scenes_ids = list(range(orig_size))[
                    world_rank * split_size : (world_rank + 1) * split_size
                ]
                logger.warn(f"Loading {len(scenes)}/{orig_size}. IDs: {scenes_ids}")

            if num_environments < 1:
                raise RuntimeError("num_environments must be strictly positive")

            if len(scenes) == 0:
                raise RuntimeError(
                    "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
                )

            random.shuffle(scenes)

            scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
            for idx in range(max(len(scene_splits), len(scenes))):
                scene_splits[idx % len(scene_splits)].append(scenes[idx % len(scenes)])

            logger.warn(f"Scene splits: {scene_splits}.")
            assert all(scene_splits)
        else:
            scenes = []

        for env_index in range(num_environments):
            proc_config = config.copy()
            with read_write(proc_config):
                if distribute_envs_across_gpus:
                    proc_config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                        env_index % torch.cuda.device_count()
                    )

                task_config = proc_config.habitat
                task_config.seed = task_config.seed + env_index
                remove_measure_names = []
                if not is_first_rank:
                    # Filter out non rank0_measure from the task config if we are not on rank0.
                    remove_measure_names.extend(task_config.task.rank0_measure_names)
                if (env_index != 0) or not is_first_rank:
                    # Filter out non-rank0_env0 measures from the task config if we
                    # are not on rank0 env0.
                    remove_measure_names.extend(
                        task_config.task.rank0_env0_measure_names
                    )

                task_config.task.measurements = {
                    k: v
                    for k, v in task_config.task.measurements.items()
                    if k not in remove_measure_names
                }

                if len(scenes) > 0:
                    task_config.dataset.content_scenes = scene_splits[env_index]

            configs.append(proc_config)

        vector_env_cls: Type[Any]
        if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
            logger.warn(
                "Using the debug Vector environment interface. Expect slower performance."
            )
            vector_env_cls = ThreadedVectorEnv
        else:
            vector_env_cls = ShortestPathVectorEnv

        envs = vector_env_cls(
            make_env_fn=get_make_env_func_by_name(make_env_func_name),
            env_fn_args=tuple((c,) for c in configs),
            workers_ignore_signals=workers_ignore_signals,
        )

        if config.habitat.simulator.renderer.enable_batch_renderer:
            envs.initialize_batch_renderer(config)

        return envs
