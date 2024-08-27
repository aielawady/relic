# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
import os
import random
from typing import TYPE_CHECKING, Any, List, Type

from habitat import ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat.config import read_write
from habitat.gym import make_gym_from_config
from habitat_baselines.common.env_factory import VectorEnvFactory
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size
import torch
import importlib

if TYPE_CHECKING:
    from omegaconf import DictConfig


def get_make_env_func_by_name(name):
    if "." in name:
        module_name, func_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
    else:
        func = globals()[name]
    return func


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
            vector_env_cls = VectorEnv

        envs = vector_env_cls(
            make_env_fn=get_make_env_func_by_name(make_env_func_name),
            env_fn_args=tuple((c,) for c in configs),
            workers_ignore_signals=workers_ignore_signals,
        )

        if config.habitat.simulator.renderer.enable_batch_renderer:
            envs.initialize_batch_renderer(config)

        return envs
