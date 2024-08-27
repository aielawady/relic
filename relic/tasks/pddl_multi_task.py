from collections import Counter
import copy
from functools import cache, lru_cache
import inspect
import json
import os.path as osp
import random
from typing import Any, Dict, List

import numpy as np
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType,
    PddlEntity,
    SimulatorObjectType,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask

from omegaconf import DictConfig, ListConfig
import pandas as pd

import relic
from habitat.datasets.rearrange.navmesh_utils import (
    unoccluded_navmesh_snap,
    snap_point_is_occluded,
)

from habitat.datasets.rearrange.samplers.receptacle import (
    find_receptacles,
)
from habitat.core.dataset import Episode

from relic.tasks.utils import get_navigation_points, get_navigation_points_grid

from collections import defaultdict


def get_pddl(task_config) -> PddlDomain:
    config_path = osp.dirname(inspect.getfile(relic.default_structured_configs))
    domain_file_path = osp.join(
        config_path,
        task_config.task_spec_base_path,
        task_config.pddl_domain_def + ".yaml",
    )
    return PddlDomain(
        domain_file_path,
        task_config,
    )


class CustomRearrangeTask(RearrangeTask):
    def step(self, action: Dict[str, Any], episode: Episode):
        action_args = action.get("action_args", {})
        action["action_args"] = action_args
        output = super().step(action=action, episode=episode)
        self._last_success = self.measurements.get_metrics()["extobjnav_success"]
        self._last_num_steps = self.measurements.get_metrics()["num_steps"]
        return output


@registry.register_task(name="PddlMultiTask-v0")
class PddlMultiTask(CustomRearrangeTask):
    """
    Task that is specified by a PDDL goal expression and a set of PDDL start
    predicates.
    """

    def __init__(self, *args, config, **kwargs):
        self.pddl = get_pddl(config)

        super().__init__(*args, config=config, **kwargs)

        self._start_template = self._config.start_template
        self._goal_template = self._config.goal_template
        self._sample_entities = self._config.sample_entities
        self.fix_position_same_episode = self._config.fix_position_same_episode
        self.fix_target_same_episode = self._config.fix_target_same_episode
        self.fix_instance_index = self._config.fix_instance_index
        self.max_n_targets_per_sequence = self._config.max_n_targets_per_sequence
        self.target_sampling_strategy = self._config.target_sampling_strategy
        self.target_type = self._config.target_type
        self.cleanup_nav_points = self._config.cleanup_nav_points
        self._last_agent_state = None
        self._should_update_pos = True
        self.agent_height = 1.2
        self.robot_at_threshold = 2
        self.to_exclude_entities = []
        self.strict_target_selection = self._config.strict_target_selection
        self.selected_targets = dict()
        self.is_diff_episode = True
        self.force_is_diff = False
        self.fail_counts = defaultdict(lambda: 1)
        self.success_counts = defaultdict(lambda: 1)
        self.accum_num_steps = defaultdict(lambda: 0)

        self.to_ignore_types_all = {}  # {EPISODE-ID: [OBJECT-TYPE]}
        self.one_receptacle = self._config.get("one_receptacle", False)

        self.allowed_instances_all = {}  # {EPISODE-ID: [INSTANCE-HANDLE]}

        self._is_large_objs = self._config.get("is_large_objs", False)
        if self._is_large_objs:
            self._sample_entities = {"rec": self._sample_entities["rec"]}
        else:
            self._sample_entities = {"obj": self._sample_entities["obj"]}

        self.max_num_start_pos = self._config.get("max_num_start_pos", -1)
        self.start_pos_cache = []

    def _set_articulated_agent_start(self, agent_idx: int) -> None:
        if (
            self.max_num_start_pos > 0
            and len(self.start_pos_cache) == self.max_num_start_pos
        ):
            articulated_agent = self._sim.get_agent_data(agent_idx).articulated_agent
            i = np.random.choice(self.max_num_start_pos)
            articulated_agent.base_pos = self.start_pos_cache[i][0]
            articulated_agent.base_rot = self.start_pos_cache[i][1]
        elif self._should_update_pos or self._last_agent_state is None:
            super()._set_articulated_agent_start(agent_idx)
        else:
            articulated_agent = self._sim.get_agent_data(agent_idx).articulated_agent
            articulated_agent.base_pos = self._last_agent_state[0]
            articulated_agent.base_rot = self._last_agent_state[1]

    def _maybe_cache_start_pos(self, agent_idx: int = 0):
        if (
            self.max_num_start_pos > 0
            and len(self.start_pos_cache) < self.max_num_start_pos
        ):
            articulated_agent = self._sim.get_agent_data(agent_idx).articulated_agent
            self.start_pos_cache.append(
                (articulated_agent.base_pos, articulated_agent.base_rot)
            )

    def _setup_pddl_entities(self, episode):
        movable_entity_type = self.pddl.expr_types[
            SimulatorObjectType.MOVABLE_ENTITY.value
        ]
        # Register the specific objects in this scene as PDDL entities.
        for obj_name in self.pddl.sim_info.obj_ids:
            asset_name = _strip_instance_id(obj_name)
            asset_type = ExprType(asset_name, movable_entity_type)
            self.pddl.register_type(asset_type)
            self.pddl.register_episode_entity(PddlEntity(obj_name, asset_type))

        robot_entity_type = self.pddl.expr_types[SimulatorObjectType.ROBOT_ENTITY.value]
        for robot_id in self.pddl.sim_info.robot_ids:
            self.pddl.register_episode_entity(PddlEntity(robot_id, robot_entity_type))

        rec_entity_type = self.pddl.expr_types[
            SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
        ]
        for rec_id in self.pddl.sim_info.receptacles:
            rec_type = self.recp2type.get(rec_id.split("_")[0], None)
            if rec_type in ["bed", "couch", "chair", "tv", "plant", "toilet"]:
                rec_type = ExprType(rec_type, rec_entity_type)
                self.pddl.register_type(rec_type)
                self.pddl.register_episode_entity(PddlEntity(rec_id, rec_type))

    def get_available_types(self):
        for entity_name, entity_conds in self._sample_entities.items():
            match_type = self.pddl.expr_types[entity_conds["type"]]
            matches = list(self.pddl.find_entities(match_type))
            # Filter out the extra PDDL entities.
            matches = [
                match
                for match in matches
                if match.expr_type.name not in ["", "any"] + self.to_ignore_types
                and (not self.allowed_instances or match.name in self.allowed_instances)
            ]

            if len(matches) == 0:
                raise ValueError(
                    f"Could not find match for {entity_name}: {entity_conds}"
                )

        return set([x.expr_type.name for x in matches])

    def _load_start_info(
        self, episode, call_num, no_validation=False, object_type=None
    ):
        pddl_entities = self.pddl.all_entities
        self.pddl.bind_to_instance(self._sim, self._dataset, self, episode)
        if self.is_diff_episode:
            self.set_all_receptacle_ids()

        self._setup_pddl_entities(episode)

        if self.is_diff_episode:
            self.set_all_object_ids(episode)

        self.new_entities: Dict[str, PddlEntity] = {}
        for entity_name, entity_conds in self._sample_entities.items():
            match_type = self.pddl.expr_types[entity_conds["type"]]
            matches = list(self.pddl.find_entities(match_type))
            # Filter out the extra PDDL entities.
            matches = [
                match
                for match in matches
                if match.expr_type.name not in ["", "any"] + self.to_ignore_types
                and (not self.allowed_instances or match.name in self.allowed_instances)
                and (no_validation or match not in self.to_exclude_entities)
            ]

            if len(matches) == 0:
                raise ValueError(
                    f"Could not find match for {entity_name}: {entity_conds}"
                )

            if (
                self.max_n_targets_per_sequence > 0
                and len(self.selected_targets) >= self.max_n_targets_per_sequence
            ):
                expr_type_names_counts = self.selected_targets
            else:
                expr_type_names_counts = Counter([x.expr_type.name for x in matches])

            if object_type is not None:
                expr_type_name_rnd = object_type
            else:
                expr_type_names = []
                expr_type_names_p = []
                for name, count in expr_type_names_counts.items():
                    expr_type_names.append(name)
                    if self.target_sampling_strategy == "object_type":
                        expr_type_names_p.append(1)
                    elif self.target_sampling_strategy == "object_instance":
                        expr_type_names_p.append(count)
                    elif self.target_sampling_strategy == "inv_object_instance":
                        expr_type_names_p.append(1 / count)
                    elif self.target_sampling_strategy == "fail_count":
                        expr_type_names_p.append(self.fail_counts[name])
                    elif self.target_sampling_strategy == "UCB":
                        expr_type_names_p.append(
                            np.sqrt(
                                1
                                / self.success_counts[name]
                                * np.log(sum(self.success_counts.values()))
                            )
                        )
                    elif self.target_sampling_strategy == "neg_num_steps":
                        expr_type_names_p.append(
                            -self.accum_num_steps[name]
                            + max(self.accum_num_steps.values())
                            + 1
                        )
                    else:
                        raise ValueError
                expr_type_names_p = np.asarray(expr_type_names_p) / sum(
                    expr_type_names_p
                )

                expr_type_name_rnd = np.random.choice(
                    expr_type_names, p=expr_type_names_p
                )

        self.selected_targets[expr_type_name_rnd] = expr_type_names_counts[
            expr_type_name_rnd
        ]
        matches = sorted(matches, key=lambda x: x.name)
        if self.target_type == "object_type":
            object_index = None
        elif self.target_type == "object_instance":
            object_index = np.random.choice(
                len([x for x in matches if x.expr_type.name == expr_type_name_rnd])
            )
        else:
            raise ValueError

        if self.one_receptacle:
            receptacle = np.random.choice(
                [self.obj_handle2rec_type[x.name] for x in matches]
            )
            if receptacle == "unknown":
                receptacle = None
        else:
            receptacle = None

        result = self.process_matches(
            expr_type_name_rnd,
            object_index,
            episode.episode_id,
            no_validation,
            receptacle,
            call_num,
            tuple(self.to_exclude_entities),
        )
        if result:
            (
                self.new_entities,
                self.all_obj_pos,
                self.all_snapped_obj_pos,
                self.all_snapped_obj_pos_sizes,
                self.new_entities,
                self._sampled_names,
            ) = result
        else:
            (
                self.new_entities,
                self.all_obj_pos,
                self.all_snapped_obj_pos,
                self.all_snapped_obj_pos_sizes,
                self.new_entities,
                self._sampled_names,
            ) = ({}, [], [], [], {}, [])
            self._goal_expr = None
            return False
        self._goal_expr = self._load_goal_preds(episode)
        self._goal_expr, _ = self.pddl.expand_quantifiers(self._goal_expr)
        self.last_expr_type_name_rnd = expr_type_name_rnd
        self._receptacle_name = receptacle
        if self._config.get("goal_point_from_snapped", False):
            obj_i_ = np.random.choice(len(self.all_obj_pos))
            if obj_i_ == 0:
                start_idx = 0
                end_idx = self.all_snapped_obj_pos_sizes[0]
            else:
                start_idx = self.all_snapped_obj_pos_sizes[obj_i_ - 1]
                end_idx = self.all_snapped_obj_pos_sizes[obj_i_]
            self.random_obj_pos = np.asarray(
                self.all_snapped_obj_pos[start_idx:end_idx][
                    np.random.choice(end_idx - start_idx)
                ]
            )
        else:
            self.random_obj_pos = np.asarray(
                self.all_obj_pos[np.random.choice(len(self.all_obj_pos))]
            )
        return True

    @lru_cache(maxsize=None)
    def process_matches(
        self,
        expr_type_name_rnd,
        object_index,
        episode_id,
        no_validation,
        receptacle,
        call_num,
        to_exclude_entities,
    ):

        for entity_name, entity_conds in self._sample_entities.items():
            match_type = self.pddl.expr_types[entity_conds["type"]]
            matches = list(self.pddl.find_entities(match_type))
            # Filter out the extra PDDL entities.
            matches = [
                match
                for match in matches
                if match.expr_type.name not in ["", "any"] + self.to_ignore_types
                and (not self.allowed_instances or match.name in self.allowed_instances)
                and (no_validation or match not in self.to_exclude_entities)
            ]

        matches = sorted(matches, key=lambda x: x.name)
        if self.target_type == "object_type":
            new_entities = {
                f"obj{i}": ent
                for i, ent in enumerate(
                    [x for x in matches if x.expr_type.name == expr_type_name_rnd]
                )
            }
        elif self.target_type == "object_instance":
            new_entities = {
                f"obj{i}": ent
                for i, ent in enumerate(
                    [
                        [x for x in matches if x.expr_type.name == expr_type_name_rnd][
                            object_index
                        ]
                    ]
                )
            }
        else:
            raise ValueError

        if self.one_receptacle:
            new_entities = {
                k: v
                for k, v in new_entities.items()
                if self.obj_handle2rec_type[v.name] == receptacle
            }

        if len(new_entities) >= 10:
            print(
                f"Ignoring {expr_type_name_rnd} objects in episode {episode_id} because it has more than 9 objects."
            )
            return False

        all_obj_pos = [
            self.pddl.sim_info.get_entity_pos(entity)
            for entity in new_entities.values()
        ]
        all_snapped_obj_pos = [
            get_navigation_points_grid(
                pos,
                task=self,
                r=self.robot_at_threshold,
                target_object_id=self.object_handle2id.get(entity.name, None),
                ignore_non_negative=False,
                cleanup_nav_points=self.cleanup_nav_points,
            )
            for pos, entity in zip(all_obj_pos, new_entities.values())
        ]
        to_keep = [len(x) > 0 for x in all_snapped_obj_pos]

        if self.strict_target_selection and not all(to_keep):
            print(
                f"Ignoring {expr_type_name_rnd} objects in episode {episode_id} because it has at least one object that is not reachable."
            )
            return False

        if no_validation:
            to_keep = [True] * len(to_keep)
            print(f"Not validating the navigability in episode {episode_id}.")

        if not all(to_keep):
            print(
                f"Removing {len([x for x in to_keep if not x])} targets out of {len(to_keep)} in episode {episode_id}."
            )

        if not to_keep or not any(to_keep):
            print(
                f"Object type {expr_type_name_rnd} is not navigable in episode {episode_id}."
            )
            self.to_exclude_entities.extend(new_entities.values())
            return False

        if self.fix_instance_index:
            _indexes = [i for i, tk in enumerate(to_keep) if tk]
            to_keep = [False] * len(to_keep)
            to_keep[_indexes[self._instance_index % len(_indexes)]] = True

        all_obj_pos = [x for x, tk in zip(all_obj_pos, to_keep) if tk]
        all_snapped_obj_pos = [x for x, tk in zip(all_snapped_obj_pos, to_keep) if tk]
        all_snapped_obj_pos_sizes = np.cumsum([len(x) for x in all_snapped_obj_pos])
        all_snapped_obj_pos = sum(all_snapped_obj_pos, [])
        new_entities = {k: v for (k, v), tk in zip(new_entities.items(), to_keep) if tk}
        _sampled_names = list(new_entities.keys())
        if len(new_entities) == 0:
            return False

        return (
            new_entities,
            all_obj_pos,
            all_snapped_obj_pos,
            all_snapped_obj_pos_sizes,
            new_entities,
            _sampled_names,
        )

    def _load_goal_preds(self, episode):
        # Load from the config.
        # goal_d = dict(self._goal_template)
        goal_d = {
            "expr_type": "OR",
            "sub_exprs": [
                "robot_at(obj, robot_0)".replace("obj", name)
                for name in self.new_entities
            ],
        }
        goal_d = _recur_dict_replace(goal_d, self.new_entities)
        return self.pddl.parse_only_logical_expr(goal_d, self.pddl.all_entities)

    def _load_start_preds(self, episode):
        # Load from the config.
        start_preds = self._start_template[:]
        for pred in start_preds:
            for k, entity in self.new_entities.items():
                pred = pred.replace(k, entity.name)
            pred = self.pddl.parse_predicate(pred, self.pddl.all_entities)
            pred.set_state(self.pddl.sim_info)

    def is_goal_satisfied(self):
        return self.pddl.is_expr_true(self._goal_expr)

    def overwrite_sim_config(self, sim_config: "DictConfig", episode):
        if self.fix_position_same_episode:
            try:
                articulated_agent = self._sim.get_agent_data(0).articulated_agent
                self._last_agent_state = (
                    articulated_agent.base_pos,
                    articulated_agent.base_rot,
                )
            except Exception:
                pass

        return sim_config

    def set_all_object_ids(self, episode):
        all_objects = [
            entity
            for entity in self.pddl.find_entities(
                self.pddl.expr_types["movable_entity_type"]
            )
            if entity.expr_type.name not in ["any", ""]
        ]
        all_objects += [
            entity
            for entity in self.pddl.find_entities(
                self.pddl.expr_types["static_receptacle_entity_type"]
            )
            if entity.expr_type.name not in ["any", ""]
        ]
        rom = self._sim.get_rigid_object_manager()
        self.object_handle2id = {
            entity.name: rom.get_object_id_by_handle(entity.name)
            for entity in all_objects
        }
        self.all_object_ids = list(self.object_handle2id.values())
        self.obj_handle2rec_type = episode.name_to_receptacle

    def set_all_receptacle_ids(self):
        all_receptacle_handles = set(
            [(x.unique_name.split("|")[0]) for x in find_receptacles(self._sim)]
        )
        rom = self._sim.get_rigid_object_manager()
        self.all_receptacles_ids = [
            rom.get_object_id_by_handle(name) for name in all_receptacle_handles
        ]

        # TODO: Use config
        df = pd.read_csv("data/scene_datasets/hssd-hab/semantics/objects.csv")
        self.recp2type = {
            k: v
            for k, v in df[["id", "main_category"]].values
            if v in ["bed", "couch", "chair", "tv", "plant", "toilet"]
        }

    def reset(self, episode):
        if episode.episode_id in self.to_ignore_types_all:
            self.to_ignore_types = self.to_ignore_types_all[episode.episode_id]
        else:
            self.to_ignore_types = []

        if episode.episode_id in self.allowed_instances_all:
            self.allowed_instances = self.allowed_instances_all[episode.episode_id]
        else:
            self.allowed_instances = []

        if hasattr(self, "last_expr_type_name_rnd") and hasattr(self, "_last_success"):
            self.fail_counts[self.last_expr_type_name_rnd] += not self._last_success
            self.success_counts[self.last_expr_type_name_rnd] += self._last_success
        if hasattr(self, "last_expr_type_name_rnd") and hasattr(self, "_last_success"):
            self.accum_num_steps[self.last_expr_type_name_rnd] += self._last_num_steps

        self.to_exclude_entities = []
        self.is_diff_episode = self.force_is_diff or (
            self._episode_id != episode.episode_id
        )
        if self.is_diff_episode:
            self._instance_index = None
            self.selected_targets = dict()
            self.start_pos_cache = []

        if not self.fix_position_same_episode or self.is_diff_episode:
            self._should_update_pos = True
        else:
            self._should_update_pos = False
        self.num_steps = 0

        if self.fix_instance_index or self.is_diff_episode:
            self._instance_index = np.random.randint(100)

        super().reset(episode, fetch_observations=False)

        if not self.fix_target_same_episode or self.is_diff_episode:
            for call_num in range(50):
                if self._load_start_info(episode, call_num=call_num):
                    break
            else:
                print(f"Failed to load valid object for episode {episode.episode_id}.")
                self._load_start_info(episode, call_num=call_num, no_validation=True)

            self._load_start_preds(episode)

        self._sim.maybe_update_articulated_agent()
        self.agent_start = self._sim.articulated_agent.base_transformation
        self.force_is_diff = False

        self._maybe_cache_start_pos()

        return self._get_observations(episode)

    def get_sampled(self) -> List[PddlEntity]:
        return [self.new_entities[k] for k in self._sampled_names]

    def after_update(self):
        self.force_is_diff = True


def _recur_dict_replace(d: Any, replaces: Dict[str, PddlEntity]) -> Any:
    """
    Replace all string entries in `d` with the replace name to PDDL entity
    mapping in replaces.
    """
    if isinstance(d, ListConfig):
        d = list(d)
    if isinstance(d, DictConfig):
        d = dict(d)

    if isinstance(d, str):
        for name, entity in replaces.items():
            d = d.replace(f"{name}.type", entity.expr_type.name)
            d = d.replace(name, entity.name)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            d[i] = _recur_dict_replace(v, replaces)
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _recur_dict_replace(d[k], replaces)
    return d


def _strip_instance_id(instance_id: str) -> str:
    # Strip off the unique instance ID of the object and only return the asset
    # name.
    return "_".join(instance_id.split("_")[:-1])
