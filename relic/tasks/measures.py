from typing import Any, List, Optional, Tuple, Union
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import (
    BadCalledTerminate,
    DoesWantTerminate,
    RearrangeReward,
)
import habitat_sim
import numpy as np
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface
from omegaconf import DictConfig
from habitat.core.simulator import (
    Simulator,
)
from habitat.core.embodied_task import EmbodiedTask

from relic.tasks.utils import get_2d_point, get_obj_pixel_counts, is_target_occluded
import magnum as mn


@registry.register_measure
class PddlTaskSuccess(Measure):
    cls_uuid: str = "extobjnav_success"

    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._must_call_stop = config.must_call_stop
        self._must_see_object = config.must_see_object
        self._max_angle = config.max_angle
        self._sees_vertical_margin = config.sees_vertical_margin
        self._sees_horizontal_margin = config.sees_horizontal_margin
        self._ignore_objects = config.ignore_objects
        self._ignore_receptacles = config.ignore_receptacles
        self._ignore_non_negative = config.ignore_non_negative
        self._pixels_threshold = config.get("pixels_threshold", 10)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PddlTaskSuccess.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        if self._must_call_stop:
            task.measurements.check_measure_dependencies(
                self.uuid, [DoesWantTerminate.cls_uuid]
            )

        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RotDistToClosestGoal.cls_uuid,
            ],
        )

        self.update_metric(*args, task=task, **kwargs)

    def sees_object(self, task):
        agent_pos = task._sim.articulated_agent.base_pos + mn.Vector3(0, 1.2, 0)

        entities = list(task.new_entities.values())

        for i, (target_ent, target_pos) in enumerate(zip(entities, task.all_obj_pos)):
            if (
                not np.linalg.norm(np.asarray(target_pos - agent_pos)[[0, 2]])
                <= task.robot_at_threshold
            ):
                continue

            if i == 0:
                start_idx = 0
                end_idx = task.all_snapped_obj_pos_sizes[i]
            else:
                start_idx = task.all_snapped_obj_pos_sizes[i - 1]
                end_idx = task.all_snapped_obj_pos_sizes[i]

            snapped_points = np.asarray(task.all_snapped_obj_pos[start_idx:end_idx])
            if len(snapped_points) == 0:
                print("Issue with measures...")
                continue
            agent_norms = np.linalg.norm(
                (snapped_points - np.asarray(agent_pos))[:, [0, 2]], axis=1
            )
            closest_point = agent_norms.argmin()
            norm = np.linalg.norm(
                (snapped_points[closest_point] - np.asarray(target_pos))[[0, 2]]
            )
            if not np.linalg.norm(np.asarray(target_pos - agent_pos)[[0, 2]]) <= max(
                norm + 0.5, 2
            ):
                continue

            object_id = task._sim.get_rigid_object_manager().get_object_id_by_handle(
                target_ent.name
            )
            if (
                get_obj_pixel_counts(task, margin=5, strict=False).get(object_id, 0)
                < self._pixels_threshold
            ):
                continue

            return True

        return False

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.is_goal_satisfied()

        if self._metric and self._must_see_object:
            curr_angle = task.measurements.measures[
                RotDistToClosestGoal.cls_uuid
            ].get_metric()

            self._metric = (
                self._metric
                and (curr_angle < self._max_angle / 180 * np.pi)
                and self.sees_object(task)
            )

        if self._must_call_stop:
            does_action_want_stop = task.measurements.measures[
                DoesWantTerminate.cls_uuid
            ].get_metric()
            self._metric = self._metric and does_action_want_stop

            if does_action_want_stop:
                task.should_end = True


@registry.register_measure
class NamedNavToObjReward(RearrangeReward):
    cls_uuid: str = "named_nav_to_obj_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NamedNavToObjReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GeoDiscDistance.cls_uuid,
                L2Distance.cls_uuid,
                RotDistToClosestGoal.cls_uuid,
                BadCalledTerminate.cls_uuid,
            ],
        )
        self._cur_angle_dist = -1
        self._prev_dist = (
            -1.0 if self._config.max_reward_dist < 0 else self._config.max_reward_dist
        )

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        reward = self._metric

        cur_geodisc_dist = (
            task.measurements.measures[GeoDiscDistance.cls_uuid].get_metric()
        ) ** self._config.get("dist_reward_pow", 1)

        cur_geodisc_dist_no_pow = task.measurements.measures[
            GeoDiscDistance.cls_uuid
        ].get_metric()

        cur_l2_dist = task.measurements.measures[L2Distance.cls_uuid].get_metric()

        if self._config.use_max_dist:
            cur_geodisc_dist = max(cur_geodisc_dist, cur_l2_dist)
            cur_l2_dist = max(cur_geodisc_dist, cur_l2_dist)

        cur_geodisc_dist = (
            cur_geodisc_dist
            if self._config.max_reward_dist < 0
            else min(cur_geodisc_dist, self._config.max_reward_dist)
        )

        if self._prev_dist < 0.0:
            dist_diff = 0.0
        else:
            dist_diff = self._prev_dist - cur_geodisc_dist

        reward += self._config.dist_reward * dist_diff
        self._prev_dist = cur_geodisc_dist

        if self._config.should_reward_turn:
            if cur_geodisc_dist_no_pow < self._config.turn_reward_dist:
                angle_dist = task.measurements.measures[
                    RotDistToClosestGoal.cls_uuid
                ].get_metric()
                if self._cur_angle_dist < 0:
                    self._cur_angle_dist = angle_dist

                angle_diff = self._cur_angle_dist - angle_dist

                reward += self._config.angle_dist_reward * angle_diff
                self._cur_angle_dist = angle_dist

        bad_called_terminate = task.measurements.measures[
            BadCalledTerminate.cls_uuid
        ].get_metric()
        reward -= self._config.bad_term_pen * bad_called_terminate

        if self._config.end_on_bad_termination and bad_called_terminate:
            task.should_end = True

        self._metric = reward


@registry.register_measure
class GeoDiscDistance(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "geo_disc_distance"

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self._lock_closest_object = config.lock_closest_object
        self._add_point2object_dst = config.get("add_point2object_dst", False)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GeoDiscDistance.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.closest_object_index = -1
        self._locked_dsts = []
        self.update_metric(
            *args, task=task, episode=episode, observations=observations, **kwargs
        )

    def get_geodisc_distance(self, task, episode):
        agent_pos = task._sim.articulated_agent.base_pos
        agent_pos = task._sim.safe_snap_point(agent_pos)

        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = agent_pos
        if not self._locked_dsts:
            path.requested_ends = task.all_snapped_obj_pos
        else:
            path.requested_ends = self._locked_dsts
        did_find_a_path = task._sim.pathfinder.find_path(path)
        dist = path.geodesic_distance
        if self._lock_closest_object and not self._locked_dsts:
            closest_object_index = np.searchsorted(
                task.all_snapped_obj_pos_sizes,
                path.closest_end_point_index,
                side="right",
            )
            if closest_object_index == 0:
                start_index = 0
                end_index = task.all_snapped_obj_pos_sizes[closest_object_index]
            else:
                start_index = task.all_snapped_obj_pos_sizes[closest_object_index - 1]
                end_index = task.all_snapped_obj_pos_sizes[closest_object_index]
            self._locked_dsts = task.all_snapped_obj_pos[start_index:end_index]

        elif not self._lock_closest_object:
            self.closest_object_index = np.searchsorted(
                task.all_snapped_obj_pos_sizes,
                path.closest_end_point_index,
                side="right",
            )

        if not (did_find_a_path and not np.isnan(dist) and not np.isinf(dist)):
            print(
                f"Failed to calculate distance between {path.requested_start} and {path.requested_ends} in episode {episode.episode_id}. ",
                did_find_a_path,
                dist,
            )
            dist = 0
            self.closest_object_index = -1
        elif self._add_point2object_dst:
            dist += np.linalg.norm(
                (
                    task.all_snapped_obj_pos[path.closest_end_point_index]
                    - task.all_obj_pos[self.closest_object_index]
                )[[0, 2]]
            )

        return dist

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self.get_geodisc_distance(task, episode)


@registry.register_measure
class L2Distance(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "l2_distance"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return L2Distance.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.closest_object_index = -1
        self.update_metric(
            *args, task=task, episode=episode, observations=observations, **kwargs
        )

    def get_l2_distance(self, task, episode):
        agent_pos = task._sim.articulated_agent.base_pos
        agent_pos = task._sim.safe_snap_point(agent_pos)

        distances = [
            np.linalg.norm((np.asarray(p) - np.asarray(agent_pos))[[0, 2]])
            for p in task.all_obj_pos
        ]
        self.closest_object_index = np.argmin(distances)
        dist = np.min(distances)
        return dist

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self.get_l2_distance(task, episode)


@registry.register_measure
class RotDistToClosestGoal(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "rot_dist_to_closest_goal"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RotDistToClosestGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GeoDiscDistance.cls_uuid,
                L2Distance.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self.closest_object_index = task.measurements.measures[
            GeoDiscDistance.cls_uuid
        ].closest_object_index

        if self.closest_object_index < 0:
            self.closest_object_index = task.measurements.measures[
                L2Distance.cls_uuid
            ].closest_object_index

        targ = task.all_obj_pos[self.closest_object_index]

        # Get the agent
        robot = self._sim.get_agent_data(self.agent_id).articulated_agent
        # Get the base transformation
        T = robot.base_transformation
        # Do transformation
        pos = T.inverted().transform_point(targ)
        # Project to 2D plane (x,y,z=0)
        pos[2] = 0.0
        # Unit vector of the pos
        pos = pos.normalized()
        # Define the coordinate of the robot
        pos_robot = np.array([1.0, 0.0, 0.0])
        # Get the angle
        angle = np.arccos(np.dot(pos, pos_robot))
        metric = np.abs(float(angle))
        if np.isinf(metric) or np.isnan(metric):
            print(
                f"Setting angle to 0 becuase it's {metric}. The target is {targ} and after transformation is {pos}."
            )
            metric = 0

        self._metric = metric

    def is_occluded(self, task):
        agent_pos = task._sim.articulated_agent.base_pos + mn.Vector3(0, 1.2, 0)
        object_name = list(task.new_entities.values())[self.closest_object_index].name
        object_pos = task.all_obj_pos[self.closest_object_index]
        object_id = task._sim.get_rigid_object_manager().get_object_id_by_handle(
            object_name
        )
        return is_target_occluded(
            object_pos,
            agent_pos,
            0,
            task,
            ignore_object_ids=set([object_id]),
            ignore_non_negative=True,
        )


@registry.register_measure
class CamRotDistToClosestGoal(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "cam_rot_dist_to_closest_goal"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RotDistToClosestGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GeoDiscDistance.cls_uuid,
                L2Distance.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self.closest_object_index = task.measurements.measures[
            GeoDiscDistance.cls_uuid
        ].closest_object_index

        if self.closest_object_index < 0:
            self.closest_object_index = task.measurements.measures[
                L2Distance.cls_uuid
            ].closest_object_index

        targ = task.all_obj_pos[self.closest_object_index]

        render_camera = task._sim._sensors["head_rgb"]._sensor_object.render_camera

        # use the camera and projection matrices to transform the point onto the near plane
        pos = render_camera.projection_matrix.transform_point(
            render_camera.camera_matrix.transform_point(targ)
        ).normalized()

        pos_robot = np.array([0.0, 0.0, 1.0])

        # Get the angle
        angle = np.arccos(np.dot(pos, pos_robot))
        metric = np.abs(float(angle))
        if np.isinf(metric) or np.isnan(metric):
            print(
                f"Setting angle to 0 becuase it's {metric}. The target is {targ} and after transformation is {pos}."
            )
            metric = 0

        self._metric = metric

    def is_occluded(self, task):
        agent_pos = task._sim.articulated_agent.base_pos + mn.Vector3(0, 1.2, 0)
        object_name = list(task.new_entities.values())[self.closest_object_index].name
        object_pos = task.all_obj_pos[self.closest_object_index]
        object_id = task._sim.get_rigid_object_manager().get_object_id_by_handle(
            object_name
        )
        return is_target_occluded(
            object_pos,
            agent_pos,
            0,
            task,
            ignore_object_ids=set([object_id]),
            ignore_non_negative=True,
        )


@registry.register_measure
class SPLGeodisc(Measure):
    r"""SPL (PddlTaskSuccess weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and PddlTaskSuccess measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl_geodisc"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GeoDiscDistance.cls_uuid, PddlTaskSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            GeoDiscDistance.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        ep_success = task.measurements.measures[PddlTaskSuccess.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(self._start_end_episode_distance, self._agent_episode_distance, 1e-5)
        )


@registry.register_measure
class SoftSPLGeodisc(SPLGeodisc):
    r"""Soft SPL

    Similar to spl with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "soft_spl_geodisc"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GeoDiscDistance.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            GeoDiscDistance.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            GeoDiscDistance.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / max(self._start_end_episode_distance, 1e-5))
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(self._start_end_episode_distance, self._agent_episode_distance, 1e-5)
        )
