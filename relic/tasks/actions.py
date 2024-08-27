#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO, lots of typing errors in here

from copy import deepcopy
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import attr
import numpy as np
import quaternion
from gym import spaces

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
)
from habitat.tasks.rearrange.actions.articulated_agent_action import (
    ArticulatedAgentAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import ActionSpace, EmptySpace
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat_sim import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass

try:
    import magnum as mn
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig

#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, cast

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.articulated_agent_action import (
    ArticulatedAgentAction,
)

# flake8: noqa
# These actions need to be imported since there is a Python evaluation
# statement which dynamically creates the desired grip controller.
from habitat.tasks.rearrange.actions.grip_actions import (
    GazeGraspAction,
    GripSimulatorTaskAction,
    MagicGraspAction,
    SuctionGraspAction,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import rearrange_collision, rearrange_logger

from habitat.tasks.rearrange.actions.actions import BaseVelAction

cv2 = try_cv2_import()


def _strafe_body(
    sim,
    move_amount: float,
    strafe_angle_deg: float,
    noise_amount: float,
):
    # Get the state of the agent
    agent_state = sim.get_agent_state()
    # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
    normalized_quaternion = agent_state.rotation
    agent_mn_quat = mn.Quaternion(
        normalized_quaternion.imag, normalized_quaternion.real
    )
    forward = agent_mn_quat.transform_vector(-mn.Vector3.z_axis())
    strafe_angle = np.random.uniform(
        (1 - noise_amount) * strafe_angle_deg,
        (1 + noise_amount) * strafe_angle_deg,
    )
    strafe_angle = mn.Deg(strafe_angle)
    rotation = mn.Quaternion.rotation(strafe_angle, mn.Vector3.y_axis())
    move_amount = np.random.uniform(
        (1 - noise_amount) * move_amount, (1 + noise_amount) * move_amount
    )
    delta_position = rotation.transform_vector(forward) * move_amount
    final_position = sim.pathfinder.try_step(  # type: ignore
        agent_state.position, agent_state.position + delta_position
    )
    sim.set_agent_state(
        final_position,
        [*rotation.vector, rotation.scalar],
        reset_sensors=False,
    )


class NavigationMovementAgentAction(ArticulatedAgentAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self._tilt_angle = config.tilt_angle
        self.max_angle = 100
        self.min_angle = 20

    def _move_camera_vertical(self, amount: float):
        agent_data = self._sim.get_agent_data(0)
        for cam_name in agent_data.articulated_agent.params.cameras:
            *curr_xy, _ = agent_data.articulated_agent.params.cameras[
                cam_name
            ].cam_look_at_pos
            norm = np.linalg.norm(curr_xy)

            angle = np.arctan2(*curr_xy) - amount / 180 * np.pi
            angle = min(
                max(angle, self.min_angle / 180 * np.pi), self.max_angle / 180 * np.pi
            )
            x, y = np.sin(angle) * norm, np.cos(angle) * norm

            agent_data.articulated_agent.params.cameras[cam_name].cam_look_at_pos[0] = x
            agent_data.articulated_agent.params.cameras[cam_name].cam_look_at_pos[1] = y


@registry.register_task_action
class RearrangeMoveForwardAction(ArticulatedAgentAction):
    name: str = "rearrange_move_forward"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        # return self._sim.step(HabitatSimActions.move_forward)
        _strafe_body(self._sim, 0.25, 0, 0)


@registry.register_task_action
class RearrangeTurnLeftAction(ArticulatedAgentAction):
    name: str = "rearrange_turn_left"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        _strafe_body(self._sim, 0.0, 30, 0)


@registry.register_task_action
class RearrangeTurnRightAction(ArticulatedAgentAction):
    name: str = "rearrange_turn_right"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        _strafe_body(self._sim, 0.0, -30, 0)


@registry.register_task_action
class RearrangeStopAction(ArticulatedAgentAction):
    name: str = "rearrange_stop"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        self.does_want_terminate = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        self.does_want_terminate = True  # type: ignore


@registry.register_task_action
class RearrangeLookUpAction(NavigationMovementAgentAction):
    name: str = "rearrange_look_up"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        agent_data = self._sim.get_agent_data(0)
        self.cams_init_pos = {}

        for cam_name in agent_data.articulated_agent.params.cameras:
            self.cams_init_pos[cam_name] = list(
                agent_data.articulated_agent.params.cameras[cam_name].cam_look_at_pos
            )

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        self._move_camera_vertical(self._tilt_angle)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

        agent_data = self._sim.get_agent_data(0)
        for cam_name in agent_data.articulated_agent.params.cameras:
            agent_data.articulated_agent.params.cameras[
                cam_name
            ].cam_look_at_pos = self.cams_init_pos[cam_name].copy()


@registry.register_task_action
class RearrangeLookDownAction(NavigationMovementAgentAction):
    name: str = "rearrange_look_down"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        agent_data = self._sim.get_agent_data(0)
        self.cams_init_pos = {}

        for cam_name in agent_data.articulated_agent.params.cameras:
            self.cams_init_pos[cam_name] = list(
                agent_data.articulated_agent.params.cameras[cam_name].cam_look_at_pos
            )

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        self._move_camera_vertical(-self._tilt_angle)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

        agent_data = self._sim.get_agent_data(0)
        for cam_name in agent_data.articulated_agent.params.cameras:
            agent_data.articulated_agent.params.cameras[
                cam_name
            ].cam_look_at_pos = self.cams_init_pos[cam_name].copy()


@registry.register_task_action
class DiscreteMoveForward(BaseVelAction):
    """
    The articulated agent base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    def step(self, *args, **kwargs):
        lin_vel = 0.25
        ang_vel = 0

        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()


@registry.register_task_action
class DiscreteTurnLeft(BaseVelAction):
    """
    The articulated agent base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    def step(self, *args, **kwargs):
        lin_vel = 0
        ang_vel = np.pi / 180 * 30

        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()


@registry.register_task_action
class DiscreteTurnRight(BaseVelAction):
    """
    The articulated agent base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    def step(self, *args, **kwargs):
        lin_vel = 0
        ang_vel = -np.pi / 180 * 30

        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()


@registry.register_task_action
class DiscreteTurnLeft(BaseVelAction):
    """
    The articulated agent base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    def step(self, *args, **kwargs):
        lin_vel = 0
        ang_vel = np.pi / 180 * 30

        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()


@registry.register_task_action
class DiscreteMoveGeneric(BaseVelAction):
    """
    The articulated agent base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `allow_dyn_slide` is True
    """

    @property
    def action_space(self):
        return EmptySpace()

    def step(self, *args, **kwargs):
        lin_vel = self._lin_speed * 30 / 0.25
        ang_vel = self._ang_speed

        if not self._allow_back:
            lin_vel = np.maximum(lin_vel, 0)

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()


@registry.register_task_action
class RearrangeCameraZoom(ArticulatedAgentAction):
    name: str = "rearrange_zoom"

    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self._zoom_amount = config.zoom_amount

    def _zoom(self, amount: Optional[float] = None):
        sensors_info = self._sim.agents[0]._sensors
        for cam in sensors_info.values():
            if amount is None:
                cam.reset_zoom()
            else:
                cam.zoom(amount)

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        self._zoom(self._zoom_amount)
