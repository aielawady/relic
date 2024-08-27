"""
Contains the structured config definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from habitat.config.default_structured_configs import (
    LabSensorConfig,
    MeasurementConfig,
    TaskConfig,
    BaseVelocityActionConfig,
    HabitatSimSemanticSensorConfig,
    SimulatorCameraSensorConfig,
    PointGoalSensorConfig,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesBaseConfig,
    HabitatBaselinesRLConfig,
    PolicyConfig,
    PPOConfig,
    DDPPOConfig,
    RLConfig,
)
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from relic.trainer.transformers_agent_access_mgr import TransformerSingleAgentAccessMgr
from habitat.config.default_structured_configs import ActionConfig, HabitatBaseConfig

cs = ConfigStore.instance()


##########################################################################
# Trainer
##########################################################################


@dataclass
class TransformerConfig(HabitatBaselinesBaseConfig):
    model_name: str = "llamarl"
    n_layers: int = 24
    n_heads: int = 16
    n_hidden: int = 2048
    n_mlp_hidden: int = 8192
    kv_size: int = 128
    activation: str = "gelu_new"
    depth_dropout_p: float = 0.0
    inter_episodes_attention: bool = False
    reset_position_index: bool = True
    add_sequence_idx_embed: bool = False
    sequence_embed_type: str = "learnable"
    position_embed_type: str = "rope"
    # from [STABILIZING TRANSFORMERS FOR REINFORCEMENT LEARNING](https://arxiv.org/pdf/1910.06764.pdf)
    gated_residual: bool = False
    # The length of history prepended to the input batch
    context_len: int = 0
    # Force tokens to attend to at most `context_len` tokens
    banded_attention: bool = False
    # Don't process time steps of episodes that didn't start in the batch
    orphan_steps_attention: bool = True
    # Whether to include the context tokens in the loss or not
    add_context_loss: bool = False
    max_position_embeddings: int = 2048

    add_sink_tokens: bool = False

    add_sink_kv: bool = False
    mul_factor_for_sink_attn: bool = True
    is_sink_v_trainable: bool = True
    is_sink_k_trainable: bool = True

    num_sink_tokens: int = 1

    mem_len: int = -1


@dataclass
class VC1Config(HabitatBaselinesBaseConfig):
    avg_pool_size: int = 2
    is_2d_output: bool = False


@dataclass
class TrainingPrecisionConfig(HabitatBaselinesBaseConfig):
    visual_encoder: str = "float32"
    heads: str = "float32"
    others: str = "float32"


@dataclass
class CustomPolicyConfig(PolicyConfig):
    transformer_config: TransformerConfig = field(
        default_factory=lambda: TransformerConfig()
    )
    vc1_config: VC1Config = VC1Config()
    training_precision_config: TrainingPrecisionConfig = TrainingPrecisionConfig()


@dataclass
class CustomPPOConfig(PPOConfig):
    updates_per_rollout: int = 1
    full_updates_per_rollout: int = 1
    percent_envs_update: Optional[float] = None
    slice_in_partial_update: bool = False
    update_stale_kv: bool = False
    update_stale_values: bool = False
    update_stale_action_probs: bool = False
    shuffle_old_episodes: bool = False
    shift_scene_every: int = 0
    shift_scene_staggered: bool = True
    force_env_reset_every: int = -1

    context_len: int = 32
    skipgrad: bool = False
    skipgrad_factor1: float = 0.1
    skipgrad_factor2: int = 2

    # Not used at the moment
    optimizer_name: str = "adam"
    adamw_weight_decay: float = 0.01

    lr_scheduler: str = ""
    warmup: bool = False
    warmup_total_iters: int = 300
    warmup_start_factor: float = 0.3
    warmup_end_factor: float = 1

    initial_lr: float = 1e-7
    lr_scheduler_restart_step: int = 5_000_000
    lrsched_T_0: int = 2500
    lrsched_T_mult: int = 1
    lrsched_T_max: int = 2500
    lrsched_eta_min: float = 0

    grad_accum_mini_batches: int = 1
    storage_low_precision: bool = False
    ignore_old_obs_grad: bool = False
    gradient_checkpointing: bool = False

    acting_context: Optional[int] = None

    shortest_path_follower: bool = False
    init_checkpoint: str = ""
    append_global_avg_pool: bool = False


@dataclass
class CustomPPORL2Config(CustomPPOConfig):
    change_done_masks: bool = True
    set_done_to_false_during_rollout: bool = True
    eval_use_rl2_modifications: bool = True


@dataclass
class CustomTaskConfig(TaskConfig):
    start_template: Optional[List[str]] = MISSING
    goal_template: Optional[Dict[str, Any]] = MISSING
    sample_entities: Dict[str, Any] = MISSING
    fix_position_same_episode: bool = False
    fix_target_same_episode: bool = False
    fix_instance_index: bool = False
    target_sampling_strategy: str = "object_type"  # object_instance or object_type
    target_type: str = "object_type"  # object_instance or object_type

    cleanup_nav_points: bool = False
    strict_target_selection: bool = False
    max_n_targets_per_sequence: int = -1

    one_receptacle: bool = False
    is_large_objs: bool = False
    goal_point_from_snapped: bool = False
    max_num_start_pos: int = -1

    make_env_fn: str = "make_gym_from_config"


cs.store(
    package="habitat.task",
    group="habitat/task",
    name="custom_task_config_base",
    node=CustomTaskConfig,
)


@dataclass
class CustomRLConfig(RLConfig):
    ppo: CustomPPOConfig = CustomPPOConfig()
    policy: Dict[str, Any] = field(
        default_factory=lambda: {"main_agent": CustomPolicyConfig()}
    )


@dataclass
class CustomRL2Config(CustomRLConfig):
    ppo: CustomPPORL2Config = CustomPPORL2Config()


@dataclass
class CustomHabitatBaselinesRLConfig(HabitatBaselinesRLConfig):
    reset_envs_after_update: bool = False
    call_after_update_env: bool = False
    separate_envs_and_policy: bool = False
    separate_rollout_and_policy: bool = False
    rollout_on_cpu: bool = False
    eval_data_dir: str = MISSING
    rl: CustomRLConfig = CustomRLConfig()


@dataclass
class CustomHabitatBaselinesRL2Config(CustomHabitatBaselinesRLConfig):
    rl: CustomRL2Config = CustomRL2Config()


@dataclass
class CustomPddlTaskSuccessConfig(MeasurementConfig):
    type: str = "PddlTaskSuccess"
    must_call_stop: bool = False
    must_see_object: bool = False
    max_angle: float = 360
    sees_vertical_margin: int = 5
    sees_horizontal_margin: int = 5
    ignore_objects: bool = False
    ignore_receptacles: bool = False
    ignore_non_negative: bool = False
    pixels_threshold: int = 10


@dataclass
class CustomPddlTaskRewardConfig(MeasurementConfig):
    type: str = "NamedNavToObjReward"
    # reward the agent for facing the object?
    should_reward_turn: bool = False
    # what distance do we start giving the reward for facing the object?
    turn_reward_dist: float = 3.0
    # multiplier on the angle distance to the goal.
    angle_dist_reward: float = 0.0
    dist_reward: float = 0.0
    dist_reward_pow: float = 1
    constraint_violate_pen: float = 1.0
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    bad_term_pen: float = 1.0
    end_on_bad_termination: bool = False
    max_reward_dist: float = -1
    use_max_dist: bool = False


@dataclass
class GeoDiscDistanceConfig(MeasurementConfig):
    type: str = "GeoDiscDistance"
    lock_closest_object: bool = False
    add_point2object_dst: bool = False


@dataclass
class L2DistanceConfig(MeasurementConfig):
    type: str = "L2Distance"


@dataclass
class RotDistToClosestGoalConfig(MeasurementConfig):
    type: str = "RotDistToClosestGoal"


@dataclass
class CamRotDistToClosestGoalConfig(MeasurementConfig):
    type: str = "CamRotDistToClosestGoal"


@dataclass
class SPLGeodiscMeasurementConfig(MeasurementConfig):
    r"""
    For Navigation tasks only, Measures the SPL (Success weighted by Path Length)
    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    Measure is always 0 except at success where it will be
    the ratio of the optimal distance from start to goal over the total distance
    traveled by the agent. Maximum value is 1.
    SPL = success * optimal_distance_to_goal / distance_traveled_so_far
    """
    type: str = "SPLGeodisc"


@dataclass
class SoftSPLGeodiscMeasurementConfig(MeasurementConfig):
    r"""
    For Navigation tasks only, Similar to SPL, but instead of a boolean,
    success is now calculated as 1 - (ratio of distance covered to target).
    SoftSPL = max(0, 1 - distance_to_goal / optimal_distance_to_goal) * optimal_distance_to_goal / distance_traveled_so_far
    """
    type: str = "SoftSPLGeodisc"


@dataclass
class HeadSimSemanticSensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "head_semantic"
    width: int = 256
    height: int = 256


@dataclass
class RearrangeDiscreteNavigationActionConfig(ActionConfig):
    tilt_angle: int = 15  # angle to tilt the camera up or down in degrees


@dataclass
class RearrangeMoveForwardActionConfig(RearrangeDiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will move the robot forward by
    a fixed amount determined by the SimulatorConfig.forward_step_size amount.
    """
    type: str = "RearrangeMoveForwardAction"


@dataclass
class RearrangeTurnLeftActionConfig(RearrangeDiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot to the left
    by a fixed amount determined by the SimulatorConfig.turn_angle amount.
    """
    type: str = "RearrangeTurnLeftAction"


@dataclass
class RearrangeTurnRightActionConfig(RearrangeDiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot to the right
    by a fixed amount determined by the SimulatorConfig.turn_angle amount.
    """
    type: str = "RearrangeTurnRightAction"


@dataclass
class RearrangeLookUpActionConfig(RearrangeDiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera up
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "RearrangeLookUpAction"


@dataclass
class RearrangeLookDownActionConfig(RearrangeDiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "RearrangeLookDownAction"


@dataclass
class MoveForwardActionConfig(BaseVelocityActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "DiscreteMoveGeneric"
    lin_speed: float = 0.15
    ang_speed: float = 0


@dataclass
class MoveBackwardActionConfig(BaseVelocityActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "DiscreteMoveGeneric"
    lin_speed: float = -0.1
    ang_speed: float = 0


@dataclass
class TurnLeftActionConfig(BaseVelocityActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "DiscreteMoveGeneric"
    lin_speed: float = 0.0
    ang_speed: float = 15


@dataclass
class TurnRightActionConfig(BaseVelocityActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "DiscreteMoveGeneric"
    lin_speed: float = 0.0
    ang_speed: float = -15


@dataclass
class ZoomInActionConfig(ActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "RearrangeCameraZoom"
    zoom_amount: float = 1.1


@dataclass
class ZoomResetActionConfig(ActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "RearrangeCameraZoom"
    zoom_amount: Optional[float] = None


@dataclass
class OneHotTargetSensorConfig(LabSensorConfig):
    type: str = "OneHotTargetSensor"


@dataclass
class OneHotReceptacleSensorConfig(LabSensorConfig):
    type: str = "OneHotReceptacleSensor"


@dataclass
class MaskedSemanticSensorConfig(LabSensorConfig):
    type: str = "MaskedSemanticSensor"
    width: int = 256
    height: int = 256


@dataclass
class MaskedFlattenedSemanticSensorConfig(LabSensorConfig):
    type: str = "MaskedFlattenedSemanticSensor"
    n_cells: int = 16


@dataclass
class RelativeLocalizationSensorConifg(LabSensorConfig):
    type: str = "RelativeLocalizationSensor"


@dataclass
class PointGoalWithGPSCompassSensorV3Config(PointGoalSensorConfig):
    """
    Indicates the position of the point goal in the frame of reference of the robot.
    """

    type: str = "PointGoalWithGPSCompassSensorV3"
    std_noise_1m: float = 0.0


cs.store(
    package="habitat.task.lab_sensors.pointgoal_with_gps_compass_v3_sensor",
    group="habitat/task/lab_sensors",
    name="pointgoal_with_gps_compass_v3_sensor",
    node=PointGoalWithGPSCompassSensorV3Config,
)


cs.store(
    package="habitat.task.lab_sensors.one_hot_target_sensor",
    group="habitat/task/lab_sensors",
    name="one_hot_target_sensor",
    node=OneHotTargetSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.one_hot_receptacle_sensor",
    group="habitat/task/lab_sensors",
    name="one_hot_receptacle_sensor",
    node=OneHotReceptacleSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.masked_semantic_sensor",
    group="habitat/task/lab_sensors",
    name="masked_semantic_sensor",
    node=MaskedSemanticSensorConfig,
)


cs.store(
    package="habitat.task.lab_sensors.masked_flattened_semantic_sensor",
    group="habitat/task/lab_sensors",
    name="masked_flattened_semantic_sensor",
    node=MaskedFlattenedSemanticSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.rel_localization_sensor",
    group="habitat/task/lab_sensors",
    name="rel_localization_sensor",
    node=RelativeLocalizationSensorConifg,
)

cs.store(
    package="habitat.task.measurements.spl_geodisc",
    group="habitat/task/measurements",
    name="spl_geodisc",
    node=SPLGeodiscMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.soft_spl_geodisc",
    group="habitat/task/measurements",
    name="soft_spl_geodisc",
    node=SoftSPLGeodiscMeasurementConfig,
)

cs.store(
    group="habitat_baselines/rl/policy",
    name="policy_base",
    node=CustomPolicyConfig,
)

# Register configs to config store
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=CustomHabitatBaselinesRLConfig(),
)

cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl2_config_base",
    node=CustomHabitatBaselinesRL2Config(),
)

cs.store(
    package="habitat.task.measurements.custom_predicate_task_success",
    group="habitat/task/measurements",
    name="extobjnav_success",
    node=CustomPddlTaskSuccessConfig,
)

cs.store(
    package="habitat.task.measurements.custom_predicate_task_reward",
    group="habitat/task/measurements",
    name="named_nav_to_obj_reward",
    node=CustomPddlTaskRewardConfig,
)

cs.store(
    package="habitat.task.measurements.geo_disc_distance",
    group="habitat/task/measurements",
    name="geo_disc_distance",
    node=GeoDiscDistanceConfig,
)

cs.store(
    package="habitat.task.measurements.l2_distance",
    group="habitat/task/measurements",
    name="l2_distance",
    node=L2DistanceConfig,
)

cs.store(
    package="habitat.task.measurements.rot_dist_to_closest_goal",
    group="habitat/task/measurements",
    name="rot_dist_to_closest_goal",
    node=RotDistToClosestGoalConfig,
)

cs.store(
    package="habitat.task.measurements.cam_rot_dist_to_closest_goal",
    group="habitat/task/measurements",
    name="cam_rot_dist_to_closest_goal",
    node=CamRotDistToClosestGoalConfig,
)

cs.store(
    package="habitat.task.actions.rearrange_move_forward",
    group="habitat/task/actions",
    name="rearrange_move_forward",
    node=RearrangeMoveForwardActionConfig,
)
cs.store(
    package="habitat.task.actions.rearrange_turn_left",
    group="habitat/task/actions",
    name="rearrange_turn_left",
    node=RearrangeTurnLeftActionConfig,
)
cs.store(
    package="habitat.task.actions.rearrange_turn_right",
    group="habitat/task/actions",
    name="rearrange_turn_right",
    node=RearrangeTurnRightActionConfig,
)
cs.store(
    package="habitat.task.actions.rearrange_look_up",
    group="habitat/task/actions",
    name="rearrange_look_up",
    node=RearrangeLookUpActionConfig,
)
cs.store(
    package="habitat.task.actions.rearrange_look_down",
    group="habitat/task/actions",
    name="rearrange_look_down",
    node=RearrangeLookDownActionConfig,
)

cs.store(
    package="habitat.task.actions.move_forward_custom",
    group="habitat/task/actions",
    name="move_forward_custom",
    node=MoveForwardActionConfig,
)
cs.store(
    package="habitat.task.actions.move_backward_custom",
    group="habitat/task/actions",
    name="move_backward_custom",
    node=MoveBackwardActionConfig,
)

cs.store(
    package="habitat.task.actions.turn_left_custom",
    group="habitat/task/actions",
    name="turn_left_custom",
    node=TurnLeftActionConfig,
)
cs.store(
    package="habitat.task.actions.turn_right_custom",
    group="habitat/task/actions",
    name="turn_right_custom",
    node=TurnRightActionConfig,
)
cs.store(
    package="habitat.task.actions.zoom_in_custom",
    group="habitat/task/actions",
    name="zoom_in_custom",
    node=ZoomInActionConfig,
)
cs.store(
    package="habitat.task.actions.zoom_reset_custom",
    group="habitat/task/actions",
    name="zoom_reset_custom",
    node=ZoomResetActionConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="head_semantic_sensor",
    node=HeadSimSemanticSensorConfig,
)
