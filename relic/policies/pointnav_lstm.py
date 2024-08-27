#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn

from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet, PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions

if TYPE_CHECKING:
    from omegaconf import DictConfig

try:
    import clip
except ImportError:
    clip = None

from relic.policies.pointnav import ResNetCLIPEncoder, ResNetEncoder, Vc1Wrapper


@baseline_registry.register_policy
class PointNavResNetLstmPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        vc1_config=None,
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        gradient_checkpointing: bool = False,
        append_global_avg_pool: bool = False,
        **kwargs,
    ):
        # TODO: Check if we need gradient checkpointing and append global avg pool
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone; one of [
            "resnet18", "resnet50", "resneXt50", "se_resnet50", "se_resneXt50",
            "se_resneXt101", "resnet50_clip_avgpool", "resnet50_clip_attnpool"
        ]
        """

        assert backbone in [
            "vc1",
            "resnet18",
            "resnet50",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
            "resnet50_clip_avgpool",
            "resnet50_clip_attnpool",
        ] or backbone.startswith("vc1"), f"{backbone} backbone is not recognized."

        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            PointNavResNetLstmNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                vc1_config=vc1_config,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                gradient_checkpointing=gradient_checkpointing,
                append_global_avg_pool=append_global_avg_pool,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    # TODO: Check if we need the precision stuff

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                ((k, v) for k, v in observation_space.items() if k not in ignore_names)
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            # TODO: Check if the following works
            vc1_config=config.habitat_baselines.rl.policy[agent_name].vc1_config,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
            gradient_checkpointing=config.habitat_baselines.rl.ppo.gradient_checkpointing,
            append_global_avg_pool=config.habitat_baselines.rl.ppo.get(
                "append_global_avg_pool", False
            ),
        )


class PointNavResNetLstmNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        vc1_config,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        gradient_checkpointing: bool = False,
        append_global_avg_pool: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.append_global_avg_pool = append_global_avg_pool
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_prev_action = hidden_size
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(num_actions, self._n_prev_action)
        self._n_prev_action = hidden_size
        rnn_input_size = self._n_prev_action  # test

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
                "one_hot_target_sensor",
                "one_hot_receptacle_sensor",
                "localization_sensor",
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0] for k in self._fuse_keys_1d
            )

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32

        if "localization_sensor" in observation_space.spaces:
            n_input_goal = observation_space.spaces["localization_sensor"].shape[0]
            self.locs_embeding = nn.Linear(n_input_goal, hidden_size)
            rnn_input_size += hidden_size

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observation_space.spaces:
                goal_observation_space = spaces.Dict(
                    {"rgb": observation_space.spaces[uuid]}
                )
                goal_visual_encoder = ResNetEncoder(
                    goal_observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(resnet, backbone),
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)

                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(goal_visual_encoder.output_shape), hidden_size),
                    nn.ReLU(True),
                )
                setattr(self, f"{uuid}_fc", goal_visual_fc)

                rnn_input_size += hidden_size

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )
        if backbone.startswith("vc1"):
            if backbone == "vc1":
                model_id = None
            else:
                model_id = backbone.split("_", 1)[1]

            self.visual_encoder = Vc1Wrapper(
                use_obs_space, model_id=model_id, vc1_config=vc1_config
            )
            if gradient_checkpointing:
                self.visual_encoder.set_grad_checkpointing()
            if not self.visual_encoder.is_blind:
                if vc1_config.is_2d_output:
                    # from habitat_baselines.rl.ddppo.policy.resnet import conv1x1, ResNet, BasicBlock
                    # def _make_layer(
                    #     inplanes,
                    #     block,
                    #     ngroups: int,
                    #     planes: int,
                    #     blocks: int,
                    #     stride: int = 1,
                    # ):
                    #     downsample = None
                    #     if stride != 1 or inplanes != planes * block.expansion:
                    #         downsample = nn.Sequential(
                    #             conv1x1(inplanes, planes * block.expansion, stride),
                    #             nn.GroupNorm(ngroups, planes * block.expansion),
                    #         )

                    #     layers = []
                    #     layers.append(
                    #         block(
                    #             inplanes,
                    #             planes,
                    #             ngroups,
                    #             stride,
                    #             downsample,
                    #             cardinality=1,
                    #         )
                    #     )
                    #     inplanes = planes * block.expansion
                    #     for _i in range(1, blocks):
                    #         layers.append(block(inplanes, planes, ngroups))

                    #     return nn.Sequential(*layers)

                    # self.visual_fc = nn.Sequential(
                    #     nn.Unflatten(-1, (self.visual_encoder.feats_size, self.visual_encoder.out_dim, self.visual_encoder.out_dim)),
                    #     _make_layer(
                    #         self.visual_encoder.feats_size,
                    #         BasicBlock,
                    #         16,
                    #         self.visual_encoder.feats_size//2,
                    #         2,
                    #         1
                    #     ),
                    #     _make_layer(
                    #         self.visual_encoder.feats_size//2,
                    #         BasicBlock,
                    #         16,
                    #         self.visual_encoder.feats_size//4,
                    #         2,
                    #         2
                    #     ),
                    #     nn.Flatten(),
                    #     nn.Linear(
                    #         self.visual_encoder.feats_size//4 * round(self.visual_encoder.out_dim/2)**2, hidden_size
                    #     ),
                    #     nn.ReLU(True),
                    # )

                    # ======================

                    # compression, output_shape, output_size = create_compression_layer(
                    #     self.visual_encoder.feats_size,
                    #     self.visual_encoder.out_dim
                    # )

                    # self.visual_fc = nn.Sequential(
                    #     nn.Unflatten(-1, (self.visual_encoder.feats_size, self.visual_encoder.out_dim, self.visual_encoder.out_dim)),
                    #     compression,
                    #     nn.Linear(
                    #         output_size, hidden_size
                    #     ),
                    #     nn.ReLU(True),
                    # )

                    # ======================

                    if self.append_global_avg_pool:
                        self.visual_pooler = nn.Sequential(
                            nn.Unflatten(
                                -1,
                                (
                                    self.visual_encoder.feats_size,
                                    self.visual_encoder.out_dim,
                                    self.visual_encoder.out_dim,
                                ),
                            ),
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                        )

                    self.visual_fc = nn.Sequential(
                        nn.Unflatten(
                            -1,
                            (
                                self.visual_encoder.feats_size,
                                self.visual_encoder.out_dim,
                                self.visual_encoder.out_dim,
                            ),
                        ),
                        nn.Conv2d(
                            self.visual_encoder.feats_size,
                            self.visual_encoder.feats_size // 2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.ReLU(True),
                        nn.Conv2d(
                            self.visual_encoder.feats_size // 2,
                            self.visual_encoder.feats_size // 4,
                            kernel_size=3,
                            stride=(2, 2),
                            padding=1,
                            bias=False,
                        ),
                        nn.GroupNorm(1, self.visual_encoder.feats_size // 4),
                        nn.ReLU(True),
                        nn.Flatten(),
                        nn.Linear(
                            self.visual_encoder.feats_size
                            // 4
                            * round(self.visual_encoder.out_dim / 2) ** 2,
                            hidden_size,
                        ),
                        nn.ReLU(True),
                    )
                else:
                    self.visual_fc = nn.Sequential(
                        nn.Linear(self.visual_encoder.feats_size, hidden_size),
                        nn.ReLU(True),
                    )
        elif backbone.startswith("resnet50_clip"):
            self.visual_encoder = ResNetCLIPEncoder(
                observation_space if not force_blind_policy else spaces.Dict({}),
                pooling="avgpool" if "avgpool" in backbone else "attnpool",
            )
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Linear(self.visual_encoder.output_shape[0], hidden_size),
                    nn.ReLU(True),
                )
        else:
            self.visual_encoder = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
                    nn.ReLU(True),
                )

        if "one_hot_target_sensor" in observation_space.spaces:
            n_input_goal = observation_space.spaces["one_hot_target_sensor"].shape[0]
            self.ohts_embeding = nn.Linear(n_input_goal, hidden_size)
            rnn_input_size += hidden_size

        if "one_hot_receptacle_sensor" in observation_space.spaces:
            n_input_goal = observation_space.spaces["one_hot_receptacle_sensor"].shape[
                0
            ]
            self.ohrs_embeding = nn.Linear(n_input_goal, hidden_size)
            rnn_input_size += hidden_size

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observations
            ):
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)

            if self.append_global_avg_pool:
                gavg_pool = self.visual_pooler(visual_feats)

            visual_feats = self.visual_fc(visual_feats)
            if self.append_global_avg_pool:
                visual_feats = torch.concat([visual_feats, gavg_pool], dim=-1)

            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert goal_observations.shape[1] == 3, "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        with torch.autocast("cuda"):
            if "one_hot_target_sensor" in observations:
                object_goal = observations["one_hot_target_sensor"].float()
                x.append(self.ohts_embeding(object_goal))
                # target_embs = self.ohts_embeding(object_goal)
                target_embs = 0
            else:
                target_embs = 0

        if "one_hot_receptacle_sensor" in observations:
            receptacle_goal = observations["one_hot_receptacle_sensor"]
            x.append(self.ohrs_embeding(receptacle_goal))

        if "localization_sensor" in observations:
            object_goal = observations["localization_sensor"]
            x.append(self.locs_embeding(object_goal))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observations:
                goal_image = observations[uuid]

                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})

                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1) + target_embs
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state
