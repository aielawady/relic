#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
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
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from relic.policies.transformer_wrappers import (
    TransformerWrapper,
)

from relic.policies.visual_encoders import Vc1Wrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.utils.timing import g_timer
from vc_models.models.compression_layer import create_compression_layer

try:
    import clip
except ImportError:
    clip = None


@baseline_registry.register_policy
class PointNavResNetTransformerPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        transformer_config,
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
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone; one of ["resnet18", "resnet50", "resneXt50", "se_resnet50", "se_resneXt50", "se_resneXt101", "resnet50_clip_avgpool", "resnet50_clip_attnpool"]
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
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                transformer_config=transformer_config,
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

        self.setup_precision(policy_config)

    def setup_precision(self, policy_config):
        def str2torch_dtype(dtype):
            return getattr(torch, dtype)

        precision_config = policy_config.training_precision_config
        self.to(str2torch_dtype(precision_config.others))
        self.net.visual_encoder.to(str2torch_dtype(precision_config.visual_encoder))
        self.critic.to(str2torch_dtype(precision_config.heads))
        self.action_distribution.to(str2torch_dtype(precision_config.heads))

    @torch.autocast("cuda")
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs,
    ):
        features, rnn_hidden_states, _, *output = self.net(
            observations, rnn_hidden_states, prev_actions, masks, **kwargs
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        if output:
            return (
                PolicyActionData(
                    values=value,
                    actions=action,
                    action_log_probs=action_log_probs,
                    rnn_hidden_states=rnn_hidden_states,
                ),
                output[0],
            )
        else:
            return PolicyActionData(
                values=value,
                actions=action,
                action_log_probs=action_log_probs,
                rnn_hidden_states=rnn_hidden_states,
            )

    @property
    def context_len(self):
        return self.net.context_len

    @property
    def memory_size(self):
        return self.net.memory_size

    @property
    def banded_attention(self):
        return self.net.banded_attention

    @property
    def add_context_loss(self):
        return self.net.add_context_loss

    @property
    def num_heads(self):
        return self.net.num_heads

    @torch.autocast("cuda")
    @g_timer.avg_time("net_policy.get_value", level=1)
    def get_value(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
        full_rnn_state=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info=rnn_build_seq_info,
            full_rnn_state=full_rnn_state,
        )
        if full_rnn_state:
            return self.critic(features), rnn_hidden_states
        else:
            return self.critic(features)

    @torch.autocast("cuda")
    @g_timer.avg_time("net_policy.get_value_action_prob", level=1)
    def get_value_action_prob(
        self,
        observations,
        rnn_hidden_states,
        actions,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
        full_rnn_state=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info=rnn_build_seq_info,
            full_rnn_state=full_rnn_state,
        )
        distribution = self.action_distribution(features)
        action_log_probs = distribution.log_probs(actions)

        if full_rnn_state:
            return self.critic(features), action_log_probs, rnn_hidden_states
        else:
            return self.critic(features), action_log_probs

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
            transformer_config=config.habitat_baselines.rl.policy[
                agent_name
            ].transformer_config,
            vc1_config=config.habitat_baselines.rl.policy[agent_name].vc1_config,
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

    @torch.autocast("cuda")
    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
        return_logits=False,
    ):
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        batch = dict(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            action=action,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        aux_loss_res = {
            k: v(aux_loss_state, batch) for k, v in self.aux_loss_modules.items()
        }

        return (
            value,
            action_log_probs if not return_logits else distribution.logits,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )

    def policy_parameters(self) -> Iterable[torch.Tensor]:
        for c in [self.net, self.action_distribution]:
            yield from c.parameters()


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        gradient_checkpointing: bool = False,
        append_global_avg_pool: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.append_global_avg_pool = append_global_avg_pool
        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]
        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            spatial_size_h = observation_space.spaces[self.visual_keys[0]].shape[0]
            spatial_size_w = observation_space.spaces[self.visual_keys[0]].shape[1]
            self.backbone = make_backbone(self._n_input_channels, baseplanes, ngroups)

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial_h * final_spatial_w))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @torch.autocast("cuda")
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = obs_k * self.key_needs_rescaling[k]  # normalize
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        # x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)

        def fwd_func(x):
            x = self.backbone(x)
            x = self.compression(x)
            return x

        output = torch.zeros(
            (x.shape[0], *self.output_shape), dtype=x.dtype, device=x.device
        )
        if self.gradient_checkpointing and self.training:
            for i in range(0, x.shape[0], 2048):
                output[i : i + 2048] = torch.utils.checkpoint.checkpoint(
                    fwd_func, x[i : i + 2048], use_reentrant=False
                )
        else:
            for i in range(0, x.shape[0], 2048):
                output[i : i + 2048] = fwd_func(x[i : i + 2048])
        return output


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        pooling="attnpool",
    ):
        super().__init__()

        self.head_rgb = "head_rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if not self.is_blind:
            if clip is None:
                raise ImportError(
                    "Need to install CLIP (run `pip install git+https://github.com/openai/CLIP.git@40f5484c1c74edd83cb9cf687c6ab92b28d8b656`)"
                )

            model, preprocess = clip.load("RN50")

            # expected input: C x H x W (np.uint8 in [0-255])
            self.preprocess = T.Compose(
                [
                    # resize and center crop to 224
                    preprocess.transforms[0],
                    preprocess.transforms[1],
                    # already tensor, but want float
                    T.ConvertImageDtype(torch.float),
                    # normalize with CLIP mean, std
                    preprocess.transforms[4],
                ]
            )
            # expected output: C x H x W (np.float32)

            self.backbone = model.visual

            if self.head_rgb and self.depth:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048,)  # type: Tuple
            elif pooling == "none":
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif pooling == "avgpool":
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048,)
            else:
                self.output_shape = (1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self.head_rgb:
            head_rgb_observations = observations["head_rgb"]
            head_rgb_observations = head_rgb_observations.permute(
                0, 3, 1, 2
            )  # BATCH x CHANNEL x HEIGHT X WIDTH
            head_rgb_observations = torch.stack(
                [
                    self.preprocess(head_rgb_image)
                    for head_rgb_image in head_rgb_observations
                ]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            head_rgb_x = self.backbone(head_rgb_observations).float()
            cnn_input.append(head_rgb_x)

        if self.depth:
            depth_observations = observations["depth"][
                ..., 0
            ]  # [BATCH x HEIGHT X WIDTH]
            ddd = torch.stack(
                [depth_observations] * 3, dim=1
            )  # [BATCH x 3 x HEIGHT X WIDTH]
            ddd = torch.stack(
                [
                    self.preprocess(TF.convert_image_dtype(depth_map, torch.uint8))
                    for depth_map in ddd
                ]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            depth_x = self.backbone(ddd).float()
            cnn_input.append(depth_x)

        if self.head_rgb and self.depth:
            x = F.adaptive_avg_pool2d(cnn_input[0] + cnn_input[1], 1)
            x = x.flatten(1)
        else:
            x = torch.cat(cnn_input, dim=1)

        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        transformer_config,
        vc1_config,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        gradient_checkpointing: bool = False,
        append_global_avg_pool: bool = False,
    ):
        super().__init__()
        hidden_size = transformer_config.n_hidden

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
                gradient_checkpointing=gradient_checkpointing,
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
        self.state_encoder = TransformerWrapper(
            (
                0
                if self.is_blind
                else (
                    self._hidden_size
                    + (
                        self.visual_encoder.feats_size
                        if self.append_global_avg_pool
                        else 0
                    )
                )
            )
            + rnn_input_size,
            config=transformer_config,
        )
        if gradient_checkpointing:
            self.state_encoder.gradient_checkpointing_enable()
        # self.state_encoder = build_rnn_state_encoder(
        #     (0 if self.is_blind else self._hidden_size) + rnn_input_size,
        #     self._hidden_size,
        #     rnn_type=rnn_type,
        #     num_layers=num_recurrent_layers,
        # )

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
    def num_heads(self):
        return self.state_encoder.n_head

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def context_len(self):
        return self.state_encoder.context_len

    @property
    def memory_size(self):
        return self.state_encoder.memory_size

    @property
    def banded_attention(self):
        return self.state_encoder.banded_attention

    @property
    def add_context_loss(self):
        return self.state_encoder.add_context_loss

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    @torch.autocast("cuda")
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
        full_rnn_state=False,
        **kwargs,
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
            x.append(fuse_states)

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

        if "one_hot_target_sensor" in observations:
            object_goal = observations["one_hot_target_sensor"]
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

        if len(masks.shape) == 3:
            act_mask = masks[:, -1]
        else:
            act_mask = masks

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(act_mask.view(-1), prev_actions + 1, start_token).long()
            )
        else:
            prev_actions = self.prev_action_embedding(act_mask * prev_actions)

        x.append(prev_actions)

        out = torch.cat(x, dim=1) + target_embs
        out, rnn_hidden_states, *output = self.state_encoder(
            out,
            rnn_hidden_states,
            masks,
            rnn_build_seq_info,
            full_rnn_state=full_rnn_state,
            **kwargs,
        )
        aux_loss_state["rnn_output"] = out
        return out, rnn_hidden_states, aux_loss_state, *output
