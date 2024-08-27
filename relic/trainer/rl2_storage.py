#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Iterator, Optional

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)
from habitat_baselines.utils.common import get_action_space_info


@baseline_registry.register_storage
class RL2RolloutStorage(RolloutStorage):
    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        actor_critic,
        is_double_buffered: bool = False,
        change_done_masks: bool = True,
        set_done_to_false_during_rollout: bool = False,
    ):
        self.change_done_masks = change_done_masks
        self.set_done_to_false_during_rollout = set_done_to_false_during_rollout
        action_shape, discrete_actions = get_action_space_info(action_space)

        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            actor_critic.num_recurrent_layers,
            actor_critic.recurrent_hidden_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(numsteps + 1, num_envs, 1)

        if action_shape is None:
            action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(numsteps + 1, num_envs, *action_shape)
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if discrete_actions:
            assert isinstance(self.buffers["actions"], torch.Tensor)
            assert isinstance(self.buffers["prev_actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(numsteps + 1, num_envs, 1, dtype=torch.bool)

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.num_steps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

        # The default device to torch is the CPU, so everything is on the CPU.
        self.device = torch.device("cpu")

    def reset_recurrent_hidden_states(self):
        self.buffers[0]["recurrent_hidden_states"] = torch.zeros_like(
            self.buffers[0]["recurrent_hidden_states"]
        )

    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
    ) -> Iterator[DictTree]:
        assert isinstance(self.buffers["returns"], torch.Tensor)
        num_environments = self.buffers["returns"].size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_environments, num_mini_batch)
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        dones_cpu = (
            torch.logical_not(self.buffers["masks"])
            .cpu()
            .view(-1, self._num_envs)
            .numpy()
        )
        if self.change_done_masks and not self.set_done_to_false_during_rollout:
            dones_cpu = np.zeros_like(dones_cpu, dtype=bool)

        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            curr_slice = (slice(0, self.current_rollout_step_idx), inds)

            batch = self.buffers[curr_slice]
            if advantages is not None:
                batch["advantages"] = advantages[curr_slice]
            batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]

            batch.map_in_place(lambda v: v.flatten(0, 1))

            batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                device=self.device,
                build_fn_result=build_pack_info_from_dones(
                    dones_cpu[0 : self.current_rollout_step_idx, inds.numpy()].reshape(
                        -1, len(inds)
                    ),
                ),
            )

            yield batch.to_tree()
