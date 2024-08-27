from itertools import chain
import warnings

import numpy as np
import torch
import torch.nn as nn
import transformers
from habitat import logger
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder

from relic.policies.llamarl.configuration_llamarl import (
    LlamaRLConfig,
)
from relic.policies.llamarl.modeling_llamarl import LlamaRLModel

# from transformers import TransfoXLModel, TransfoXLConfig
from transformers import TransfoXLConfig
from relic.policies.transformerxl.modeling_transformerxl import TransfoXLModel


class TransformerWrapper(nn.Module):
    def __init__(self, input_size: int, config):
        super().__init__()
        self.model_name = config.model_name
        self.inter_episodes_attention = config.inter_episodes_attention
        self.reset_position_index = config.reset_position_index
        self.add_sequence_idx_embed = config.add_sequence_idx_embed
        self.n_layers = config.n_layers
        self.n_embed = config.n_hidden
        self.n_mlp_hidden = config.n_mlp_hidden
        self.n_head = config.n_heads
        self.activation = config.activation
        self.position_embed_type = config.position_embed_type
        self.sequence_embed_type = config.sequence_embed_type
        self.depth_dropout_p = config.depth_dropout_p
        self.gated_residual = config.gated_residual
        self.add_sink_kv = config.add_sink_kv
        self.mul_factor_for_sink_attn = config.mul_factor_for_sink_attn
        self.is_sink_v_trainable = config.is_sink_v_trainable
        self.is_sink_k_trainable = config.get("is_sink_k_trainable", True)
        self.add_sink_tokens = config.add_sink_tokens
        self.num_sink_tokens = config.num_sink_tokens

        self.context_len = config.context_len
        self.mem_len = config.get("mem_len", -1)
        self.banded_attention = config.banded_attention
        self.orphan_steps_attention = config.orphan_steps_attention
        self.add_context_loss = config.add_context_loss
        self.max_position_embeddings = config.max_position_embeddings
        self.feats_proj = nn.Linear(input_size, self.n_embed)
        self.feats_out = nn.Linear(self.n_embed, self.n_embed)

        if self.model_name == "gpt":
            self.hf_config = transformers.GPT2Config(
                vocab_size=0,
                n_embd=self.n_embed,
                n_layer=self.n_layers,
                n_head=self.n_head,
            )

            self.model = transformers.GPT2Model(self.hf_config)
            self.model.wte.weight.requires_grad_(False)
        elif self.model_name == "llamarl":
            self.hf_config = LlamaRLConfig(
                hidden_size=self.n_embed,
                intermediate_size=self.n_mlp_hidden,
                num_hidden_layers=self.n_layers,
                num_attention_heads=self.n_head,
                hidden_act=self.activation,
                inter_episodes_attention=self.inter_episodes_attention,
                reset_position_index=self.reset_position_index,
                add_sequence_idx_embed=self.add_sequence_idx_embed,
                position_embed_type=self.position_embed_type,
                gated_residual=self.gated_residual,
                context_len=self.context_len,
                banded_attention=self.banded_attention,
                orphan_steps_attention=self.orphan_steps_attention,
                depth_dropout_p=self.depth_dropout_p,
                max_position_embeddings=self.max_position_embeddings,
                add_sink_kv=self.add_sink_kv,
                mul_factor_for_sink_attn=self.mul_factor_for_sink_attn,
                is_sink_v_trainable=self.is_sink_v_trainable,
                is_sink_k_trainable=self.is_sink_k_trainable,
                add_sink_tokens=self.add_sink_tokens,
                num_sink_tokens=self.num_sink_tokens,
                sequence_embed_type=self.sequence_embed_type,
            )

            self.model = LlamaRLModel(self.hf_config)
        elif self.model_name == "transformerxl":
            self.hf_config = TransfoXLConfig(
                d_model=self.n_embed,
                d_embed=self.n_embed,
                n_head=self.n_head,
                d_inner=self.n_mlp_hidden,
                pre_lnorm=True,
                n_layer=self.n_layers,
                mem_len=self.mem_len,
                dropout=self.depth_dropout_p,
            )
            self.model = TransfoXLModel(self.hf_config)
        else:
            raise ValueError(f"Unrecognized {self.model_name}")

        logger.info(f"Done loading llama")

    def postprocess_past_key_value(self, past_key_values, full_rnn_state=False):
        # past_key_values.shape -> [nL, 2(k and v), bs(nE), nH, nS, nE/nH]
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])
        if not full_rnn_state:
            return past_key_values.permute(2, 0, 1, 3, 4, 5)[..., -1, :].flatten(2, 4)
        else:
            return past_key_values.permute(4, 2, 0, 1, 3, 5).flatten(3, 5)

    def stack_past_key_values(self, past_key_values, last_step=False):
        if past_key_values is None:
            return None
        if self.model_name == "transformerxl":
            return torch.stack(past_key_values, dim=0)
        if last_step:
            past_key_values = torch.stack(
                [torch.stack([y[..., -1, :] for y in x]) for x in past_key_values]
            )
        else:
            past_key_values = torch.stack([torch.stack(x) for x in past_key_values])
        return past_key_values

    def preprocess_past_key_value(self, past_key_values):
        # past_key_values.shape -> [nS, bs, nL, 2*nH*nE/nH]
        bs, nS, nL, _ = past_key_values.shape
        nH = self.n_head
        nE = self.n_embed
        return past_key_values.reshape(bs, nS, nL, 2, nH, nE // nH).permute(
            2, 3, 0, 4, 1, 5
        )

    def forward(
        self,
        feats,
        rnn_hidden_states,
        masks,
        rnn_build_seq_info,
        full_rnn_state=False,
        **kwargs,
    ):
        if rnn_build_seq_info is None:
            past_key_values = (
                rnn_hidden_states if np.prod(rnn_hidden_states.shape) > 0 else None
            )
            n_envs = rnn_hidden_states.shape[2]
            seq_len = 1
            masks = masks.squeeze(-1).float()
            stop_grad_steps = 0
            use_cache = True
        else:
            n_envs, seq_len = rnn_build_seq_info["dims"]
            if self.model_name == "transformerxl":
                past_key_values = rnn_hidden_states
                past_key_values = (
                    past_key_values.squeeze(-1)
                    .unflatten(0, (self.num_recurrent_layers, self.memory_size))
                    .float()
                )
                use_cache = False
            else:
                past_key_values = None
                use_cache = full_rnn_state
            masks = masks.squeeze(-1).unflatten(0, (n_envs, seq_len)).float()
            if "stop_grad_steps" in rnn_build_seq_info:
                stop_grad_steps = rnn_build_seq_info["stop_grad_steps"]
            else:
                stop_grad_steps = 0

        feats = feats.unflatten(0, (n_envs, seq_len))

        if rnn_build_seq_info is not None:
            old_context_length = rnn_build_seq_info["old_context_length"]
        else:
            old_context_length = 0

        feats = torch.concat(
            [
                feats[:, :old_context_length].detach(),
                feats[:, old_context_length:],
            ],
            dim=1,
        )

        if (
            rnn_build_seq_info is not None
            and not rnn_build_seq_info["is_first"]
            and not self.add_context_loss
        ):
            feats = torch.concat(
                [
                    feats[:, : rnn_build_seq_info["old_context_length"]].detach(),
                    feats[:, rnn_build_seq_info["old_context_length"] :],
                ],
                dim=1,
            )

        if stop_grad_steps:
            feats_ = feats[:, :stop_grad_steps].detach()
            masks_ = masks[:, :stop_grad_steps].detach()
            feats = feats[:, stop_grad_steps:]

            # TODO check why torch.no_grad doesn't work.
            feats_ = self.feats_proj(feats_)
            output_ = self.model(
                inputs_embeds=feats_,
                past_key_values=None,
                attention_mask=masks_,
            )
            feats_ = output_.last_hidden_state
            feats_ = self.feats_out(feats_)

            past_key_values = output_.past_key_values

            feats_ = feats_.detach()
            past_key_values = self.stack_past_key_values(past_key_values).detach()

        feats = self.feats_proj(feats)
        if self.model_name == "transformerxl":
            output = self.model(
                inputs_embeds=feats,
                mems=past_key_values,
                # attention_mask=masks,
                # use_cache=use_cache,
                # **kwargs
            )
            output.past_key_values = output.mems
        else:
            output = self.model(
                inputs_embeds=feats,
                past_key_values=past_key_values,
                attention_mask=masks,
                use_cache=use_cache,
                **kwargs,
            )

        feats = output.last_hidden_state
        feats = self.feats_out(feats)

        if (
            rnn_build_seq_info is not None
            and not rnn_build_seq_info["is_first"]
            and not self.add_context_loss
        ):
            feats = feats[:, rnn_build_seq_info["old_context_length"] :]

        if stop_grad_steps:
            feats = torch.concat([feats_, feats], dim=1)
        feats = feats.flatten(0, 1)
        if kwargs:
            return (
                feats,
                self.stack_past_key_values(
                    output.past_key_values,
                    last_step=self.model_name != "transformerxl" and not full_rnn_state,
                ),
                output,
            )
        else:
            return feats, self.stack_past_key_values(
                output.past_key_values,
                last_step=self.model_name != "transformerxl" and not full_rnn_state,
            )

    def get_trainable_params(self):
        return chain(
            self.feats_proj.parameters(),
            self.model.named_parameters(),
            self.feats_out.parameters(),
        )

    @property
    def num_recurrent_layers(self):
        return self.n_layers

    @property
    def recurrent_hidden_size(self):
        return self.n_embed

    @property
    def memory_size(self):
        return self.mem_len

    def gradient_checkpointing_enable(self):
        return self.model.gradient_checkpointing_enable()
