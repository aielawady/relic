from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesBaseConfig,
)

from relic.policies.llamarl.configuration_llamarl import (
    LlamaRLConfig,
)
from relic.policies.llamarl.modeling_llamarl import LlamaRLModel
from relic.policies.transformer_wrappers import (
    TransformerWrapper,
)


@dataclass
class TransformerConfig(HabitatBaselinesBaseConfig):
    model_name: str = "llamarl"
    n_layers: int = 24
    n_heads: int = 16
    n_hidden: int = 2048
    n_mlp_hidden: int = 8192
    kv_size: int = 128
    activation: str = "gelu_new"
    inter_episodes_attention: bool = False
    reset_position_index: bool = True
    add_sequence_idx_embed: bool = False
    position_embed_type: str = "learnable"
    depth_dropout_p: float = 0.0
    gated_residual: bool = False
    # The length of history prepended to the input batch
    context_len: int = 0
    # Force tokens to attend to at most `context_len` tokens
    banded_attention: bool = False
    # Don't process time steps of episodes that didn't start in the batch
    orphan_steps_attention: bool = True
    # Whether to include the context tokens in the loss or not
    add_context_loss: bool = False


def stack_past_key_values(past_key_values):
    past_key_values = torch.stack([torch.stack(x) for x in past_key_values])
    return past_key_values


def test_llamarl_rope_perm():
    config_kwargs = dict(
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        position_embed_type="rope",
    )

    batch_size = 8
    seq_len = 128
    feats_len = 32
    n_seq = 3
    data = torch.tensor(np.random.randn(batch_size, seq_len, feats_len)).float()
    masks = torch.tensor(np.ones((batch_size, seq_len))).float()
    masks[:, :: seq_len // n_seq] = 0

    keys_to_change = [
        "inter_episodes_attention",
        "reset_position_index",
        "add_sequence_idx_embed",
    ]
    for vals in product([True, False], repeat=3):
        config_kwargs.update(dict(zip(keys_to_change, vals)))
        config = LlamaRLConfig(**config_kwargs)
        model = LlamaRLModel(config)

        with torch.no_grad():
            output_no_perm = model(inputs_embeds=data, attention_mask=masks)

            perms = np.random.permutation(batch_size)
            output_perm = model(
                inputs_embeds=data[perms, ...],
                attention_mask=masks[perms, ...],
            )

            assert (
                output_no_perm.last_hidden_state[perms, ...]
                == output_perm.last_hidden_state
            ).all()

            assert (
                stack_past_key_values(output_no_perm.past_key_values)[:, :, perms]
                == stack_past_key_values(output_perm.past_key_values)
            ).all()


def test_transformerwrapper_rope_perm():
    config_kwargs = dict(
        n_hidden=32,
        n_mlp_hidden=128,
        n_layers=3,
        n_heads=2,
        position_embed_type="rope",
    )

    batch_size = 8
    seq_len = 128
    feats_len = 32
    n_seq = 3
    data = torch.tensor(np.random.randn(batch_size, seq_len, feats_len)).float()
    masks = torch.tensor(np.ones((batch_size, seq_len))).float()
    masks[:, :: seq_len // n_seq] = 0
    masks = masks

    rnn_build_seq_info = {
        "dims": (batch_size, seq_len),
        "is_first": False,
        "old_context_length": 0,
    }

    keys_to_change = [
        "inter_episodes_attention",
        "reset_position_index",
        "add_sequence_idx_embed",
    ]
    for vals in product([True, False], repeat=3):
        config_kwargs.update(dict(zip(keys_to_change, vals)))
        config = TransformerConfig(**config_kwargs)
        model = TransformerWrapper(32, config)

        with torch.no_grad():
            output_no_perm, pkv_no_perm = model(
                data.flatten(0, 1),
                None,
                masks.flatten(0, 1),
                rnn_build_seq_info,
            )

            perms = np.random.permutation(batch_size)
            output_perm, pkv_perm = model(
                data[perms, ...].flatten(0, 1),
                None,
                masks[perms, ...].flatten(0, 1),
                rnn_build_seq_info,
            )

            assert (
                output_no_perm.unflatten(0, (batch_size, seq_len))[perms, :]
                == output_perm.unflatten(0, (batch_size, seq_len))
            ).all()

            assert (pkv_no_perm[:, :, perms, :] == pkv_perm).all()


def test_transformerwrapper_rope_roll():
    config_kwargs = dict(
        n_hidden=32,
        n_mlp_hidden=128,
        n_layers=3,
        n_heads=2,
        position_embed_type="rope",
    )

    batch_size = 8
    seq_len = 128
    feats_len = 32
    n_seq = 4
    data = torch.tensor(np.random.randn(seq_len, batch_size, feats_len)).float()
    masks = torch.tensor(np.ones((seq_len, batch_size))).float()
    masks[:: seq_len // n_seq, :] = 0
    masks = masks

    assert (masks.roll((seq_len // n_seq), 0) == masks).all(), masks

    keys_to_change = [
        "inter_episodes_attention",
        "reset_position_index",
        "add_sequence_idx_embed",
    ]
    for vals in [(False, True, False), (False, False, False)]:
        config_kwargs.update(dict(zip(keys_to_change, vals)))
        config = TransformerConfig(**config_kwargs)

        model = TransformerWrapper(32, config)
        model.eval()

        rnn_build_seq_info = {
            "dims": (batch_size, seq_len),
            "is_first": False,
            "old_context_length": 0,
        }

        with torch.no_grad():
            output_no_roll, pkv_no_roll = model(
                data.flatten(0, 1),
                None,
                masks.flatten(0, 1),
                rnn_build_seq_info,
            )

            output_roll, pkv_roll = model(
                data.roll(seq_len // n_seq, 0).flatten(0, 1),
                None,
                masks.roll(seq_len // n_seq, 0).flatten(0, 1),
                rnn_build_seq_info,
            )

            assert np.allclose(
                output_no_roll.unflatten(0, (seq_len, batch_size)).roll(
                    (seq_len // n_seq), 0
                ),
                output_roll.unflatten(0, (seq_len, batch_size)),
            ), vals


def test_attention_masks():
    config_kwargs = dict(
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        position_embed_type="rope",
        reset_position_index=True,
        add_sequence_idx_embed=False,
        gated_residual=False,
    )

    masks = torch.tensor(
        [
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    data = torch.zeros_like(masks)

    config_kwargs.update(
        dict(
            context_len=0,
            inter_episodes_attention=True,
            banded_attention=False,
            orphan_steps_attention=True,
        )
    )
    config = LlamaRLConfig(**config_kwargs)
    model = LlamaRLModel(config)

    output_mask = model._prepare_decoder_attention_mask(
        masks, data.shape, data, past_key_values_length=0
    )

    output_mask_gt = torch.zeros((masks.shape[0], 1, masks.shape[1], masks.shape[1]))
    output_mask_gt.masked_fill_(
        torch.tensor(
            ~np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype="bool",
            )
        ),
        torch.finfo(masks.dtype).min,
    )

    assert (output_mask == output_mask_gt).all()

    config_kwargs.update(
        dict(
            context_len=2,
            inter_episodes_attention=True,
            banded_attention=True,
            orphan_steps_attention=True,
        )
    )
    config = LlamaRLConfig(**config_kwargs)
    model = LlamaRLModel(config)

    output_mask = model._prepare_decoder_attention_mask(
        masks, data.shape, data, past_key_values_length=0
    )

    output_mask_gt = torch.zeros((masks.shape[0], 1, masks.shape[1], masks.shape[1]))
    output_mask_gt.masked_fill_(
        torch.tensor(
            ~np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype="bool",
            )
        ),
        torch.finfo(masks.dtype).min,
    )
    assert (
        (output_mask <= torch.finfo(masks.dtype).min)
        == (output_mask_gt <= torch.finfo(masks.dtype).min)
    ).all()

    config_kwargs.update(
        dict(
            context_len=0,
            inter_episodes_attention=True,
            banded_attention=False,
            orphan_steps_attention=False,
        )
    )
    config = LlamaRLConfig(**config_kwargs)
    model = LlamaRLModel(config)

    output_mask = model._prepare_decoder_attention_mask(
        masks, data.shape, data, past_key_values_length=0
    )

    output_mask_gt = torch.zeros(
        (masks.shape[0], 1, masks.shape[1], masks.shape[1]), dtype=masks.dtype
    )
    output_mask_gt.masked_fill_(
        torch.tensor(
            ~np.array(
                [
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ],
                ],
                dtype="bool",
            )
        ),
        torch.finfo(masks.dtype).min,
    )

    assert (
        (output_mask <= torch.finfo(masks.dtype).min)
        == (output_mask_gt <= torch.finfo(masks.dtype).min)
    ).all()

    config_kwargs.update(
        dict(
            context_len=0,
            inter_episodes_attention=False,
            banded_attention=False,
            orphan_steps_attention=True,
        )
    )
    config = LlamaRLConfig(**config_kwargs)
    model = LlamaRLModel(config)

    output_mask = model._prepare_decoder_attention_mask(
        masks, data.shape, data, past_key_values_length=0
    )

    output_mask_gt = torch.zeros(
        (masks.shape[0], 1, masks.shape[1], masks.shape[1]), dtype=masks.dtype
    )
    output_mask_gt.masked_fill_(
        torch.tensor(
            ~np.array(
                [
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        ]
                    ],
                ],
                dtype="bool",
            )
        ),
        torch.finfo(masks.dtype).min,
    )

    assert (
        (output_mask <= torch.finfo(masks.dtype).min)
        == (output_mask_gt <= torch.finfo(masks.dtype).min)
    ).all()

    config_kwargs.update(
        dict(
            context_len=2,
            inter_episodes_attention=False,
            banded_attention=True,
            orphan_steps_attention=True,
        )
    )
    config = LlamaRLConfig(**config_kwargs)
    model = LlamaRLModel(config)

    output_mask = model._prepare_decoder_attention_mask(
        masks, data.shape, data, past_key_values_length=0
    )

    output_mask_gt = torch.zeros(
        (masks.shape[0], 1, masks.shape[1], masks.shape[1]), dtype=masks.dtype
    )
    output_mask_gt.masked_fill_(
        torch.tensor(
            ~np.array(
                [
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        ]
                    ],
                    [
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        ]
                    ],
                ],
                dtype="bool",
            )
        ),
        torch.finfo(masks.dtype).min,
    )

    assert (
        (output_mask <= torch.finfo(masks.dtype).min)
        == (output_mask_gt <= torch.finfo(masks.dtype).min)
    ).all()
