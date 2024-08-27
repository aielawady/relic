from math import ceil
import torch
import torch.nn as nn
from vc_models.models.vit import model_utils
from torch.nn import functional as F


class Vc1Wrapper(nn.Module):
    """
    Wrapper for the VC1 visual encoder. This will automatically download the model if it's not already.
    """

    def __init__(self, im_obs_space, model_id=None, vc1_config=None):
        super().__init__()
        assert vc1_config is not None, "Make sure you pass vc1_config to Vc1Wrapper."
        self.vc1_config = vc1_config

        if model_id is None:
            model_id = model_utils.VC1_BASE_NAME
        print(f"loading {model_id}.")
        (
            self.net,
            self.embd_size,
            self.model_transforms,
            model_info,
        ) = model_utils.load_model(model_id)
        self._image_obs_keys = [k for k in im_obs_space.spaces.keys() if k != "depth"]

        # Count total # of channels
        self._n_input_channels = sum(
            im_obs_space.spaces[k].shape[2] for k in self._image_obs_keys
        )
        if self.vc1_config.is_2d_output and self.vc1_config.avg_pool_size:
            self.postprocess = nn.AvgPool2d(
                self.vc1_config.avg_pool_size, ceil_mode=True
            )
            self.out_dim = int(ceil(14 / self.vc1_config.avg_pool_size))
        else:
            self.postprocess = nn.Identity()
            self.out_dim = 1

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    @torch.autocast("cuda")
    def forward(self, obs):
        # Extract tensors that are shape [batch_size, img_width, img_height, img_channels]
        feats = []
        imgs = [v for k, v in obs.items() if k in self._image_obs_keys]
        for img in imgs:
            if img.shape[-1] != 3:
                img = torch.concat([img] * 3, dim=-1)
                scale_factor = 1.0
            else:
                scale_factor = 255.0

            img = self.model_transforms(
                img.permute(0, 3, 1, 2).contiguous() / scale_factor
            )

            feats.append(self.net(img))

        if len(feats) == 2:
            # feats = (feats[0] + feats[1])/2
            feats = torch.concat(feats, dim=-1)
        else:
            feats = feats[0]

        return self.postprocess(feats).flatten(1)

    @property
    def output_shape(self):
        return (
            self.out_dim * self.out_dim * self.embd_size * len(self._image_obs_keys),
        )

    @property
    def feats_size(self):
        return self.embd_size * len(self._image_obs_keys)

    def set_grad_checkpointing(self):
        return self.net.set_grad_checkpointing()
