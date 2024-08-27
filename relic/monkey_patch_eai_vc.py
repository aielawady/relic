import torch
import vc_models.models.vit.model_utils as model_utils
import vc_models.models.vit.vit as vit
from vc_models.models.vit.model_utils import _EAI_VC1_BASE_URL, _download_url
from vc_models.models.vit.vit import resize_pos_embed
import os
import vc_models
import hydra
import omegaconf
import logging

logger = logging.getLogger(__name__)
logger.warn("Monkey patching vc_models package to load model by config path.")


def load_model(model_name):
    """
    Loads a model from the vc_models package.
    Args:
        model_name (str): name of the model to load
    Returns:
        model (torch.nn.Module): the model
        embedding_dim (int): the dimension of the embedding
        transform (torchvision.transforms): the transform to apply to the image
        metadata (dict): the metadata of the model
    """
    if os.path.exists(model_name):
        cfg_path = model_name
    else:
        models_filepath = os.path.dirname(os.path.abspath(vc_models.__file__))
        cfg_path = os.path.join(models_filepath, "conf", "model", f"{model_name}.yaml")

    model_cfg = omegaconf.OmegaConf.load(cfg_path)
    # returns tuple of model, embedding_dim, transform, metadata
    return hydra.utils.call(model_cfg)


def download_model_if_needed(ckpt_file):
    if not os.path.exists(ckpt_file):
        model_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
        )
        ckpt_file = os.path.join(model_base_dir, ckpt_file)

    if not os.path.exists(ckpt_file):
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

        model_name = ckpt_file.split("/")[-1]
        model_url = _EAI_VC1_BASE_URL + model_name
        _download_url(model_url, ckpt_file)


def load_mae_encoder(model, checkpoint_path=None):
    if checkpoint_path is None:
        return model
    else:
        model_utils.download_model_if_needed(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        model_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
        )
        checkpoint_path = os.path.join(model_base_dir, checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(
            state_dict["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

    # filter out keys with name decoder or mask_token
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if "decoder" not in k and "mask_token" not in k
    }

    if model.classifier_feature == "global_pool":
        # remove layer that start with norm
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("norm")}
        # add fc_norm in the state dict from the model
        state_dict["fc_norm.weight"] = model.fc_norm.weight
        state_dict["fc_norm.bias"] = model.fc_norm.bias

    model.load_state_dict(state_dict)
    return model


model_utils.load_model = load_model
model_utils.download_model_if_needed = download_model_if_needed
vit.load_mae_encoder = load_mae_encoder
