import logging
from typing import Any, Dict

import gin  # type: ignore
import torch

from ssl_framework.utils.gin import gin_keep_info, update_gin

logging.basicConfig(level=logging.INFO)


def load_checkpoint(ckpt_path: str, gin_file: str, save_dict: Dict[str, Any]) -> Any:
    # load checkpoint
    logging.info("Loading checkpoint from: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    logging.info("Loading checkpoint done!")
    # update the checkpoint gin file with the new info
    logging.info("Updating the checkpoint gin file with the new info ...")
    # filter the original gin to keep only the info related with the backbone
    ckpt["gin_info"] = gin_keep_info(ckpt["gin_info"])
    # parse the gin
    with gin.unlock_config():
        gin.parse_config_files_and_bindings([], ckpt["gin_info"])
    logging.info("Updating the checkpoint gin file done!")
    logging.info("GIN INFO FROM CHECKPOINT: {}".format(gin_file))
    for i in ckpt["gin_info"]:
        logging.info("\t {}".format(i))
    save_dict["gin_info"] = update_gin(ckpt["gin_info"], gin_file)
    return ckpt


def update_backbone(
    model: Any,
    model_ckpt: Dict[str, Any],
    freeze_backbone: bool,
    key_replace: str,
) -> Any:
    """Load a subpart of a trained model.

    Args:
      model: a ResNet Model with the new head.
      ckpt_path: the path to the trained checkpoint
      freeze_backbone: weather to freeze the backbone or not

    Returns:
      the model with the new values.
    """
    model_dict = model.state_dict()
    # 1. filter out the head/classification part
    if any([True if key_replace in i else False for i in list(model_ckpt.keys())]):
        tmp = {}
        for key in model_ckpt.keys():
            tmp[key.replace(key_replace, "")] = model_ckpt[key]
        model_ckpt = tmp.copy()
    # start with for making the different between teacher and student
    ckpt_dict = {k: v for k, v in model_ckpt.items() if k.startswith("backbone")}
    # 2. overwrite entries in the existing state dict
    model_dict.update(ckpt_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)
    for name, p in model.named_parameters():
        if "backbone" in name:
            # checking the parameters backbone parameters
            error = "Model not loaded properly"
            assert torch.allclose(model_dict[name], ckpt_dict[name]), error
            # freezing parameters if needed
            if freeze_backbone:
                p.requires_grad = False
    return model


@gin.configurable  # type: ignore
def get_key_replace(ckpt_path: str, model_type: str = "student") -> str:
    key_replace = "module."
    if "clustering" in ckpt_path:
        key_replace = "student.module." if model_type == "student" else "teacher."
    if "byol" in ckpt_path:
        key_replace = "student.module." if model_type == "student" else "teacher."
    return key_replace


def get_exp_id(ckpt_path: str) -> str:
    paradigm = [
        i
        for i in [
            "feature_stats",
            "contrastive",
            "clustering",
            "barlow_twins",
            "byol",
        ]
        if i in ckpt_path
    ]
    assert len(paradigm) == 1
    return "/".join([paradigm[0], ckpt_path.split("/")[-2]])
