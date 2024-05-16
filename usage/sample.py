import torch

from ssl_framework.models import Backbone


def load_model(name: str) -> Backbone:
    """
    Create model and load its weights into Backbone model.
    ---
    Args:
        name (str):
            Model name.
            Can be "byol", "clustering", "barlow_twins", "contrastive", and "feature_stats".

    Returns:
        Backbone:
            Pre-trained self-supervised model :)
    """
    ssl_model = Backbone(mono=True, duration=4, sr=16000)
    weights = torch.load(f"weights/{name}.pt", map_location=torch.device("cpu"))

    if name in ["byol", "clustering"]:
        key_replace = "student.module.backbone."
    elif name in ["barlow_twins", "contrastive", "feature_stats"]:
        key_replace = "module.backbone."
    else:
        raise ValueError(f"{name} is not a model implemented in this work!")

    filtered_weights = {k.replace(key_replace, ""): v for k, v in weights["model"].items() if key_replace in k}

    ssl_model.load_state_dict(filtered_weights, strict=True)

    return ssl_model


if __name__ == "__main__":
    # Sample code
    model = load_model("barlow_twins")
