import math
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio  # type: ignore

from ssl_framework.models import Backbone

# Constants
DURATION = 4
SAMPLE_RATE = 16000
WEIGHTS_PATH = "weights"
SEG_NUM_SAMPLES = SAMPLE_RATE * DURATION


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
    ssl_model = Backbone(mono=True, duration=DURATION, sr=SAMPLE_RATE)
    weights = torch.load(f"{WEIGHTS_PATH}/{name}.pt", map_location=torch.device("cpu"))

    if name in ["byol", "clustering"]:
        key_replace = "student.module.backbone."
    elif name in ["barlow_twins", "contrastive", "feature_stats"]:
        key_replace = "module.backbone."
    else:
        raise ValueError(f"{name} is not a model implemented in this work!")

    filtered_weights = {k.replace(key_replace, ""): v for k, v in weights["model"].items() if key_replace in k}

    ssl_model.load_state_dict(filtered_weights, strict=True)

    return ssl_model


def load_audio(path: str) -> Any:
    """
    Args:
        path (str):
            Audio file path.

    Returns:
        torch.Tensor:
            Waveform.
    """
    waveform, sr_in = torchaudio.load(path, channels_first=False)

    if len(waveform) == 0:
        raise ValueError("No audio to process")

    # Mono, resample, and normalize audio
    waveform = torch.mean(waveform, dim=1).unsqueeze(-1)
    waveform = torchaudio.functional.resample(waveform.T, sr_in, SAMPLE_RATE).T
    waveform = waveform / waveform.abs().max()

    # Pad to multiple of SEG_NUM_SAMPLES for batching
    pad_sample = SEG_NUM_SAMPLES * math.ceil(waveform.shape[0] / SEG_NUM_SAMPLES)
    waveform = F.pad(input=waveform, pad=(0, 0, 0, pad_sample - waveform.shape[0]), mode="constant", value=0)

    return waveform


if __name__ == "__main__":
    # Sample code
    model = load_model("barlow_twins")
    waveform = load_audio("audio_example.mp3")
