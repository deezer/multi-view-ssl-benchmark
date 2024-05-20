import math
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio  # type: ignore
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset

from ssl_framework.models import Backbone

# Constants
DURATION = 4
SAMPLE_RATE = 16000
SEG_NUM_SAMPLES = SAMPLE_RATE * DURATION


class SelfSupervisedFramework:
    def __init__(self, name: str, batch_size: int, weights_path: str = "weights", device: str = "cpu") -> None:
        """
        Args:
            name (str):
                Model name.
                Can be "byol", "clustering", "barlow_twins", "contrastive", and "feature_stats".

            weights_path (str, optional):
                Path to model weights.

            device (str, optional):
                Device on which computations shall be done.
        """
        self.device = device
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.model = self.load_model(name)

    def load_model(self, name: str) -> Backbone:
        """
        Create model and load its weights into Backbone model.
        ---
        Args:
            name (str):
                Model name.

        Returns:
            Backbone:
                Pre-trained self-supervised model :)
        """
        ssl_model = Backbone(mono=True, duration=DURATION, sr=SAMPLE_RATE)
        weights = torch.load(f"{self.weights_path}/{name}.pt", map_location=torch.device("cpu"))

        if name in ["byol", "clustering"]:
            key_replace = "student.module.backbone."
        elif name in ["barlow_twins", "contrastive", "feature_stats"]:
            key_replace = "module.backbone."
        else:
            raise ValueError(f"{name} is not a model implemented in this work!")

        filtered_weights = {k.replace(key_replace, ""): v for k, v in weights["model"].items() if key_replace in k}
        ssl_model.load_state_dict(filtered_weights, strict=True)
        ssl_model.eval()
        ssl_model.to(self.device)

        return ssl_model

    def load_audio(self, path: str) -> Any:
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

    def create_dataset(self, batches: torch.Tensor) -> DataLoader[tuple[torch.Tensor, ...]]:
        """
        Dataloader creation to iter through batches.
        """
        dataset = TensorDataset(batches)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        return loader

    @torch.inference_mode()
    def inference(self, waveform: torch.Tensor) -> torch.Tensor | None:
        output = None
        batches = waveform.squeeze(-1).unsqueeze(0)[
            :, : SEG_NUM_SAMPLES * math.floor(waveform.shape[0] / SEG_NUM_SAMPLES)
        ]
        batches = rearrange(batches, "c (b s) -> b s c", b=waveform.shape[0] // SEG_NUM_SAMPLES, s=SEG_NUM_SAMPLES)

        data_loader = self.create_dataset(batches)

        for [batch] in iter(data_loader):
            batch_out = self.model(batch.to(self.device))
            if output is None:
                output = batch_out
            else:
                output = torch.cat((output, batch_out), dim=0)

        return output

    def __call__(self, path: str, mean_embedding: bool = True) -> torch.Tensor | None:
        waveform = self.load_audio(path)
        embeddings = self.inference(waveform)

        if mean_embedding and embeddings is not None:
            embeddings = embeddings.mean(dim=0)

        return embeddings


if __name__ == "__main__":
    # Sample usage
    sslFramework = SelfSupervisedFramework(name="barlow_twins", batch_size=4)
    file_mean_embeddings = sslFramework("audio_example.mp3", mean_embedding=True)
