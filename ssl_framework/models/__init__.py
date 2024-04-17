from typing import Any, List

import gin  # type: ignore
import torch
import torchaudio  # type: ignore
from einops import rearrange
from torch import nn

from ssl_framework.models.resnet import ModifiedResNet
from ssl_framework.models.utils import MyWrapperModule


@gin.configurable
class SpecModule(MyWrapperModule):
    def __init__(
        self,
        mono: bool,
        duration: float,
        sr: int,
        name: str = "melspetrogram",
        n_fft: int = 1024,
        f_min: int = 0,
        f_max: int = 11025,
        n_mels: int = 128,
        mel_scale: str = "slaney",
        top_db: int = 100,
    ) -> None:
        """
        Waveform to melspetrogram:
            Input -> waveform of form [batch, n_channels, samples]
            Output -> melspectrogram of form [batch, n_channels, n_mels, time_frames]
        """
        super(SpecModule, self).__init__(name)
        self.mono = mono
        self.duration = duration
        self.sr = sr
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.mel_scale = mel_scale
        self.duration_samples = int(self.duration * self.sr)
        self.n_frames = self.duration_samples // (self.n_fft // 2) + 1
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=True,
            mel_scale=mel_scale,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)

    def summary(self) -> None:
        super(SpecModule, self)._summary((*[self.duration_samples, 1 if self.mono else 2],))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b s c -> b c s")
        x = self.amplitude_to_db(self.spec(x))
        if self.mono and x.shape[1] != 1:
            x = torch.mean(x, dim=1, keepdim=True)
        return x


@gin.configurable
class Head(MyWrapperModule):
    def __init__(self, feature_dim: int, layers_feature_dim: List[int]) -> None:
        super(Head, self).__init__()
        assert len(layers_feature_dim) > 0
        self.feature_dim = feature_dim
        self.ops = nn.ModuleList([])
        self.in_features = feature_dim
        self.output_dim = layers_feature_dim[-1]
        for feature_dim in layers_feature_dim[:-1]:
            out_features = feature_dim
            self.ops.append(nn.Linear(self.in_features, out_features, bias=False))
            self.ops.append(nn.BatchNorm1d(out_features))
            self.ops.append(nn.ReLU())
            self.in_features = out_features
        self.ops.append(nn.Linear(self.in_features, layers_feature_dim[-1], bias=False))

    def summary(self) -> None:
        super(Head, self)._summary((*[16, self.feature_dim],))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2
        for op in self.ops:
            if op.__class__.__name__ in ["BatchNorm1d", "SyncBatchNorm"] and len(x.shape) > 2:
                x = rearrange(x, "b s f -> b f s")
            x = op(x)
            if op.__class__.__name__ in ["BatchNorm1d", "SyncBatchNorm"] and len(x.shape) > 2:
                x = rearrange(x, "b f s -> b s f")
        return x


@gin.configurable
class Backbone(MyWrapperModule):
    def __init__(
        self,
        mono: bool,
        duration: float,
        sr: int,
        feature_dim: int = 1024,
    ) -> None:
        super(Backbone, self).__init__()
        self.feature_dim = feature_dim
        self.spec = SpecModule(mono=mono, duration=duration, sr=sr)
        self.embedding = ModifiedResNet(
            n_frames=self.spec.n_frames,
            n_mels=self.spec.n_mels,
            mono=self.spec.mono,
            feature_dim=self.feature_dim,
        )

    def summary(self) -> None:
        super(Backbone, self)._summary((*[16, self.spec.duration_samples, 1 if self.spec.mono else 2],))

    def get_output_dim(self) -> Any:
        return self.forward(torch.rand([2, self.spec.duration_samples, 1 if self.spec.mono else 2])).shape[1:]

    def forward(self, x: torch.Tensor) -> Any:
        x = self.spec(x)
        return self.embedding(x)
