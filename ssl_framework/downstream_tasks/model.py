import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import gin  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from ssl_framework.models import Backbone, Head
from ssl_framework.models.utils import MyWrapperModule
from ssl_framework.utils.downstream import get_exp_id, get_key_replace, update_backbone

logging.basicConfig(level=logging.INFO)


@gin.configurable
class DownstreamLoss(MyWrapperModule):
    def __init__(
        self,
        n_classes: int,
        loss_type: List[str] = ["multilabel"],
        activation: List[str] = ["softmax"],
        name: str = "supervised_loss",
        temp: float = 1.0,
    ):
        super(DownstreamLoss, self).__init__(name=name)
        assert all([i in ["cross", "multilabel"] for i in loss_type])
        assert all([i in ["softmax", "sigmoid"] for i in activation])
        self.n_classes = n_classes
        self.loss_type = loss_type
        self.activation = activation
        self.temp = temp
        self.log_clap: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.clamp(torch.log(x), min=-100)

    def forward(
        self, y_h: torch.Tensor, t_arr: npt.NDArray[np.float32]
    ) -> Tuple[torch.Tensor, Dict[str, Union[int, float]], Dict[str, torch.Tensor]]:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        y = {}
        for activation in self.activation:
            if activation == "softmax":
                y[activation] = F.softmax(torch.div(y_h, self.temp), dim=-1)
            elif activation == "sigmoid":
                y[activation] = torch.sigmoid(torch.clamp(y_h, min=-10))
        loss_print = {}
        i = 0
        t = torch.tensor(t_arr).to(y_h.device)
        for activation, v in y.items():
            for loss_type in self.loss_type:
                # to keep the proper gradient
                if loss_type == "cross":
                    # cross entropy loss
                    loss_tmp = torch.mean(-t * self.log_clap(v), dim=-1).mean()
                    loss_print["{}-{}".format(loss_type, activation)] = loss_tmp.item()
                elif loss_type == "multilabel":
                    # multi-label loss
                    loss_tmp = (
                        -1
                        * torch.mean(
                            ((t * self.log_clap(v)) + (1 - t) * self.log_clap(1 - v)),
                            dim=-1,
                        ).mean()
                    )
                    loss_print["{}-{}".format(loss_type, activation)] = loss_tmp.item()
                if i == 0:
                    loss_back = loss_tmp
                else:
                    loss_back += loss_tmp
                i += 1
        loss_back /= i
        loss_print["total"] = loss_back.item()
        return loss_back, loss_print, y


@gin.configurable
class DownstreamModel(MyWrapperModule):
    def __init__(
        self,
        name: str,
        mono: bool,
        sr: int,
        dur: float,
        task: str,
        n_classes: int,
        extra_name: str = "",
        head_feature_dim: List[int] = [],
    ) -> None:
        super().__init__()
        self.task = task
        self.n_classes = n_classes
        self.sr = sr
        self.dur = dur
        self.backbone = Backbone(mono=mono, duration=dur, sr=sr)
        self.name = "{}/{}_{}_feature_dim_{}_{}/".format(
            task,
            name,
            self.backbone.model_architecture,
            self.backbone.feature_dim,
            extra_name,
        )
        feature_dim = self.backbone.get_output_dim()[-1]
        self.head = Head(feature_dim=feature_dim, layers_feature_dim=head_feature_dim + [n_classes])

    def summary(self) -> None:
        print("------ BACKBONE ------")
        self.backbone.summary()
        print("------ HEAD FOR TASK: {} ------".format(self.task.upper()))
        self.head.summary()

    def forward(self, x: torch.Tensor, mean_embedding: bool = False) -> Any:
        y_b = self.backbone(x)
        if mean_embedding:
            y_b = y_b.mean(dim=0, keepdims=True)
        y = self.head(y_b)
        return y


@gin.configurable  # type: ignore
def create_model_for_downstream_task(
    ckpt: Dict[str, Any],
    ckpt_path: str,
    task: str,
    n_classes: int,
    mono: bool,
    sr: int,
    dur: float,
    extra_name: str,
    freeze_backbone: bool = True,
) -> DownstreamModel:
    # 1- this function update the gin file to be sure backbone is the same
    name = get_exp_id(ckpt_path)

    # # 2- create the model
    model = DownstreamModel(
        name=name,
        mono=mono,
        sr=sr,
        dur=dur,
        task=task,
        n_classes=n_classes,
        extra_name=extra_name,
    )
    # 3- update parameters
    model = update_backbone(model, ckpt["model"], freeze_backbone, get_key_replace(ckpt_path))
    logging.info("--- MODEL DEFINITION ---")
    model.summary()
    return model
