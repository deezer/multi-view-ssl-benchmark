from typing import Any, List, Tuple

import gin  # type: ignore
import torch
from torch import Tensor

from ssl_framework.constants import DURATION, MONO, SR
from ssl_framework.models import Backbone, Head
from ssl_framework.models.utils import MyWrapperModule
from ssl_framework.utils.parallel import get_rank, is_main_process, model_to_distributed
from ssl_framework.utils.scheduler import get_network_momentum

log_clap = lambda x: torch.clamp(torch.log(x), min=-100)


@gin.configurable
class SSLModel(MyWrapperModule):
    def __init__(
        self,
        pretext: str,
        head_feature_dim: List[int] = [1024, 1024, 2048],
        units_factor: List[int] = [],
    ) -> None:
        super().__init__()
        self.pretext = pretext
        self.backbone = Backbone(mono=MONO, duration=DURATION, sr=SR)
        self.name = "{}/feature_dim_{}/".format(self.backbone.model_architecture, self.backbone.feature_dim)
        feature_dim = self.backbone.get_output_dim()[-1]
        if units_factor:
            head_feature_dim = [feature_dim * i for i in units_factor]
        self.head_feature_dim = head_feature_dim
        self.head = Head(feature_dim=feature_dim, layers_feature_dim=head_feature_dim)

    def summary(self) -> None:
        print("------ BACKBONE ------")
        self.backbone.summary()
        print("------ HEAD FOR {} TRAINING ------".format(self.pretext))
        self.head.summary()
        return

    def update_norm(self, z: Tensor) -> None:
        if not hasattr(self, "norm_embedding"):
            self.norm_embedding = []
        self.norm_embedding += torch.norm(z, dim=1).detach().to("cpu").tolist()
        return

    def get_norm(self) -> Any:
        value = torch.tensor(self.norm_embedding).mean().item()
        self.norm_embedding = []
        return value

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        yb = []
        yh = []
        for view in x:
            b = self.backbone(view)
            self.update_norm(b)
            h = self.head(b)
            yb.append(b.unsqueeze(0))
            yh.append(h.unsqueeze(0))
        return (torch.cat(yh, 0), torch.cat(yb, 0))


@gin.configurable
class SSLBYOL(MyWrapperModule):
    def __init__(
        self,
        pretext: str,
        n_epochs: int,
        n_steps: int,
        head_feature_dim: List[int] = [1024, 1024, 2048],
        predictor_feature_dim: List[int] = [256, 2048],
    ) -> None:
        super().__init__()
        assert head_feature_dim[-1] == predictor_feature_dim[-1]
        self.pretext = pretext
        self.student = SSLModel(pretext=pretext, head_feature_dim=head_feature_dim)
        self.teacher = SSLModel(pretext=pretext, head_feature_dim=head_feature_dim)
        self.projector = Head(feature_dim=head_feature_dim[-1], layers_feature_dim=predictor_feature_dim)
        self.student.name = "student"
        self.teacher.name = "teacher"
        device = get_rank()

        self.name = "{}/feature_dim_{}/".format(
            self.student.backbone.model_architecture, self.student.backbone.feature_dim
        )

        self.network_momentum = get_network_momentum(n_epochs, n_steps)

        if is_main_process():
            self.student.summary()
            self.projector.summary()
        self.student, self.teacher, self.projector = (
            self.student.cuda(device),
            self.teacher.cuda(device),
            self.projector.cuda(device),
        )
        self.teacher.load_state_dict(self.student.state_dict())
        # to distributed
        self.student = model_to_distributed(self.student, device)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def update_teacher(self, current_global_step: int) -> None:
        m_step = self.network_momentum[current_global_step]  # momentum parameter
        # EMA update for the teacher
        with torch.no_grad():
            student_dict = self.student.state_dict()
            teacher_dict = self.teacher.state_dict()
            for key, param_t in teacher_dict.items():
                param_s = student_dict[key]
                param_t.data.mul_(m_step).add_((1 - m_step) * param_s.detach().data)
        return

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sh, sb = self.student(x)
        # the output of the student is passed into the projector
        sp = torch.cat([self.projector(view).unsqueeze(0) for view in sh], 0)
        with torch.no_grad():
            th, tb = self.teacher(x)
        return sp, sb, th, tb


@gin.configurable
class SSLClustering(MyWrapperModule):
    def __init__(
        self,
        pretext: str,
        n_epochs: int,
        n_steps: int,
        head_feature_dim: List[int] = [1024, 1024, 256, 4096],
        parallel: bool = True,
    ) -> None:
        super().__init__()
        self.pretext = pretext
        self.student = SSLModel(pretext=pretext, head_feature_dim=head_feature_dim)
        self.teacher = SSLModel(pretext=pretext, head_feature_dim=head_feature_dim)
        self.n_classes = head_feature_dim[-1]
        self.student.name = "student"
        self.teacher.name = "teacher"
        device = get_rank()

        self.name = "{}/feature_dim_{}/".format(
            self.student.backbone.model_architecture, self.student.backbone.feature_dim
        )

        self.network_momentum = get_network_momentum(n_epochs, n_steps)

        if is_main_process():
            self.student.summary()
        self.student, self.teacher = (
            self.student.cuda(device),
            self.teacher.cuda(device),
        )
        self.teacher.load_state_dict(self.student.state_dict())
        if parallel:
            # to distributed
            self.student = model_to_distributed(self.student, device)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def update_teacher(self, current_global_step: int) -> None:
        m_step = self.network_momentum[current_global_step]  # momentum parameter
        # EMA update for the teacher
        with torch.no_grad():
            for param_s, param_t in zip(
                self.student.parameters(),
                self.teacher.parameters(),
            ):
                param_t.data.mul_(m_step).add_((1 - m_step) * param_s.detach().data)
        return

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sh, sb = self.student(x)
        with torch.no_grad():
            th, tb = self.teacher(x)
        return sh, sb, th, tb


def selfsupervised_model(self: Any, pretext: str) -> MyWrapperModule:
    model: MyWrapperModule
    if pretext in ["feature_stats", "contrastive", "barlow_twins", "joint", "whole"]:
        model = SSLModel(pretext=pretext)
    elif pretext == "clustering":
        model = SSLClustering(pretext=pretext, n_epochs=self.n_epochs, n_steps=self.n_steps)
    elif pretext == "byol":
        model = SSLBYOL(pretext=pretext, n_epochs=self.n_epochs, n_steps=self.n_steps)
    else:
        raise ValueError(f"Approch {pretext} does not return a self-supervised model!")
    return model


def selfsupervised_one_step(self: Any, batch: Any, pretext: str) -> Any:
    if pretext in ["feature_stats", "contrastive", "barlow_twins"]:
        y, _ = self.model(batch)
        loss, loss_print = self.loss_fn(y)
    elif pretext == "clustering":
        sh, _, th, _ = self.model(batch)
        loss, loss_print = self.loss_fn(sh, th, self.current_epoch)
    elif pretext == "byol":
        sh, _, th, _ = self.model(batch)
        loss, loss_print = self.loss_fn(sh, th)
    else:
        raise ValueError(f"Approch {pretext} does not return a self-supervised model!")
    return loss, loss_print
