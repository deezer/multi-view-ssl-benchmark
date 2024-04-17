from typing import Any, Callable, Dict, Tuple, Union

import gin  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ssl_framework.models.utils import MyWrapperModule
from ssl_framework.utils.parallel import FullGatherLayer, get_world_size
from ssl_framework.utils.scheduler import (
    get_temp_student_schedule,
    get_temp_teacher_schedule,
)

log_clap: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.clamp(torch.log(x), min=-100)


def ema(x: torch.Tensor, momentum: float, x_batch: torch.Tensor) -> float:
    return x * momentum + x_batch * (1 - momentum)


def update_static_parameter(module: Any, key: str, value: float) -> Any:
    state_dict = module.state_dict()
    state_dict[key] = value
    module.load_state_dict(state_dict)
    return module


@gin.configurable
class BYOLLoss(nn.Module):
    def __init__(self) -> None:
        super(BYOLLoss, self).__init__()

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = []
        for i, t in enumerate(teacher_out):
            for j, s in enumerate(student_out):
                # we skip cases where student and teacher operate on the same view
                if i != j:
                    t = F.normalize(t, dim=-1, p=2)
                    s = F.normalize(s, dim=-1, p=2)
                    losses.append(2 - 2 * (t * s).sum(dim=-0))
        loss = torch.cat(losses, 0)
        if get_world_size() > 1:
            loss = torch.cat(FullGatherLayer.apply(loss), dim=0)  # type: ignore
        loss = loss.mean()
        return loss, {"byol": loss.item(), "total": loss.item()}


@gin.configurable
class ClusteringLoss(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_epochs: int,
        center_momentum: float = 0.9,  # 0.996,
    ):
        """
        Adapted code from https://github.com/facebookresearch/dino
        More info at M. Caron, H. Touvron et al.
        https://arxiv.org/abs/2104.14294

        - teacher_temp: Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.
        """
        super(ClusteringLoss, self).__init__()
        self.n_classes = n_classes
        self.center_momentum = center_momentum
        self.center = nn.Parameter(torch.zeros(1, n_classes), requires_grad=False)
        self.teacher_temp_schedule = get_temp_teacher_schedule(n_epochs)
        self.student_temp_schedule = get_temp_student_schedule(n_epochs)

    def _prepare_data(
        self, student: torch.Tensor, teacher: torch.Tensor, epoch: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # student sharpening
        student_out = F.softmax(student / self.student_temp_schedule[epoch], dim=-1)
        # teacher centering and sharpening
        teacher_out = F.softmax((teacher - self.center) / self.teacher_temp_schedule[epoch], dim=-1)
        return student_out, teacher_out

    @torch.no_grad()
    def update_center(self, output_teacher: torch.Tensor) -> None:
        """
        Update center used for teacher output.
        """
        if get_world_size() > 1:
            output_teacher = torch.cat(FullGatherLayer.apply(output_teacher), dim=1)  # type: ignore
        output_teacher = rearrange(output_teacher, "v b c -> (v b) c")
        # ema update for the center
        new_center = ema(
            self.center,
            self.center_momentum,
            torch.mean(output_teacher, dim=0, keepdim=True),
        )
        self = update_static_parameter(self, "center", new_center)
        return

    def forward(
        self, output_student: torch.Tensor, output_teacher: torch.Tensor, epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, Union[int, float]]]:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out, teacher_out = self._prepare_data(output_student, output_teacher, epoch)
        # compute loss
        list_loss = []
        for i, t in enumerate(teacher_out):
            if get_world_size() > 1:
                t = torch.cat(FullGatherLayer.apply(t), dim=0)  # type: ignore
            for j, s in enumerate(student_out):
                # we skip cases where student and teacher operate on the same view
                if get_world_size() > 1:
                    s = torch.cat(FullGatherLayer.apply(s), dim=0)  # type: ignore
                if i != j:
                    list_loss.append(torch.sum(-t * log_clap(s), dim=-1))

        loss = torch.cat(list_loss, 0)
        if output_student.requires_grad:
            # update the center with the values before softmax during training
            self.update_center(output_teacher)
        loss = loss.mean()
        return loss, {"clustering": loss.item(), "total": loss.item()}


@gin.configurable
class NT_Xent(nn.Module):
    def __init__(self, temp: float = 0.1, similarity: str = "cosine"):
        super(NT_Xent, self).__init__()
        assert similarity in ["cosine", "dot"]
        self.similarity_fn = self.get_similarity_fn(similarity)
        self.temp = temp

    def get_similarity_fn(self, similarity: str) -> Callable[[torch.Tensor], Any]:
        if similarity == "cosine":
            return lambda x: nn.CosineSimilarity(dim=2)(x.unsqueeze(0), x.unsqueeze(1))
        elif similarity == "dot":
            return lambda x: (x.type(torch.float32) @ x.T.type(torch.float32)) / (x.shape[1] - 1)
        else:
            raise ValueError(f"Similarity function {similarity} does not exist!")

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Union[int, float]]]:
        # y -> "(views, batch, features)"
        device = y.device
        if get_world_size() > 1:
            y = torch.cat(FullGatherLayer.apply(y), dim=1)  # type: ignore
        n_views, batch_size, _ = y.shape
        y = rearrange(y, "v b f -> (v b) f")
        mask = torch.eye(n_views * batch_size, device=device).bool()
        y = self.similarity_fn(y) / self.temp
        # similarity bewteen views coming from the same audio
        pos_pairs = torch.cat(
            [torch.diag(y, batch_size * i) for i in range(1, n_views)]
            + [torch.diag(y, -batch_size * i) for i in range(1, n_views)]
        ).reshape(batch_size * n_views, -1)
        neg_pairs = y[~mask].reshape(batch_size * n_views, -1)
        max_val = torch.max(neg_pairs, dim=1, keepdim=True)[0].detach()
        loss = -log_clap(
            torch.exp(pos_pairs - max_val) / torch.sum(torch.exp(neg_pairs - max_val), dim=1).unsqueeze(-1)
        )
        loss = loss.mean()
        return loss, {"nt_xent": loss.item(), "total": loss.item()}


@gin.configurable
class DiagonalLoss(nn.Module):
    def __init__(self, n_features: int, off_coef: float = 1.0):
        """
        Adapted code from https://github.com/facebookresearch/barlowtwins
        More info at Zbontar, Jure, et al.
        https://arxiv.org/abs/2103.03230
        """
        super(DiagonalLoss, self).__init__()
        self.off_coef = off_coef
        self.bn = nn.BatchNorm1d(n_features, affine=False)

    def compute_cross_corelation_matrix(self, t: torch.Tensor, s: torch.Tensor) -> Any:
        batch_size = t.shape[0]
        return (self.bn(t).T @ self.bn(s)) / (batch_size - 1)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Union[float, int]]]:
        list_on_loss = []
        list_off_loss = []
        mask = torch.eye(y[0].shape[-1], device=y.device).bool()
        for i, a in enumerate(y):
            if get_world_size() > 1:
                a = torch.cat(FullGatherLayer.apply(a), dim=0)  # type: ignore
            # we skip cases where student and teacher operate on the same view
            for b in y[i + 1 :]:
                if get_world_size() > 1:
                    b = torch.cat(FullGatherLayer.apply(b), dim=0)  # type: ignore
                # empirical cross-correlation matrix between the centroids of each class
                cc = self.compute_cross_corelation_matrix(a, b)
                list_on_loss.append((1 - cc[mask]).pow(2).unsqueeze(0))
                list_off_loss.append(cc[~mask].pow(2).unsqueeze(0))
        on_loss = torch.cat(list_on_loss, 0).mean()
        off_loss = torch.cat(list_off_loss, 0).mean()
        loss = on_loss + self.off_coef * off_loss
        loss_print = {
            "on_diag": on_loss.item(),
            "off_diag": self.off_coef * off_loss.item(),
            "total": loss.item(),
        }
        return loss.mean(), loss_print


@gin.configurable
class FeatureStatsLoss(MyWrapperModule):
    def __init__(
        self,
        n_features: int,
        inv_coeff: float = 1.0,  # 50.0,  # 25.0
        var_coeff: float = 1.0,  # 25.0
        cov_coeff: float = 1.0,  # 25.0,  # 1.0,
        gamma: float = 1.0,
        name: str = "feature_stats_loss",
    ):
        """
        Adapted code from https://github.com/facebookresearch/vicreg
        More info at Adrien Bardes, Jean Ponce and Yann LeCun
        https://arxiv.org/abs/2105.04906
        https://generallyintelligent.com/open-source/2022-04-21-vicreg/

        The default parameter try weight each loss to have a similar contribution
        """
        super(FeatureStatsLoss, self).__init__(name=name)
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.bn = nn.BatchNorm1d(n_features, affine=False)

    def stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        var_x = torch.sqrt(torch.var(x, dim=0) + torch.tensor(1e-5))
        cov_x = (self.bn(x).T @ self.bn(x)) / (batch_size - 1)
        return var_x, cov_x

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Union[float, int]]]:
        loss_print = {}
        mask = torch.eye(y[0].shape[-1], device=y.device).bool()
        list_inv_loss = []
        list_var_loss = []
        list_cov_loss = []
        for i, a in enumerate(y):
            if get_world_size() > 1:
                a = torch.cat(FullGatherLayer.apply(a), dim=0)  # type: ignore
            # we skip cases where student and teacher operate on the same view
            for b in y[i + 1 :]:
                if get_world_size() > 1:
                    b = torch.cat(FullGatherLayer.apply(b), dim=0)  # type: ignore

                var_a, cov_a = self.stats(a)
                var_b, cov_b = self.stats(b)

                # Invariance term: makes distance between views the small
                inv = torch.square(a - b)
                list_inv_loss.append(inv)

                # Variance term: makes features values different across the view in the batch.
                list_var_loss.append(F.relu(self.gamma - var_a).unsqueeze(0))
                list_var_loss.append(F.relu(self.gamma - var_b).unsqueeze(0))

                # Covariance term: make each feature decorrelated from the rest
                list_cov_loss.append(cov_a[~mask].pow_(2).unsqueeze(0))
                list_cov_loss.append(cov_b[~mask].pow_(2).unsqueeze(0))

        inv_loss = torch.cat(list_inv_loss, 0).mean() * self.inv_coeff
        var_loss = torch.cat(list_var_loss, 0).mean() * self.var_coeff
        cov_loss = torch.cat(list_cov_loss, 0).mean() * self.cov_coeff
        loss_back = inv_loss + var_loss + cov_loss
        loss_print["total"] = loss_back.item()
        loss_print["inv"] = inv_loss.item()
        loss_print["var"] = var_loss.item()
        loss_print["cov"] = cov_loss.item()
        return loss_back, loss_print


def selfsupervised_loss(self: Any, pretext: str) -> Any:
    loss_fn: Any
    if pretext == "feature_stats":
        loss_fn = FeatureStatsLoss(n_features=self.model.module.head_feature_dim[-1]).cuda(self.device)
    if pretext == "contrastive":
        loss_fn = NT_Xent().cuda(self.device)
    if pretext == "clustering":
        loss_fn = ClusteringLoss(n_epochs=self.n_epochs, n_classes=self.model.n_classes).cuda(self.device)
    if pretext == "barlow_twins":
        loss_fn = DiagonalLoss(n_features=self.model.module.head_feature_dim[-1]).cuda(self.device)
    if pretext == "byol":
        loss_fn = BYOLLoss().cuda(self.device)
    return loss_fn
