import logging
from typing import Any, Dict

import gin  # type: ignore
import torch
from torchmetrics import AUROC, AveragePrecision, ConfusionMatrix

logging.basicConfig(level=logging.INFO)


class Metrics:
    def __init__(self, num_classes: int, metric_name: Any, device: str) -> None:
        if metric_name == "map_macro":
            self.metric = AveragePrecision(
                task="multilabel",
                num_labels=num_classes,
                average="macro",
            )
        elif metric_name == "roc_macro":
            self.metric = AUROC(  # type: ignore
                task="multilabel",
                num_labels=num_classes,
                average="macro",
            )
        elif metric_name == "map_micro":
            self.metric = AveragePrecision(
                task="multilabel",
                num_labels=num_classes,
                average="micro",  # type: ignore
            )
        elif metric_name == "roc_micro":
            self.metric = AUROC(  # type: ignore
                task="multilabel",
                num_labels=num_classes,
                average="micro",  # type: ignore
            )
        elif metric_name == "confusion_matrix":
            self.metric = ConfusionMatrix(  # type: ignore
                task="multiclass", num_classes=num_classes, normalize="true"
            )
        else:
            raise ValueError("Metric {} has not been implemented yet :(".format(metric_name))
        self.metric_name = metric_name

    def __call__(
        self,
        pred: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Any:
        """Compute metric for given ground truths and predictions."""
        if "map" in self.metric_name or "roc" in self.metric_name:
            metric_value = self.metric(pred, ground_truth.int())  # type: ignore
            self.metric.reset()  # type: ignore
        elif "confusion" in self.metric_name:
            metric_value = self.metric(pred.argmax(-1), ground_truth.argmax(-1))  # type: ignore
            self.metric.reset()  # type: ignore
        return metric_value


@gin.configurable  # type: ignore
def get_metrics(
    downstream_dataset: str,
    n_classes: int,
    device: str,
    add_confusion_matrix: bool = False,
) -> Dict[str, Any]:
    assert downstream_dataset in [
        "jam_genre",
        "jam_overall",
        "jam_top50",
        "jam_instrument",
        "jam_mood",
        "MTAT",
    ]
    val_metrics = ["map_macro", "roc_macro", "map_micro", "roc_micro"]
    if add_confusion_matrix:
        val_metrics.append("confusion_matrix")
    return {name: Metrics(num_classes=n_classes, metric_name=name, device=device) for name in val_metrics}
