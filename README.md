# An Experimental Comparison Of Multi-view Self-supervised Methods For Music Tagging

by [Gabriel Meseguer-Brocal](https://www.linkedin.com/in/gabriel-meseguer-brocal-1032a42b), [Dorian Desblancs](https://www.linkedin.com/in/dorian-desblancs), and [Romain Hennequin](http://romain-hennequin.fr/En/index.html).

<p align="center">
        <img src="https://github.com/deezer/multi-view-ssl-benchmark/blob/main/poster/poster.pdf" width="300">
</p>

## About

This repository contains the models and losses used to generate our results. All trained, self-supervised model weights can be found in the [Releases](https://github.com/deezer/multi-view-ssl-benchmark/releases) section of this repository.

## Getting Started

In order to explore our repository, one can start with the following:
```bash
# Clone and enter repository
git clone https://github.com/deezer/multi-view-ssl-benchmark
cd multi-view-ssl-benchmark

# Install dependencies
pip install poetry
poetry install

# Download sample audio example
wget https://github.com/deezer/multi-view-ssl-benchmark/releases/download/v0.0.1/weights.zip
unzip weights.zip
```

One can then get started with the following Python code snippet to explore the self-supervised model outputs:

```python
import torch
from ssl_framework.models import Backbone

ssl_model = Backbone(mono=True, duration=4, sr=16000)
weights = torch.load('weights/<model_name>.pt', map_location=torch.device('cpu'))

# Use the following values for the following models:
# barlow_twins, contrastive, and feature_stats: module.backbone.
# byol and clustering: student.module.backbone.
key_replace = <value>
filtered_weights = {k.replace(key_replace, ""): v for k, v in weights["model"].items() if key_replace in k}

# Load weights to model
ssl_model.load_state_dict(filtered_weights, strict=True)
```

## Other

We also include the processing of two downstream tasks in `downstream_tasks/`, the `jamendo` and `mtat`. We unfortunately cannot do the same for the Million Song Dataset since it is mapped to songs in the Deezer catalogue.

## Reference

If you use this repository, please consider citing:

```
@inproceedings{meseguer2024experimental,
  title={An Experimental Comparison of Multi-View Self-Supervised Methods for Music Tagging},
  author={Meseguer-Brocal, Gabriel and Desblancs, Dorian and Hennequin, Romain},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1141--1145},
  year={2024},
  organization={IEEE}
}
```

Our paper can be found on [arXiv](https://arxiv.org/abs/2404.09177) 🌟
