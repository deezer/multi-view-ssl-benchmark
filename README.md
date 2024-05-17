# An Experimental Comparison Of Multi-view Self-supervised Methods For Music Tagging

by [Gabriel Meseguer-Brocal](https://www.linkedin.com/in/gabriel-meseguer-brocal-1032a42b), [Dorian Desblancs](https://www.linkedin.com/in/dorian-desblancs), and [Romain Hennequin](http://romain-hennequin.fr/En/index.html).

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

Sample code for model weight loading, audio loading, and embedding computation can then be found in `usage/sample.py`! Use the following command to run it:
```bash
poetry run python -m usage.sample
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

Our paper can be found on [arXiv](https://arxiv.org/abs/2404.09177) 🌟 The poster we presented at ICASSP 2024 can be found in this repo 🗄️
