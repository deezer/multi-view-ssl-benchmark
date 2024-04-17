from random import sample
from typing import Any, Tuple

import _pickle as cP  # type: ignore
import gin  # type: ignore
import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from ssl_framework.downstream_tasks.datasets import DownstreamDataset
from ssl_framework.utils.parallel import set_gpus


def get_parser() -> Any:
    # TODO
    return []


def get_mtat_data(
    split_path: str,
    music_path: str,
    percentage: float = 1.0,
) -> Tuple[
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64]],
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64]],
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64]],
]:
    """
    Gather MTAT file paths and corresponding tags for train,
    validation, and test sets.
    """
    # Load appropriate files
    load_train_list = cP.load(open(split_path + "train_list_pub.cP", "rb"))
    load_val_list = cP.load(open(split_path + "valid_list_pub.cP", "rb"))
    load_test_list = cP.load(open(split_path + "test_list_pub.cP", "rb"))

    train_ys = np.load(split_path + "y_train_pub.npy")
    val_tags = np.load(split_path + "y_valid_pub.npy")
    test_tags = np.load(split_path + "y_test_pub.npy")

    train_list = []
    val_list = []
    test_list = []

    if percentage != 1.0:
        num_train = int(percentage * len(load_train_list))
        train_indices = sample(list(range(len(load_train_list))), num_train)
        train_tags = np.zeros((len(train_indices), 50))

        index = 0
        for idx, pth in enumerate(load_train_list):
            if idx in train_indices:
                train_tags[index, :] = train_ys[idx, :]
                index += 1

                real_pth = music_path + pth[:-4] + ".mp3"
                train_list.append(real_pth)

    else:
        train_tags = train_ys
        for pth in load_train_list:
            real_pth = music_path + pth[:-4] + ".mp3"
            train_list.append(real_pth)

    for pth in load_val_list:
        real_pth = music_path + pth[:-4] + ".mp3"
        val_list.append(real_pth)

    for pth in load_test_list:
        real_pth = music_path + pth[:-4] + ".mp3"
        test_list.append(real_pth)

    return (
        (np.array(train_list), train_tags),
        (np.array(val_list), val_tags),
        (np.array(test_list), test_tags),
    )


@gin.configurable  # type: ignore
def get_data_sets(
    model_duration_seconds: float,
    sampling_frequency: int,
    mono: bool,
    val_steps: int,
    batch_size: int = 256,
    experiment: str = "MTAT",
    percentage: float = 1.0,
    run_val: bool = True,
    train_mode_with_val: bool = True,
    train_mode_with_test: bool = False,
    music_path: str = "/path/to/magnatagatune/mp3/",
    split_path: str = "/path/to/music_dataset_split/MTAT_split/",
) -> Tuple[Tuple[str, int], Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]]:
    """
    Get train, val, and test datasets that can be fed into torch dataloader.
    ---
    Args:
        experiment (str):
            MTAT experiment being run.
            Can be 'MTAT'.
        percentage (float):
            Percentage of train set used for training.
        split_path (str):
            MTAT splits used for experiments in:
            https://github.com/jongpillee/music_dataset_split/tree/master/MTAT_split
    Returns:
        Tuple[str, int]:
            Experiment and number of classes.
        Tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
            Train, val, and test sets in torch format :)
    """
    if experiment != "MTAT":
        raise ValueError(f"Experiment {experiment} is not included in MTAT settings!")

    # Get appropriate data
    (
        (train_ids, train_tags),
        (val_ids, val_tags),
        (test_ids, test_tags),
    ) = get_mtat_data(split_path, music_path, percentage)

    tag_dim = 50

    # Create loaders
    train_loader = DataLoader(
        DownstreamDataset(
            train_ids,
            train_tags,
            parser=get_parser(),
            model_duration_seconds=model_duration_seconds,
            sampling_frequency=sampling_frequency,
            mono=mono,
            train=True,
        ),
        batch_size=batch_size,
    )
    val_dataset = DownstreamDataset(
        val_ids,
        val_tags,
        parser=get_parser(),
        model_duration_seconds=model_duration_seconds,
        sampling_frequency=sampling_frequency,
        mono=mono,
        train=train_mode_with_val,
    )
    if train_mode_with_val:
        val_dataset.tf_dataloader = (
            val_dataset.tf_dataloader.take(val_steps * batch_size).cache("/tmp/validation_set_cache").repeat()
        )
        if run_val:
            print("--- Runing the ds_test to cache in memory before training ---")
            _ = [None for _ in val_dataset.tf_dataloader.take((val_steps * batch_size * 2) + 1)]
    val_loader = DataLoader(val_dataset, batch_size=batch_size if train_mode_with_val else 1)
    test_loader = DataLoader(
        DownstreamDataset(
            test_ids,
            test_tags,
            parser=get_parser(),
            sampling_frequency=sampling_frequency,
            mono=mono,
            train=train_mode_with_test,
        ),
        batch_size=batch_size if train_mode_with_test else 1,
    )

    return (experiment, tag_dim), (train_loader, val_loader, test_loader)


# FOR DEBUGGING
if __name__ == "__main__":
    _ = set_gpus()

    (_, _), (train_loader, val_loader, test_loader) = get_data_sets()

    # Time train, val, and test epochs with tqdm
    for audios, tags in tqdm(train_loader):
        size_audios = audios.shape
        size_tags = tags.shape
        break

    print(f"Training batch audio shape is {size_audios} and tag shape is {size_tags}")

    for audios, tags in tqdm(val_loader):
        size_audios = audios.shape
        size_tags = tags.shape
        break

    print(f"Validation batch sample audio shape is {size_audios} and tag shape is {size_tags}")

    for audios, tags in tqdm(test_loader):
        size_audios = audios.shape
        size_tags = tags.shape
        break

    print(f"Test batch sample audio shape is {size_audios} and tag shape is {size_tags}")
