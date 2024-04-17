import csv
import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Tuple

import gin  # type: ignore
import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from ssl_framework.downstream_tasks.datasets import DownstreamDataset
from ssl_framework.utils.parallel import set_gpus

TAG_HYPHEN = "---"
CATEGORIES = ["genre", "instrument", "mood/theme"]


def get_parser() -> Any:
    # TODO
    return []


def get_length(values: Any) -> int:
    return len(str(max(values)))


def get_id(value: str) -> int:
    return int(value.split("_")[1])


def read_file(
    tsv_file: str,
) -> Tuple[Dict[int, Dict[str, Any]], DefaultDict[Any, Dict[Any, Any]], Dict[str, int]]:
    tracks: Dict[int, Dict[str, Any]] = {}
    tags: DefaultDict[Any, Dict[Any, Any]] = defaultdict(dict)

    # For statistics
    artist_ids = set()
    albums_ids = set()

    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            track_id = get_id(row[0])
            tracks[track_id] = {
                "artist_id": get_id(row[1]),
                "album_id": get_id(row[2]),
                "path": row[3],
                "duration": float(row[4]),
                "tags": row[5:],  # raw tags, not sure if will be used
            }
            tracks[track_id].update({category: set() for category in CATEGORIES})

            artist_ids.add(get_id(row[1]))
            albums_ids.add(get_id(row[2]))

            for tag_str in row[5:]:
                category, tag = tag_str.split(TAG_HYPHEN)

                if tag not in tags[category]:
                    tags[category][tag] = set()

                tags[category][tag].add(track_id)

                if category not in tracks[track_id]:
                    tracks[track_id][category] = set()

                tracks[track_id][category].update(set(tag.split(",")))

    print("Reading: {} tracks, {} albums, {} artists".format(len(tracks), len(albums_ids), len(artist_ids)))

    extra = {
        "track_id_length": get_length(tracks.keys()),
        "artist_id_length": get_length(artist_ids),
        "album_id_length": get_length(albums_ids),
    }
    return tracks, tags, extra


def get_jamendo_data(
    experiment: str, dataset_path: str, percentage: float = 1.0
) -> Tuple[
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64]],
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64]],
    Tuple[npt.NDArray[Any], npt.NDArray[np.float64]],
]:
    if experiment == "genre":
        train_path = dataset_path + "data/splits/split-0/autotagging_genre-train.tsv"
        train_split, _, _ = read_file(train_path)

        val_path = dataset_path + "data/splits/split-0/autotagging_genre-validation.tsv"
        val_split, _, _ = read_file(val_path)

        test_path = dataset_path + "data/splits/split-0/autotagging_genre-test.tsv"
        test_split, _, _ = read_file(test_path)

        tags = []
        for key, _ in train_split.items():
            for genre in train_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

        for key, _ in val_split.items():
            for genre in val_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

        for key, _ in test_split.items():
            for genre in test_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

    elif experiment == "jam_mood":
        train_path = dataset_path + "data/splits/split-0/autotagging_moodtheme-train.tsv"
        train_split, _, _ = read_file(train_path)

        val_path = dataset_path + "data/splits/split-0/autotagging_moodtheme-validation.tsv"
        val_split, _, _ = read_file(val_path)

        test_path = dataset_path + "data/splits/split-0/autotagging_moodtheme-test.tsv"
        test_split, _, _ = read_file(test_path)

        tags = []
        for key, _ in train_split.items():
            for mood in train_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

        for key, _ in val_split.items():
            for mood in val_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

        for key, _ in test_split.items():
            for mood in test_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

    elif experiment == "jam_instrument":
        train_path = dataset_path + "data/splits/split-0/autotagging_instrument-train.tsv"
        train_split, _, _ = read_file(train_path)

        val_path = dataset_path + "data/splits/split-0/autotagging_instrument-validation.tsv"
        val_split, _, _ = read_file(val_path)

        test_path = dataset_path + "data/splits/split-0/autotagging_instrument-test.tsv"
        test_split, _, _ = read_file(test_path)

        tags = []
        for key, _ in train_split.items():
            for instrument in train_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

        for key, _ in val_split.items():
            for instrument in val_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

        for key, _ in test_split.items():
            for instrument in test_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

    elif experiment == "jam_top50":
        train_path = dataset_path + "data/splits/split-0/autotagging_top50tags-train.tsv"
        train_split, _, _ = read_file(train_path)

        val_path = dataset_path + "data/splits/split-0/autotagging_top50tags-validation.tsv"
        val_split, _, _ = read_file(val_path)

        test_path = dataset_path + "data/splits/split-0/autotagging_top50tags-test.tsv"
        test_split, _, _ = read_file(test_path)

        tags = []
        for key, _ in train_split.items():
            for genre in train_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

            for mood in train_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

            for instrument in train_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

        for key, _ in val_split.items():
            for genre in val_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

            for mood in val_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

            for instrument in val_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

        for key, _ in test_split.items():
            for genre in test_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

            for mood in test_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

            for instrument in test_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

    elif experiment == "jam_overall":
        train_path = dataset_path + "data/splits/split-0/autotagging-train.tsv"
        train_split, _, _ = read_file(train_path)

        val_path = dataset_path + "data/splits/split-0/autotagging-validation.tsv"
        val_split, _, _ = read_file(val_path)

        test_path = dataset_path + "data/splits/split-0/autotagging-test.tsv"
        test_split, _, _ = read_file(test_path)

        tags = []
        for key, _ in train_split.items():
            for genre in train_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

            for mood in train_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

            for instrument in train_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

        for key, _ in val_split.items():
            for genre in val_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

            for mood in val_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

            for instrument in val_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

        for key, _ in test_split.items():
            for genre in test_split[key]["genre"]:
                if genre not in tags:
                    tags.append(genre)

            for mood in test_split[key]["mood/theme"]:
                if mood not in tags:
                    tags.append(mood)

            for instrument in test_split[key]["instrument"]:
                if instrument not in tags:
                    tags.append(instrument)

    # Create tag dictionary
    indices = range(len(tags))
    mapping = dict(zip(tags, indices))

    # Train, validation, and test set IDs
    train_ids = list(train_split.keys())

    if percentage != 1.0:
        random.shuffle(train_ids)
        num_train = int(percentage * len(train_ids))
        train_ids = train_ids[:num_train]

    val_ids = list(val_split.keys())
    test_ids = list(test_split.keys())

    # Create file path and tag arrays
    if experiment == "jam_mood":
        audio_path = dataset_path + "autotagging_moodtheme/audio/"
    else:
        audio_path = dataset_path + "raw_30s/audio/"

    train_paths = []
    val_paths = []
    test_paths = []

    train_tags = np.zeros((len(train_ids), len(tags)))
    val_tags = np.zeros((len(val_ids), len(tags)))
    test_tags = np.zeros((len(test_ids), len(tags)))

    for x, key in enumerate(train_ids):
        for tag in train_split[key]["tags"]:
            _, t = tag.split("---")
            y = mapping[t]
            train_tags[x, y] = 1

        train_paths.append(audio_path + train_split[key]["path"])

    for x, key in enumerate(val_ids):
        for tag in val_split[key]["tags"]:
            _, t = tag.split("---")
            y = mapping[t]
            val_tags[x, y] = 1

        val_paths.append(audio_path + val_split[key]["path"])

    for x, key in enumerate(test_ids):
        for tag in test_split[key]["tags"]:
            _, t = tag.split("---")
            y = mapping[t]
            test_tags[x, y] = 1

        test_paths.append(audio_path + test_split[key]["path"])

    return (
        (np.array(train_paths), train_tags),
        (np.array(val_paths), val_tags),
        (np.array(test_paths), test_tags),
    )


@gin.configurable  # type: ignore
def get_data_sets(
    model_duration_seconds: float,
    sampling_frequency: int,
    mono: bool,
    val_steps: int,
    batch_size: int = 256,
    experiment: str = "jam_top50",
    percentage: float = 1.0,
    run_val: bool = True,
    train_mode_with_val: bool = True,
    train_mode_with_test: bool = False,
    dataset_path: str = "/path/to/mtg-jamendo-dataset/",
) -> Tuple[Tuple[str, int], Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]]:
    """
    Get train, val, and test datasets that can be fed into torch dataloader.
    ---
    Args:
        experiment (str):
            MSD experiment being run.
            Can be 'MSDS', 'MSD50', 'MSD100', and 'MSD500'.
        percentage (float):
            Percentage of train set used for training.
        dataset_path (str):
            Path to Jamendo dataset as described in:
            https://github.com/MTG/mtg-jamendo-dataset

    Returns:
        Tuple[str, int]:
            Experiment and number of classes.
        Tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
            Train, val, and test sets in torch format :)
    """
    if experiment not in [
        "jam_genre",
        "jam_overall",
        "jam_top50",
        "jam_instrument",
        "jam_mood",
    ]:
        raise ValueError(f"Experiment {experiment} is not included in Jamendo settings!")

    # Get appropriate data
    if experiment == "jam_genre":
        (
            (train_ids, train_tags),
            (val_ids, val_tags),
            (test_ids, test_tags),
        ) = get_jamendo_data(experiment, dataset_path, percentage)

        tag_dim = 87

    elif experiment == "jam_overall":
        (
            (train_ids, train_tags),
            (val_ids, val_tags),
            (test_ids, test_tags),
        ) = get_jamendo_data(experiment, dataset_path, percentage)

        tag_dim = 183

    elif experiment == "jam_top50":
        (
            (train_ids, train_tags),
            (val_ids, val_tags),
            (test_ids, test_tags),
        ) = get_jamendo_data(experiment, dataset_path, percentage)

        tag_dim = 50

    elif experiment == "jam_instrument":
        (
            (train_ids, train_tags),
            (val_ids, val_tags),
            (test_ids, test_tags),
        ) = get_jamendo_data(experiment, dataset_path, percentage)

        tag_dim = 40

    elif experiment == "jam_mood":
        (
            (train_ids, train_tags),
            (val_ids, val_tags),
            (test_ids, test_tags),
        ) = get_jamendo_data(experiment, dataset_path, percentage)

        tag_dim = 56

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
