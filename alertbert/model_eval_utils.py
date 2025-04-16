import json
from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from alertbert.aitads import (
    AlertSequenceBatchSampler,
    AlertSequenceSampler,
    MultiAlertDataset,
    aitads_train_external_mail_hosts,
)
from alertbert.models import (
    BaselineClusteringModel,
    MaskedLanguageModel,
    TimeDeltaClusteringModel,
    TokenClusteringModel,
)
from alertbert.preprocessing import (
    BaseSequenceCollate,
    MaskedLangModelingSequenceCollate,
    Vocabulary,
    build_feature_vocabs,
    default_collate_fn,
    load_feature_vocabs,
)

"""This module contains functions for evaluating alert grouping models."""


def load_reports(
    model_ids: list[str], path: str
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load reports and param_dicts of saved models."""
    reports = {m: json.load(open(path + f"/{m}/report.json")) for m in model_ids}  # noqa: SIM115
    model_param_dicts = {m: reports[m]["params"] for m in model_ids}
    return reports, model_param_dicts


def load_ground_truth_label_vocabs(
    path: str, configuration: str = None
) -> dict[str, callable]:
    """Load the saved ground truth vocabs for data inspection.
    If configuration is given, the vocabularies for this configuration of AIT-ADS-A are loaded,
    otherwise the AIT-ADS vocabularies are loaded.
    IMPORTANT: These vocabs are not for model training as they do not clean the data!
    """
    if configuration:
        keys = [
            "time_label",
            "event_label",
            "short",
            "host",
            "hierarchical_event_label",
        ]
        path = f"{path}/ground_truth_label_vocabs/aitads_augmented/{configuration}"
    else:
        keys = ["time_label", "event_label", "short", "host"]
        path = f"{path}/ground_truth_label_vocabs/aitads"

    label_vocabs = {}
    for k in keys:
        label_vocabs[k] = Vocabulary()
        try:
            label_vocabs[k].load(f"{path}/{k}.json")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find {path}/{k}.json. Have the vocabularies for configuration {configuration} been built?"
            ) from e

    label_vocabs["time"] = default_collate_fn
    label_vocabs["raw_time"] = default_collate_fn
    return label_vocabs


def build_ground_truth_label_vocabs(data: MultiAlertDataset) -> dict[str, callable]:
    """Build ground truth label vocabs for data inspection from a dataset.
    IMPORTANT: These vocabs are not for model training as they do not clean the data!
    """
    features = ["time_label", "event_label", "short", "host"]
    if "hierarchical_event_label" in data.keys:
        features.append("hierarchical_event_label")

    label_vocabs = build_feature_vocabs(data, features)
    label_vocabs["time"] = default_collate_fn
    label_vocabs["raw_time"] = default_collate_fn

    return label_vocabs


def pprint_label_vocabs(label_vocabs: dict[str, callable]) -> None:
    """Pretty print time and event label vocabs."""
    n = len(label_vocabs["event_label"])
    m = len(label_vocabs["time_label"])
    assert m == 13, f"Expected 13 time labels, got {n}."
    assert n == 13, f"Expected 13 event labels, got {m}."
    print("token | event_label            | time_label")
    print("-----------------------------------------------------")
    for i in range(13):
        print(
            f"   {i:2d} | {label_vocabs['event_label'][i]:22s} | {label_vocabs['time_label'][i]:12s}"
        )


def load_data_tools(
    model_ids: list[str],
    model_param_dicts: dict[str, dict],
    path: str,
    label_vocabs: dict[str, callable],
    include_data_loaders: dict[str, Dataset] = None,
) -> dict[str, dict[str, callable]]:
    """Loads vocabs and collate functions for the models."""
    data_tools = {m: {} for m in model_ids}

    for m, params in model_param_dicts.items():
        collate_fn_map = load_feature_vocabs(
            path + f"/{m}",
            set(params["features"]) | set(params["targets"]),
            params["min_freq"],
        )
        collate_fn_map[params["encoding"]] = default_collate_fn
        if "host" in set(params["features"]) | set(params["targets"]):
            collate_fn_map["host"].remove(aitads_train_external_mail_hosts)

        data_tools[m]["vocabs"] = collate_fn_map
        data_tools[m]["inf_coll_fn"] = BaseSequenceCollate(
            label_vocabs | collate_fn_map
        )  # retains vocabs defined by the model in case of key conflicts!
        if include_data_loaders:
            for split, data in include_data_loaders.items():
                sampler = AlertSequenceBatchSampler(
                    data,
                    context_size=params["context_size"],
                    batch_size=16,
                    drop_last=False,
                    shuffle=False,
                )
                data_tools[m][f"{split}_loader"] = DataLoader(
                    data,
                    batch_sampler=sampler,
                    collate_fn=MaskedLangModelingSequenceCollate(
                        collate_fn_map,
                        params["target_ratio"],
                        params["mask_ratio"],
                        params["perturb_ratio"],
                    ),
                )

    return data_tools


def load_models(
    model_param_dicts: dict[str, dict],
    path: str,
    data_tools: dict[str, dict],
    device: torch.device,
) -> dict[str, MaskedLanguageModel]:
    """Loads models from saved state dicts."""
    models = {}

    for m, params in model_param_dicts.items():
        models[m] = MaskedLanguageModel(params=params, vocabs=data_tools[m]["vocabs"])
        models[m].load_state_dict(
            torch.load(path + f"/{m}/model.pt", weights_only=True, map_location=device),
            # strict=False,  # legacy models may have different keys
        )
        models[m].to(device)
        models[m].eval()

    return models


def get_nice_batch(
    dataset: Dataset,
    sampler: AlertSequenceSampler,
    purity: float,
    label_vocabs: dict[str, Vocabulary],
    features: Sequence[str] = ("event_label", "time_label", "host", "short", "time"),
    idx: tuple[int, slice] = None,
) -> tuple[dict[str, np.ndarray], tuple[int, slice]]:
    """A function to select a suitable batch for evaluation from the dataset and plot its features.

    Args:
        dataset: The dataset to sample from.
        sampler: The sampler to use for sampling.
        purity: The maximal purity of the batch.
        label_vocabs: The label vocabs to use for collation.
        features: The features to plot.
        idx: If None, the function will look for a batch with a certain purity. Otherwise, it will use the given index.

    Returns:
        A tuple containing the batch and its index.
    """
    it = 0
    if idx is None:
        # find a batch with a certain purity
        while True:
            it += 1
            idx = next(sampler)[0]
            sample = dataset[idx]
            s = BaseSequenceCollate(label_vocabs)([sample])
            c = Counter(s["event_label"].squeeze().tolist())
            if c.most_common(1)[0][1] <= c.total() * purity:
                break
    else:
        sample = dataset[idx]
        s = BaseSequenceCollate(label_vocabs)([sample])

    # plot the features of the batch
    fig, ax = plt.subplots(
        len(features), 1, figsize=(10, 0.75 * len(features)), squeeze=True
    )
    for i, f in enumerate(features):
        if f == "time":
            ax[i].imshow(
                s[f] - s[f][0, 0],
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
            )
        else:
            ax[i].imshow(s[f], aspect="auto", cmap="viridis", interpolation="nearest")
        ax[i].axis("off")
        ax[i].set_title(f)
    plt.tight_layout()
    plt.show()

    # print statistics of the batch
    print(f"number of iterations: {it}\n")
    print(f"index: {idx}\n")
    print(f"time span: {sample['time'][-1] - sample['time'][0]}s\n")
    for i in features:
        if i == "time":
            continue
        print(f"{i}\n{s[i]}\n")
    print(f"timestamps\n{s['time'] - s['time'][0, 0]}\n")

    return sample, idx


def context_cluster_plot(
    sample: dict[str, np.ndarray],
    models: dict[str, TokenClusteringModel],
    data_tools: dict[str, dict[str, callable]],
    label_vocabs: dict[str, Vocabulary],
    target: str = "event_label",
) -> None:
    """A function to plot the embeddings and assigned cluster labels of the models for a batch of data."""

    # compute the results of the models
    results = {}
    for key, model in models.items():
        x = data_tools[key]["inf_coll_fn"]([sample])
        results[key] = model(x)

    # plot the results as bar charts
    fig, axs = plt.subplots(
        len(models) + 3, 1, figsize=(10, 0.75 * (len(models) + 3)), squeeze=True
    )

    ax = axs[0]
    x = BaseSequenceCollate(label_vocabs)([sample])
    ax.imshow(x[target], aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(target)
    ax.axis("off")

    ax = axs[1]
    c = BaselineClusteringModel(["time"])(x)["cluster"].cpu().numpy()
    ax.imshow(c, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("timestamp")
    ax.axis("off")

    ax = axs[2]
    c = TimeDeltaClusteringModel()(x)["cluster"].cpu().numpy()
    ax.imshow(c, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("time delta 2")
    ax.axis("off")

    for i, key in enumerate(models.keys()):
        ax = axs[i + 3]
        c = results[key]["cluster"].cpu().numpy()
        ax.imshow(c, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"{key}, clusters: {np.min(c)} to {np.max(c)}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # plot the embeddings in 2d
    n_rows = len(models) // 3 + bool(len(models) % 3)
    fig, axs = plt.subplots(
        n_rows,
        3,
        figsize=(12, 4 * n_rows),  # subplot_kw={"projection": "3d"},
    )
    axs = axs.flatten()

    for i, key in enumerate(models.keys()):
        ax = axs[i]
        x = results[key]["red_dim"].cpu().numpy().squeeze()
        # t = results[id]["raw_time"].cpu().numpy().squeeze()
        c = results[key]["cluster"].cpu().numpy()
        ax.set_title(f"{key}, clusters: {np.min(c)} to {np.max(c)}")
        tf = PCA(n_components=2).fit(x)
        x = tf.transform(x)
        ax.scatter(x[:, 0], x[:, 1], c=c, s=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # the following code is used to construct the ground truth label vocabs
    from alertbert.aitads import AITAlertDataset

    # the following code constructs the ground truth label vocabs for the original AIT-ADS dataset
    if False:
        data = AITAlertDataset(split="all", flavour="original")
        label_vocabs = build_ground_truth_label_vocabs(data)

        for k, v in label_vocabs.items():
            if isinstance(v, Vocabulary):
                v.save(f"saved_models/ground_truth_label_vocabs/aitads/{k}.json")

    # the following code constructs the ground truth label vocabs for the augmented AIT-ADS dataset
    if True:
        import os
        import sys

        configs = sys.argv[1:]
        # configs = ["original", "more-noise-1", "more-noise-2", "more-noise-6", "more-noise-11", "simul-attacks"]
        for c in configs:
            print("Loading data...")
            data = AITAlertDataset(split="all", configuration=c)
            print(f"Building ground truth label vocabs for {c}...")
            label_vocabs = build_ground_truth_label_vocabs(data)

            os.makedirs(
                f"saved_models/ground_truth_label_vocabs/aitads_augmented/{c}", exist_ok=True
            )
            for k, v in label_vocabs.items():
                if isinstance(v, Vocabulary):
                    v.save(
                        f"saved_models/ground_truth_label_vocabs/aitads_augmented/{c}/{k}.json"
                    )
