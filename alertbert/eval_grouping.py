import pickle
from collections import Counter
from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from tensordict import TensorDict
from torch.utils.data import DataLoader

from alertbert.models import (
    AbstractClusteringModel,
    MaskedLangModelInferenceWrapper,
    TimeDeltaClusteringModel,
    TokenClusteringModel,
)
from alertbert.preprocessing import BaseSequenceCollate, Vocabulary

"""This module contains functions for evaluating alert grouping models.
If executed as a script, it will load a trained model and evaluate it on the training and validation sets of the AIT Alert dataset.
"""

np.seterr(all="raise")
eval_seed = 5287602698342552483


# utility functions


def entropy(counts: np.ndarray[int | float]) -> float:
    """Computes the entropy of the histogram counts."""
    counts = counts[np.nonzero(counts)]
    p = counts / np.sum(counts)
    return np.sum(-p * (np.log2(p)))


def contingency_matrix(
    true: np.ndarray[int], pred: np.ndarray[int], class_range: tuple[int, int]
) -> np.ndarray[int]:
    """This function is an adaption of sklearn.metrics.cluster.contingency_matrix which computes the contingency matrix
    for all true class labels and not just those which appear in the true labels of the current batch.

    Parameters:
    - true (np.ndarray[int]): The true class labels.
    - pred (np.ndarray[int]): The predicted cluster labels.
    - class_range (tuple[int, int]): The range of class labels.

    Returns:
    - np.ndarray[int]: The contingency matrix of shape (class_range[1] - class_range[0] + 1, n_clusters) where the value at (i, j)
        is the number of samples that have true label i and predicted cluster j.
    """
    clusters, cluster_idx = np.unique(pred, return_inverse=True)
    n_clusters = clusters.shape[0]

    contingency = coo_matrix(
        (np.ones(true.shape[0]), (true - class_range[0], cluster_idx)),
        shape=(class_range[1] - class_range[0] + 1, n_clusters),
        dtype=np.int64,
    )
    return contingency.toarray()


def get_str_labels(target_vocab: Vocabulary) -> list[str]:
    """Returns a list of string labels for the target vocabulary."""
    return target_vocab[target_vocab.offset + 1 :]


def get_labels_int(str_labels: list[str], target_vocab: Vocabulary) -> np.ndarray[int]:
    """Returns a list of integer labels for the target vocabulary."""
    return target_vocab([str_labels]).numpy().squeeze()


def get_low_level_labels(
    high_level_labels: Vocabulary, level: int, excluded_macro_label: str = "-"
) -> list[str]:
    """Returns a list of low level labels for the target vocabulary."""
    c = Counter()
    for label, count in high_level_labels.counter.items():
        if label != excluded_macro_label:
            c[".".join(label.split(".")[:level])] += count

    return [label for label, count in c.most_common() if label != excluded_macro_label]


# result computation functions


metrics = [
    "count",
    "tp",
    "fp",
    "tn",
    "fn",
    "accuracy",
    "precision",
    "recall",  # recall = tpr = 1 - fnr
    "tnr",  # tnr = 1 - fpr
    "f1",
    "mcc",
]


batch_stats = [
    "context_entropy",
    "context_purity",
    "num_labels",
    "num_clusters",
    "context_size",
    "context_time",
]


def eval_alert_grouping(
    model: AbstractClusteringModel,
    target: str = "hierarchical_event_label",
    loader: DataLoader = None,
    target_vocab: Vocabulary = None,
    epochs: int = 5,
    n_jobs: int = -1,
    excluded_macro_label: str = "-",
) -> dict[str, dict | np.ndarray]:
    """This function computes multiple evaluation metrics for a clustering model on the given dataset.
    For every batch the metrics are computed for every label in the dataset and the macro metrics are
    computed over all labels except the excluded label (which is supposed to be the false positive label).
    Further, also other batch statistics are computed and stored in the results dict.

    Args:
        model (AbstractClusteringModel): The clustering model to be evaluated.
        target (str, optional): The target label in the dataset. Defaults to "hierarchical_event_label".
        loader (DataLoader): The data loader providing the dataset. Defaults to None.
        target_vocab (Vocabulary): The vocabulary containing the target labels. Defaults to None.
        epochs (int, optional): The number of epochs to run the evaluation. Defaults to 5.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1 (use all available processors).
        excluded_macro_label (str, optional): The label to exclude from macro calculations. Defaults to "-".
    Returns:
        dict[str, dict | np.ndarray]: A dictionary containing evaluation metrics for each label,
            macro metrics and batch statistics.
    """

    # check device
    if isinstance(model, TokenClusteringModel):
        assert str(model.model.device) == "cpu", (
            "TokenClusteringModel must be on CPU, joblib does not support GPU devices."
        )

    assert target.startswith("hierarchical"), (
        "Non-hierarchical labels have been deprecated in this function."
    )

    # check n_jobs
    if n_jobs is None:
        n_jobs = 1
    assert n_jobs == -1 or n_jobs > 0, "n_jobs must be -1 or a positive integer."
    if n_jobs != 1:
        model.block_parallelization()

    # set up labels
    all_labels_str = get_str_labels(target_vocab)  # level 3 labels
    all_labels_int = get_labels_int(all_labels_str, target_vocab)
    label_range = (all_labels_int[0], all_labels_int[-1])

    lvl_2_labels = get_low_level_labels(target_vocab, 2, excluded_macro_label)
    lvl_1_labels = get_low_level_labels(target_vocab, 1, excluded_macro_label)

    # initialize results dict
    results = {
        # these will store the (aggregated) metrics for every batch
        "lvl3": {label: {metric: [] for metric in metrics} for label in all_labels_str},
        "lvl2": {label: {metric: None for metric in metrics} for label in lvl_2_labels},
        "lvl1": {label: {metric: None for metric in metrics} for label in lvl_1_labels},
        "macro": {metric: None for metric in metrics[1:]},  # aka level 0
        # meta data
        "batch_stats": {stat: [] for stat in batch_stats},
        "model_params": None,
        # these will store the (aggregated) metrics averaged over all batches
        "summary": {
            "lvl3": {
                label: {metric: (None, None) for metric in metrics}
                for label in all_labels_str
            },
            "lvl2": {
                label: {metric: (None, None) for metric in metrics}
                for label in lvl_2_labels
            },
            "lvl1": {
                label: {metric: (None, None) for metric in metrics}
                for label in lvl_1_labels
            },
            "macro": {"macro": {metric: (None, None) for metric in metrics[1:]}},
        },
    }

    # define encapsulation of model call and cluster count computation for parallelization
    def model_call(batch: TensorDict) -> tuple[np.ndarray[int], np.ndarray[int]]:
        batch = model(batch)
        true = batch[target].cpu().numpy().squeeze()
        pred = batch["cluster"].cpu().numpy().squeeze()
        counts = contingency_matrix(true, pred, label_range)
        return counts, len(true), batch["raw_time"][0, -1] - batch["raw_time"][0, 0]

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        for payload in Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(model_call)(batch) for batch in loader
        ):
            counts, context_size, context_time = payload
            cluster_sizes = counts.sum(axis=0)
            true_label_counts = counts.sum(axis=1)
            total = true_label_counts.sum()
            results["batch_stats"]["context_entropy"].append(entropy(true_label_counts))
            results["batch_stats"]["context_purity"].append(np.max(true_label_counts) / total)
            results["batch_stats"]["num_labels"].append(np.nonzero(true_label_counts)[0].shape[0])
            results["batch_stats"]["num_clusters"].append(cluster_sizes.shape[0])
            results["batch_stats"]["context_size"].append(context_size)
            results["batch_stats"]["context_time"].append(context_time)

            for label, int_label in zip(all_labels_str, all_labels_int):
                idx = int_label - label_range[0]

                # continue if label does not appear in batch
                if true_label_counts[idx] == 0:
                    for k, v in results["lvl3"][label].items():
                        if k == "count":
                            v.append(0)
                        else:
                            v.append(np.nan)
                    continue

                # compute batch results
                tp = np.sum(counts[idx] * counts[idx])
                fp = np.sum(counts[idx] * (cluster_sizes - counts[idx]))
                fn = np.sum(counts[idx] * (true_label_counts[idx] - counts[idx]))
                tn = np.sum(counts[idx] * (total - cluster_sizes - true_label_counts[idx] + counts[idx]))
                acc = (tp + tn) / (true_label_counts[idx] * total)
                prec = tp / (tp + fp)  # tp > 0 bc each token with itself is always a tp pair
                rec = tp / (tp + fn)
                tnr = tn / (fp + tn) if fp + tn > 0 else np.nan
                f1 = 2 * prec * rec / (prec + rec)
                mcc = (
                    (float(tp) * float(tn) - float(fp) * float(fn))
                    / np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
                    if float(tn + fp) * float(tn + fn) > 0
                    else np.nan
                )

                # add results to results dict
                results["lvl3"][label]["count"].append(true_label_counts[idx])
                results["lvl3"][label]["tp"].append(tp)
                results["lvl3"][label]["fp"].append(fp)
                results["lvl3"][label]["fn"].append(fn)
                results["lvl3"][label]["tn"].append(tn)
                results["lvl3"][label]["accuracy"].append(acc)
                results["lvl3"][label]["precision"].append(prec)
                results["lvl3"][label]["recall"].append(rec)
                results["lvl3"][label]["tnr"].append(tnr)
                results["lvl3"][label]["f1"].append(f1)
                results["lvl3"][label]["mcc"].append(mcc)

    # cast results to numpy arrays and compute macro results
    for metric in metrics:
        # level 3
        for label in all_labels_str:
            results["lvl3"][label][metric] = np.array(results["lvl3"][label][metric])
            results["summary"]["lvl3"][label][metric] = (
                np.nanmean(results["lvl3"][label][metric]),
                np.nanstd(results["lvl3"][label][metric]),
            )

        # level 2
        for label in lvl_2_labels:
            results["lvl2"][label][metric] = np.nanmean(
                np.stack(
                    [
                        results["lvl3"][lvl3_label][metric]
                        for lvl3_label in all_labels_str
                        if lvl3_label.startswith(label)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            results["summary"]["lvl2"][label][metric] = (
                np.nanmean(
                    [
                        results["summary"]["lvl3"][lvl3_label][metric][0]
                        for lvl3_label in all_labels_str
                        if lvl3_label.startswith(label)
                    ]
                ),
                np.nanstd(
                    [
                        results["summary"]["lvl3"][lvl3_label][metric][0]
                        for lvl3_label in all_labels_str
                        if lvl3_label.startswith(label)
                    ]
                ),
            )

        # level 1
        for label in lvl_1_labels:
            results["lvl1"][label][metric] = np.nanmean(
                np.stack(
                    [
                        results["lvl2"][lvl2_label][metric]
                        for lvl2_label in lvl_2_labels
                        if lvl2_label.startswith(label)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            results["summary"]["lvl1"][label][metric] = (
                np.nanmean(
                    [
                        results["summary"]["lvl2"][lvl2_label][metric][0]
                        for lvl2_label in lvl_2_labels
                        if lvl2_label.startswith(label)
                    ]
                ),
                np.nanstd(
                    [
                        results["summary"]["lvl2"][lvl2_label][metric][0]
                        for lvl2_label in lvl_2_labels
                        if lvl2_label.startswith(label)
                    ]
                ),
            )

        # level 0
        results["macro"][metric] = np.nanmean(
            np.stack(
                [results["lvl1"][label][metric] for label in lvl_1_labels],
                axis=0,
            ),
            axis=0,
        )
        results["summary"]["macro"]["macro"][metric] = (
            np.nanmean(
                [
                    results["summary"]["lvl1"][lvl_1_label][metric][0]
                    for lvl_1_label in lvl_1_labels
                ]
            ),
            np.nanstd(
                [
                    results["summary"]["lvl1"][lvl_1_label][metric][0]
                    for lvl_1_label in lvl_1_labels
                ]
            ),
        )

    for stat in batch_stats:
        results["batch_stats"][stat] = np.array(results["batch_stats"][stat])

    return results


# result saving and loading functions


def save_results(
    results: dict[str, dict | np.ndarray],
    path: str,
    name: str,
) -> None:
    """Saves evaluation results to a pickle file.

    Args:
        results (dict[str, dict | np.ndarray]): The results dict to be saved.
        path (str): The path to the directory where the respective model is located.
        name (str): The name of the results file.
    """
    model_id = results["model_params"]["model"]["id"]
    split = results["model_params"]["data_split"]
    path = f"{path}/{model_id}/{name}_{split}_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_results(
    path: str, model_id: str, name: str, split: Literal["train", "val"]
) -> dict[str, dict | np.ndarray]:
    """Loads evaluation results from a pickle file.

    Args:
        path (str): The path to the directory where the respective model is located.
        model_id (str): The id of the model.
        name (str): The name of the results file.
        split (Literal["train", "val"]): The data split of the results file.
    """
    path = f"{path}/{model_id}/{name}_{split}_results.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    return results


# result plotting functions


plot_cols = [("train", "label"), ("train", "macro"), ("val", "label"), ("val", "macro")]


def get_metrics(exclude_raw_metrics: bool = True) -> list[str]:
    """Returns the list of metrics to be plotted."""
    if exclude_raw_metrics:
        return metrics[5:]
    return metrics[1:]


def get_scatter_plot_figure(
    used_metrics: list[str], x_label: str = None, y_label: str = None
) -> tuple[plt.Figure, plt.Axes]:
    """This function returns a figure and axes for a scatter plot of the evaluation results."""
    fig, all_axs = plt.subplots(
        len(used_metrics),
        4,
        figsize=(4 * len(plot_cols) , 4.5 * len(used_metrics)),
        sharey="row",
        sharex="row",
    )
    all_axs = all_axs.T

    for j, col in enumerate(plot_cols):
        for i, m in enumerate(used_metrics):
            ax = all_axs[j][i]
            ax.set_axisbelow(True)
            ax.set_box_aspect(1)
            ax.grid()
            ax.set_title(f"{col[0]} - {col[1]} - {m}")
            ax.set_xlabel(x_label if x_label else m)
            ax.set_ylabel(y_label if y_label else m)
            if m in get_metrics(True):
                ax.set_ylim((-0.05 if m != "mcc" else -1.05), 1.05)
                if x_label not in batch_stats:
                    ax.set_xlim((-0.05 if m != "mcc" else -1.05), 1.05)

    return fig, all_axs


def grouping_results_v_discriminator_plot(
    train_results: dict[str, dict | np.ndarray],
    val_results: dict[str, dict | np.ndarray],
    target_vocab: Vocabulary,
    disc: str = "context_entropy",
    macro_colour: str = "context_entropy",
    exclude_raw_metrics: bool = True,
    excluded_label: str = "-",
) -> None:
    """This function plots the results of a clustering model against a discriminator variable.
    For every metric and data split the results for each label and the macro results are plotted against the discriminator variable.

    Args:
        train_results (dict[str, dict | np.ndarray]): The training results dict.
        val_results (dict[str, dict | np.ndarray]): The validation results dict.
        target_vocab (Vocabulary): The vocabulary containing the target labels.
        disc (str, optional): The discriminator variable. Defaults to "context_entropy".
        macro_colour (str, optional): The feature to use for the colouring in the macro plots. Defaults to "context_entropy".
        exclude_raw_metrics (bool, optional): Whether to exclude the raw metrics (tp, fp, tn, fn) from the plots. Defaults to True.
        excluded_label (str, optional): The label to be excluded from plotting. This is supposed to be the false positive label. Defaults to "-".
    """

    all_labels_str = get_low_level_labels(target_vocab, 1, excluded_label)
    used_metrics = get_metrics(exclude_raw_metrics)
    fig, all_axs = get_scatter_plot_figure(used_metrics, disc)

    for j, col in enumerate(plot_cols):
        axs = all_axs[j]

        results = train_results if col[0] == "train" else val_results

        d = results["batch_stats"][disc]

        for i, m in enumerate(used_metrics):
            ax = axs[i]
            if col[1] == "label":
                x_vals = []
                y_vals = []
                c_vals = []
                for j, label in enumerate(all_labels_str):
                    if label == excluded_label:
                        continue
                    x_vals.append(d)
                    y_vals.append(results["lvl1"][label][m])
                    c_vals.append(j * np.ones_like(d))
                x_vals = np.concatenate(x_vals)
                y_vals = np.concatenate(y_vals)
                c_vals = np.concatenate(c_vals)
            else:
                x_vals = d
                y_vals = results["macro"][m]
                c_vals = results["batch_stats"][macro_colour]
            ax.scatter(
                x=x_vals, y=y_vals, c=c_vals, s=20.0, alpha=0.3, edgecolors="none"
            )

    plt.tight_layout()
    plt.show()


def model_comparison_plot(
    train_results_x: dict[str, dict | np.ndarray],
    val_results_x: dict[str, dict | np.ndarray],
    train_results_y: dict[str, dict | np.ndarray],
    val_results_y: dict[str, dict | np.ndarray],
    target_vocab: Vocabulary,
    macro_colour: str = "context_entropy",
    exclude_raw_metrics: bool = True,
    excluded_label: str = "-",
) -> None:
    """This function plots the results of two clustering models against each other.
    For every metric and data split the results for each label and the macro results are plotted against each other.

    Args:
        train_results_x (dict[str, dict | np.ndarray]): The training results dict of the first model.
        val_results_x (dict[str, dict | np.ndarray]): The validation results dict of the first model.
        train_results_y (dict[str, dict | np.ndarray]): The training results dict of the second model.
        val_results_y (dict[str, dict | np.ndarray]): The validation results dict of the second model.
        target_vocab (Vocabulary): The vocabulary containing the target labels.
        macro_colour (str, optional): The feature to use for the colouring in the macro plots. Defaults to "context_entropy".
        exclude_raw_metrics (bool, optional): Whether to exclude the raw metrics (tp, fp, tn, fn) from the plots. Defaults to True.
        excluded_label (str, optional): The label to be excluded from plotting. This is supposed to be the false positive label. Defaults to "-".
    """

    all_labels_str = get_low_level_labels(target_vocab, 1, excluded_label)
    used_metrics = get_metrics(exclude_raw_metrics)
    fig, all_axs = get_scatter_plot_figure(
        used_metrics,
        train_results_x["model_params"]["model"]["id"],
        train_results_y["model_params"]["model"]["id"],
    )

    for j, col in enumerate(plot_cols):
        axs = all_axs[j]

        if col[0] == "train":
            results_x = train_results_x
            results_y = train_results_y
        else:
            results_x = val_results_x
            results_y = val_results_y

        for i, m in enumerate(used_metrics):
            ax = axs[i]
            if col[1] == "label":
                x_vals = []
                y_vals = []
                c_vals = []
                for j,label in enumerate(all_labels_str):
                    if label == excluded_label:
                        continue
                    x_vals.append(results_x["lvl1"][label][m])
                    y_vals.append(results_y["lvl1"][label][m])
                    c_vals.append(
                        j * np.ones_like(results_x["lvl1"][label][m])
                    )
                x_vals = np.concatenate(x_vals)
                y_vals = np.concatenate(y_vals)
                c_vals = np.concatenate(c_vals)
            else:
                x_vals = results_x["macro"][m]
                y_vals = results_y["macro"][m]
                c_vals = results_x["batch_stats"][macro_colour]
            ax.scatter(
                x=x_vals, y=y_vals, c=c_vals, s=20.0, alpha=0.3, edgecolors="none"
            )

    plt.tight_layout()
    plt.show()


def pprint_eval_report(
    train_results: dict[str, dict | np.ndarray],
    val_results: dict[str, dict | np.ndarray],
    target_vocab: Vocabulary,
    excluded_label: str = "-",
    exclude_raw_metrics: bool = True,
    hierarchical_label_levels: Iterable[int] = [0, 1],
) -> None:
    """Pretty prints the evaluation results of a clustering model."""
    level_labels = {
        0: ["macro"],
        1: get_low_level_labels(target_vocab, 1, excluded_label),
        2: get_low_level_labels(target_vocab, 2, excluded_label),
        3: get_str_labels(target_vocab),
    }
    used_metrics = get_metrics(exclude_raw_metrics)
    label_str_len = sum([5, 17, 2, 3][: max(hierarchical_label_levels) + 1])
    print(
        f"{'label':<{label_str_len}} | "
        + " | ".join([f"{m:<25}" for m in used_metrics])
    )
    for level in hierarchical_label_levels:
        level_str = f"lvl{level}" if level else "macro"
        print("-" * ((label_str_len + 1) + (28 * len(used_metrics))))
        for label in level_labels[level]:
            if label == excluded_label:
                continue
            print(
                f"{label:<{label_str_len}} | "
                + " | ".join(
                    [
                        f"{train_results['summary'][level_str][label][m][0]:<5.3f}±{
                            train_results['summary'][level_str][label][m][1]:<5.3f
                        } | {val_results['summary'][level_str][label][m][0]:<5.3f}±{
                            val_results['summary'][level_str][label][m][1]:<5.3f
                        }"
                        for m in used_metrics
                    ]
                )
            )


def pprint_eval_diff(
    train_results1: dict[str, dict | np.ndarray],
    val_results1: dict[str, dict | np.ndarray],
    train_results2: dict[str, dict | np.ndarray],
    val_results2: dict[str, dict | np.ndarray],
    target_vocab: Vocabulary,
    excluded_label: str = "-",
    exclude_raw_metrics: bool = True,
    hierarchical_label_levels: Iterable[int] = [0, 1],
) -> None:
    """Pretty prints the difference of the evaluation results of two clustering models."""
    level_labels = {
        0: ["macro"],
        1: get_low_level_labels(target_vocab, 1, excluded_label),
        2: get_low_level_labels(target_vocab, 2, excluded_label),
        3: get_str_labels(target_vocab),
    }
    used_metrics = get_metrics(exclude_raw_metrics)
    label_str_len = sum([5, 17, 2, 3][: max(hierarchical_label_levels) + 1])
    print(
        f"{'label':<{label_str_len}} | "
        + " | ".join([f"{m:<17}" for m in used_metrics])
    )
    for level in hierarchical_label_levels:
        level_str = f"lvl{level}" if level else "macro"
        print("-" * ((label_str_len + 1) + (20 * len(used_metrics))))
        for label in level_labels[level]:
            if label == excluded_label:
                continue
            print(
                f"{label:<{label_str_len}} | "
                + " | ".join(
                    [
                        f"{
                            train_results2['summary'][level_str][label][m][0]
                            - train_results1['summary'][level_str][label][m][0]:< 7.3f
                        } | {
                            val_results2['summary'][level_str][label][m][0]
                            - val_results1['summary'][level_str][label][m][0]:< 7.3f
                        }"
                        for m in used_metrics
                    ]
                )
            )


if __name__ == "__main__":
    import logging

    from alertbert.aitads import AITAlertDataset, AlertSequenceBatchSampler
    from alertbert.model_eval_utils import (
        load_data_tools,
        load_ground_truth_label_vocabs,
        load_models,
        load_reports,
    )
    from alertbert.models import CombinedTimeCosineMetric, TokenClusteringModel
    from alertbert.utils import get_device, log_to_stdout

    path = "saved_models"
    n_jobs = 8
    aitads_a_config = "more-noise-11"

    log_to_stdout()
    logging.info("Loading data...")
    train_data = AITAlertDataset(split="train", configuration=aitads_a_config)
    val_data = AITAlertDataset(split="val", configuration=aitads_a_config)
    label_vocabs = load_ground_truth_label_vocabs(path, aitads_a_config)

    # execute the following block of code to compute results for MLM based models
    if False:
        logging.info("Evaluating MLMs...")
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import KernelPCA

        # define dim reduction and clustering parameters
        cl_model_params = {
            "model": {
                "id": None,  # to be left blank here
                "layers": ["embedding", "encoder"],
            },
            "dim_reduction": {
                "name": "KernelPCA",
                "model_args": {
                    "n_components": 3,
                    "kernel": "cosine",
                },
            },
            "clustering": {
                "name": "DBSCAN",
                "model_args": {
                    "min_samples": 1,
                    "eps": 2.0,
                    "metric": "precomputed",
                },
            },
            "metric": {
                "name": "CombinedTimeCosineMetric",
                "model_args": {
                    "theta": 40.0,
                },
            },
            "data_split": None,  # to be left blank here
        }

        dim_reduction = KernelPCA(**cl_model_params["dim_reduction"]["model_args"])
        clustering = DBSCAN(**cl_model_params["clustering"]["model_args"])
        metric = CombinedTimeCosineMetric(**cl_model_params["metric"]["model_args"])

        # models to be used
        model_ids = [
            "mlm_1l_4h_base_3-1_60k",
            "mlm_1l_4h_base_3_3000",
            "mlm_1l_4h_no_bias_3000",
            "mlm_1l_4h_short3-1_30k",
            "mlm_1l_4h_1wd_long_5e-3lr_60k",
            "mlm_1l_4h_1wd_shor_5e-3lr_30k",
            "mlm_1l_4h_no_drop_60k",
            "mlm_1l_4h_no_ties_60k",
            "mlm_1l_4h_zero_0k",
            "mlm_1l_4h_nano_0k",
            "mlm_1l_4h_nano_1k",
            "mlm_1l_4h_nano_2k",
            "mlm_1l_4h_o_5k",
            "mlm_1l_4h_n_5k",
            "mlm_1l_4h_a_5k",
        ]

        reports, model_param_dicts = load_reports(model_ids, path)

        device = get_device("cpu")

        logging.info("Loading data tools...")
        data_tools = load_data_tools(model_ids, model_param_dicts, path, label_vocabs)

        logging.info("Loading models...")
        models = load_models(model_param_dicts, path, data_tools, device)

        logging.info("Setup complete.")

        for key, model in models.items():
            logging.info(f"Evaluating model {key} ...")

            assert model_param_dicts[key]["context_size"] == 4096, (
                "Context size must be 4096 for evaluation to be the same across models."
            )

            # set up data loaders
            train_sampler = AlertSequenceBatchSampler(
                train_data,
                context_size=model_param_dicts[key]["context_size"],
                batch_size=1,
                drop_last=False,
                generator=torch.Generator().manual_seed(eval_seed),
            )
            val_sampler = AlertSequenceBatchSampler(
                val_data,
                context_size=model_param_dicts[key]["context_size"],
                batch_size=1,
                drop_last=False,
                generator=torch.Generator().manual_seed(eval_seed),
            )

            train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                collate_fn=data_tools[key]["inf_coll_fn"],
            )
            val_loader = DataLoader(
                val_data,
                batch_sampler=val_sampler,
                collate_fn=data_tools[key]["inf_coll_fn"],
            )

            # set up model
            cl_model = TokenClusteringModel(
                model=MaskedLangModelInferenceWrapper(
                    model, cl_model_params["model"]["layers"]
                ),
                dim_reduction=dim_reduction,
                clustering=clustering,
                precomputed_metric=metric,
            )

            # evaluate model
            train_stats = eval_alert_grouping(
                model=cl_model,
                loader=train_loader,
                target_vocab=label_vocabs["hierarchical_event_label"],
                n_jobs=n_jobs,
            )
            val_stats = eval_alert_grouping(
                model=cl_model,
                loader=val_loader,
                target_vocab=label_vocabs["hierarchical_event_label"],
                n_jobs=n_jobs,
            )

            # add model params to results and save results
            # ATTENTION: The cl_model_params dict is modified in place, so the order of operations is important
            cl_model_params["model"]["id"] = key

            train_stats["model_params"] = cl_model_params
            train_stats["model_params"]["data_split"] = "train"
            save_results(
                train_stats,
                path,
                train_stats["model_params"]["dim_reduction"]["name"]
                + f"_theta_{cl_model_params['metric']['model_args']['theta']:d}_{aitads_a_config}",
            )

            val_stats["model_params"] = cl_model_params
            val_stats["model_params"]["data_split"] = "val"
            save_results(
                val_stats,
                path,
                val_stats["model_params"]["dim_reduction"]["name"]
                + f"_theta_{cl_model_params['metric']['model_args']['theta']:d}_{aitads_a_config}",
            )

    # execute the following block of code to compute results for time delta models
    if True:
        logging.info("Evaluating TimeDelta models...")

        # define parameters
        cl_model_params = {
            "model": {
                "id": "timedelta",
                "delta": None,  # to be left blank here
            },
            "data_split": None,  # to be left blank here
            "context_size": 4096, # Do not change this value! It must be the same for all models so that the evaluation is comparable.
        }

        # deltas to be used
        deltas = [1, 2, 4]

        logging.info("Setup complete.")

        for delta in deltas:
            logging.info(f"Evaluating delta = {delta} ...")

            # set up data loaders
            train_sampler = AlertSequenceBatchSampler(
                train_data,
                context_size=cl_model_params["context_size"],
                batch_size=1,
                drop_last=False,
                generator=torch.Generator().manual_seed(eval_seed),
            )
            val_sampler = AlertSequenceBatchSampler(
                val_data,
                context_size=cl_model_params["context_size"],
                batch_size=1,
                drop_last=False,
                generator=torch.Generator().manual_seed(eval_seed),
            )

            coll_fn = BaseSequenceCollate(label_vocabs)
            train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                collate_fn=coll_fn,
            )
            val_loader = DataLoader(
                val_data,
                batch_sampler=val_sampler,
                collate_fn=coll_fn,
            )

            # set up model
            cl_model = TimeDeltaClusteringModel(delta=delta)

            # evaluate model
            train_stats = eval_alert_grouping(
                model=cl_model,
                loader=train_loader,
                target_vocab=label_vocabs["hierarchical_event_label"],
                n_jobs=n_jobs,
            )
            val_stats = eval_alert_grouping(
                model=cl_model,
                loader=val_loader,
                target_vocab=label_vocabs["hierarchical_event_label"],
                n_jobs=n_jobs,
            )

            # add model params to results and save results
            # ATTENTION: The cl_model_params dict is modified in place, so the order of operations is important
            cl_model_params["model"]["delta"] = delta

            train_stats["model_params"] = cl_model_params
            train_stats["model_params"]["data_split"] = "train"
            save_results(train_stats, path, f"timedelta_{delta}_{aitads_a_config}")

            val_stats["model_params"] = cl_model_params
            val_stats["model_params"]["data_split"] = "val"
            save_results(val_stats, path, f"timedelta_{delta}_{aitads_a_config}")

    logging.info("Done.")
