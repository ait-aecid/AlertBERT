import logging
import pickle
from collections import Counter
from collections.abc import Iterable
from typing import Literal

from debugpy import log_to
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

from alertbert.aitads import AITAlertDataset, MultiAlertDataset
from alertbert.model_eval_utils import (
    load_data_tools,
    load_ground_truth_label_vocabs,
    load_models,
    load_reports,
)
from alertbert.models import (
    AbstractDatasetGroupingModel,
    AlertBERT,
    MaskedLangModelInferenceWrapper,
    TimeDelta,
)
from alertbert.preprocessing import Vocabulary
from alertbert.utils import get_device, log_to_stdout

"""This module contains functions for evaluating alert grouping models.
If executed as a script, it will load a trained model and evaluate it on the training and validation sets of the specified augmentation of the AIT Alert dataset.
"""

np.seterr(all="raise")


# utility functions


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


def eval_alert_grouping(
    model: AbstractDatasetGroupingModel = None,
    target: str = "hierarchical_event_label",
    target_vocab: Vocabulary = None,
    data: MultiAlertDataset = None,
    contingency_matrices: list[np.ndarray[int]] = None,
    excluded_macro_label: str = "-",
    ignore_excluded_macro_label: bool = True,
) -> tuple[dict[str, dict | np.ndarray], list[np.ndarray[int]]]:
    """This function computes multiple evaluation metrics for an alert grouping model on the given dataset.
    For every scenario the metrics are computed for every label in the dataset and the macro metrics are
    computed over all labels except the excluded label (which is supposed to be the false positive label).
    Alternatively to the model and data also already computed contingency matrices can be provided.
    This function only supports hierarchical labels!

    Args:
        model (AbstractDatasetGroupingModel): The alert grouping model to be evaluated.
            Only used if contingency_matrices is None.
        target (str, optional): The target label in the dataset. Defaults to "hierarchical_event_label".
        target_vocab (Vocabulary): The vocabulary containing the target labels.
        data (MultiAlertDataset): The dataset to be evaluated. Only used if contingency_matrices is None.
        contingency_matrices (list[np.ndarray[int]], optional): The contingency matrices to be used for evaluation.
            If None, the model and data will be used to compute the matrices.
        excluded_macro_label (str, optional): The label to exclude from macro calculations. Defaults to "-".
        ignore_excluded_macro_label (bool, optional): Whether to ignore the samples belonging to the
            excluded macro label in the results. Defaults to True.

    Returns:
        tuple[dict[str, dict | np.ndarray], list[np.ndarray[int]]]: A tuple containing the results dictionary and the contingency matrices.
            The results dictionary contains the metrics for every label in the dataset and the macro metrics.
            The contingency matrices are the ones used for evaluation.
    """

    assert target.startswith("hierarchical"), (
        "Non-hierarchical labels have been deprecated in this function."
    )
    if contingency_matrices is None:
        assert model is not None and data is not None, (
            "Either contingency matrices or model and data must be provided."
        )

    # set up labels
    all_labels_str = get_str_labels(target_vocab)  # level 3 labels
    all_labels_int = get_labels_int(all_labels_str, target_vocab)
    label_range = (all_labels_int[0], all_labels_int[-1])

    lvl_2_labels = get_low_level_labels(target_vocab, 2, excluded_macro_label)
    lvl_1_labels = get_low_level_labels(target_vocab, 1, excluded_macro_label)

    if ignore_excluded_macro_label:
        assert all_labels_str[0] == excluded_macro_label

    # compute contingency matrices if they are not provided
    if contingency_matrices is None:
        contingency_matrices = []

        for scenario in data.scenarios:
            pred = model(scenario).squeeze()
            true = target_vocab([scenario.data[target]]).numpy().squeeze()
            contingency_matrices.append(contingency_matrix(true, pred, label_range))

        del pred, true

    # initialize results dict
    results = {
        # these will store the (aggregated) metrics for every scenario in the dataset
        "lvl3": {label: {metric: [] for metric in metrics} for label in all_labels_str},
        "lvl2": {label: {metric: None for metric in metrics} for label in lvl_2_labels},
        "lvl1": {label: {metric: None for metric in metrics} for label in lvl_1_labels},
        "macro": {metric: None for metric in metrics[1:]},  # aka level 0
        # meta data
        "model_params": None,
        # these will store the (aggregated) metrics averaged over all scenarios
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

    for counts in contingency_matrices:
        assert counts.shape[0] == len(all_labels_int)

        # throw away the counts for the ignored label
        if ignore_excluded_macro_label:
            counts = counts[1:]

        cluster_sizes = counts.sum(axis=0)
        true_label_counts = counts.sum(axis=1)
        total = true_label_counts.sum()

        for label, int_label in zip(all_labels_str, all_labels_int):
            idx = int_label - label_range[0]
            if ignore_excluded_macro_label:
                idx -= 1

            # continue if this is the ignored label
            if ignore_excluded_macro_label and label == excluded_macro_label:
                for v in results["lvl3"][label].values():
                    v.append(np.nan)
                continue

            # continue if label does not appear in scenario
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

    return results, contingency_matrices


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
    model_id = results["model_params"]["id"]
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


def get_eval_file_name(
    grouping_model_params: dict, aitads_a_config: str, noise: bool = True
) -> str:
    suffix = "_noise" if noise else "_clean"
    if grouping_model_params["id"] == "timedelta":
        return (
            f"timedelta_{grouping_model_params['delta']}_{aitads_a_config}"
            + suffix
        )
    else:
        return (
            f"{len(grouping_model_params['layers'])}l_"
            + f"{grouping_model_params['dim_reduction']}dim"
            + f"_theta_{grouping_model_params['theta']}"
            + f"_delta_{grouping_model_params['delta']}"
            + f"_{aitads_a_config}"
            + suffix
        )


# roc curve functions


def compute_roc_trajectories(
    model_id: str,
    aitads_a_config: Literal[
        "original",
        "simul-attacks",
        "more-noise-1",
        "more-noise-2",
        "more-noise-6",
        "more-noise-11",
    ],
    deltas: list[float],
    thetas: list[float] = None,
    layers: tuple[str] = ("embedding", "encoder"),
    dim_reduction: int = 2,
    path: str = "saved_models",
) -> None:
    """Computes the ROC trajectories for the given model and data.

    Args:
        model_id (str): The id of the model.
        aitads_a_config (Literal): The configuration of the AIT-ADS-A dataset.
        deltas (list[float]): The delta values to be used for the models.
        thetas (list[float], optional): The theta values to be used for the models. Defaults to None.
        layers (tuple[str], optional): The layers to be used for the models. Defaults to ("embedding", "encoder").
        dim_reduction (int, optional): The dimensionality reduction to be used for the models. Defaults to 2.
        path (str, optional): The path to the directory where the respective model is located. Defaults to "saved_models".
    """
    if model_id != "timedelta":
        assert thetas is not None, "Theta values must be provided for AlertBert models."
        assert len(deltas) == len(thetas), (
            "Delta and theta values must have the same length."
        )
    else:
        thetas = [None] * len(deltas)

    log_to_stdout()

    not_found_results = []

    for delta, theta in zip(deltas, thetas):
        grouping_model_params = get_grouping_model_params(
            model_id,
            delta,
            theta,
            layers,
            dim_reduction,
            data_split="train",
        )
        file_name = get_eval_file_name(
            grouping_model_params, aitads_a_config, noise=True
        )
        logging.info(f"Searching for {file_name}")
        try:
            load_results(path=path, model_id=model_id, name=file_name, split="train")
            logging.info("Found!")
        except FileNotFoundError:
            logging.info("File not found, will compute results ...")
            not_found_results.append([delta, theta])

    deltas = [i[0] for i in not_found_results]
    thetas = [i[1] for i in not_found_results]
    main(
        model_ids=[model_id],
        aitads_a_config=aitads_a_config,
        deltas=deltas,
        thetas=thetas,
        layers=layers,
        dim_reduction=dim_reduction,
        path=path,
    )

    logging.info("All results computed!")


def roc(
    model_class: type[AbstractDatasetGroupingModel],
    trajectories: dict[str, np.ndarray],
    target: str = "hierarchical_event_label",
    highlight_result=None,
    label_vocabs: dict[str, Vocabulary] = None,
):
    target_vocab = label_vocabs[target]
    all_labels_str = get_str_labels(target_vocab)
    all_labels_int = get_labels_int(all_labels_str, target_vocab)
    label_range = (all_labels_int[0], all_labels_int[-1])

    l = [len(t) for t in trajectories.values()]
    assert all([x == l[0] for x in l]), "All trajectories must have the same length"
    l = l[0]

    result_trajectory = []

    for i in range(l):
        model = model_class(**{k: v[i] for k, v in trajectories.items()})
        result_trajectory.append(eval_alert_grouping(model, target))

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    cmap = plt.get_cmap("viridis")

    ax.set_box_aspect(1)
    ax.grid()
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(f"{model_class.__name__} ROC Curve")
    ax.set_xlabel("True Negative Rate")
    ax.set_ylabel("True Positive Rate")

    for i, label in enumerate(all_labels_str):
        tpr = []
        tnr = []
        for result in result_trajectory:
            tpr.append(result["summary"][label]["recall"])
            tnr.append(result["summary"][label]["tnr"])

        ax.plot(
            tnr,
            tpr,
            label=label,
            color=cmap(
                (all_labels_int[i] - label_range[0]) / (label_range[1] - label_range[0])
            ),
            alpha=0.5,
        )

        if highlight_result:
            ax.scatter(
                highlight_result["summary"][label]["tnr"],
                highlight_result["summary"][label]["recall"],
                color=cmap(
                    (all_labels_int[i] - label_range[0])
                    / (label_range[1] - label_range[0])
                ),
                marker="x",
            )

    tpr = []
    tnr = []
    for result in result_trajectory:
        tpr.append(result["summary"]["macro"]["recall"])
        tnr.append(result["summary"]["macro"]["tnr"])
    ax.plot(tnr, tpr, label="macro", color="r")

    if highlight_result:
        ax.scatter(
            highlight_result["summary"]["macro"]["tnr"],
            highlight_result["summary"]["macro"]["recall"],
            color="r",
            marker="x",
        )

    ax.legend()

    plt.tight_layout()
    plt.show()


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
        figsize=(4 * len(plot_cols), 4.5 * len(used_metrics)),
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
                ax.set_xlim((-0.05 if m != "mcc" else -1.05), 1.05)

    return fig, all_axs


def model_comparison_plot(
    train_results_x: dict[str, dict | np.ndarray],
    val_results_x: dict[str, dict | np.ndarray],
    train_results_y: dict[str, dict | np.ndarray],
    val_results_y: dict[str, dict | np.ndarray],
    target_vocab: Vocabulary,
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
        train_results_x["model_params"]["id"],
        train_results_y["model_params"]["id"],
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
                for j, label in enumerate(all_labels_str):
                    if label == excluded_label:
                        continue
                    x_vals.append(results_x["lvl1"][label][m])
                    y_vals.append(results_y["lvl1"][label][m])
                    c_vals.append(j * np.ones_like(results_x["lvl1"][label][m]))
                x_vals = np.concatenate(x_vals)
                y_vals = np.concatenate(y_vals)
                c_vals = np.concatenate(c_vals)
            else:
                x_vals = results_x["macro"][m]
                y_vals = results_y["macro"][m]
                c_vals = None  # results_x["batch_stats"][macro_colour]
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


# main functions


def get_grouping_model_params(
    model_id: str,
    delta: float,
    theta: float = None,
    layers: tuple[str] = ("embedding", "encoder"),
    dim_reduction: int = 2,
    data_split: str = None,
) -> dict:
    """Returns the parameters for the grouping model.

    Args:
        model_id (str): The id of the model.
        delta (float): The delta value for the model.
        theta (float, optional): The theta value for the model. Defaults to None.
        layers (tuple[str], optional): The layers to be used for the model. Defaults to ("embedding", "encoder").
        dim_reduction (int, optional): The dimensionality reduction to be used for the model. Defaults to 2.
        data_split (str, optional): The data split to be used for the model. Defaults to None.
    """
    if model_id == "timedelta":
        return {
            "id": "timedelta",
            "delta": delta,
            "data_split": data_split,
        }
    elif model_id.startswith("mlm"):
        return {
            "id": model_id,
            "layers": layers,
            "theta": theta,
            "delta": delta,
            "dim_reduction": dim_reduction,
            "data_split": data_split,
        }
    else:
        raise ValueError(f"Encountered invalid model id: {model_id}.")


def compute_all_eval_results(
    grouping_model: AbstractDatasetGroupingModel,
    label_vocabs: dict,
    train_data: AITAlertDataset,
    val_data: AITAlertDataset,
) -> tuple[dict, dict, dict, dict]:
    """Computes all evaluation results for the given model and data.

    Args:
        grouping_model (AbstractDatasetGroupingModel): The alert grouping model to be evaluated.
        label_vocabs (dict): The vocabularies containing the target labels.
        train_data (AITAlertDataset): The training dataset to be evaluated.
        val_data (AITAlertDataset): The validation dataset to be evaluated.
    """
    train_stats_noise, cont_matrices = eval_alert_grouping(
        model=grouping_model,
        target_vocab=label_vocabs["hierarchical_event_label"],
        data=train_data,
        ignore_excluded_macro_label=False,
    )
    train_stats_clean, _ = eval_alert_grouping(
        target_vocab=label_vocabs["hierarchical_event_label"],
        contingency_matrices=cont_matrices,
    )
    val_stats_noise, cont_matrices = eval_alert_grouping(
        model=grouping_model,
        target_vocab=label_vocabs["hierarchical_event_label"],
        data=val_data,
        ignore_excluded_macro_label=False,
    )
    val_stats_clean, _ = eval_alert_grouping(
        target_vocab=label_vocabs["hierarchical_event_label"],
        contingency_matrices=cont_matrices,
    )
    return (
        train_stats_noise,
        val_stats_noise,
        train_stats_clean,
        val_stats_clean,
    )


def save_all_eval_results(
    train_stats_noise: dict,
    val_stats_noise: dict,
    train_stats_clean: dict,
    val_stats_clean: dict,
    grouping_model_params: dict,
    aitads_a_config: str,
    path: str,
) -> None:
    """Saves all evaluation results to pickle files.

    Args:
        train_stats_noise (dict): The training results dict for the noisy results.
        val_stats_noise (dict): The validation results dict for the noisy results.
        train_stats_clean (dict): The training results dict for the clean results.
        val_stats_clean (dict): The validation results dict for the clean results.
        aitads_a_config (str): The configuration of the AIT-ADS-A dataset.
        path (str): The path to the directory where the respective model is located.
    """
    train_stats_noise["model_params"] = grouping_model_params
    train_stats_noise["model_params"]["data_split"] = "train"
    save_results(
        train_stats_noise,
        path,
        get_eval_file_name(grouping_model_params, aitads_a_config, noise=True),
    )

    val_stats_noise["model_params"] = grouping_model_params
    val_stats_noise["model_params"]["data_split"] = "val"
    save_results(
        val_stats_noise,
        path,
        get_eval_file_name(grouping_model_params, aitads_a_config, noise=True),
    )

    train_stats_clean["model_params"] = grouping_model_params
    train_stats_clean["model_params"]["data_split"] = "train"
    save_results(
        train_stats_clean,
        path,
        get_eval_file_name(grouping_model_params, aitads_a_config, noise=False),
    )

    val_stats_clean["model_params"] = grouping_model_params
    val_stats_clean["model_params"]["data_split"] = "val"
    save_results(
        val_stats_clean,
        path,
        get_eval_file_name(grouping_model_params, aitads_a_config, noise=False),
    )


def main(
    model_ids: list[str],
    aitads_a_config: Literal[
        "original",
        "simul-attacks",
        "more-noise-1",
        "more-noise-2",
        "more-noise-6",
        "more-noise-11",
    ],
    deltas: list[float],
    thetas: list[float] = None,
    layers: tuple[str] = ("embedding", "encoder"),
    dim_reduction: int = 2,
    path: str = "saved_models",
) -> None:
    """Main function for evaluating alert grouping models.
    Loads the specified models and evaluates them on the training and validation sets of the specified augmentation of the AIT Alert dataset.
    It is possible to either evaluate multiple models with the same delta and theta values or to evaluate a single model with different delta and theta values.
    TimeDelta models can only be evaluated in the single model case.

    Args:
        model_ids (list[str]): The ids of the models to be evaluated.
        aitads_a_config (Literal): The configuration of the AIT-ADS-A dataset.
        deltas (list[float]): The delta values to be used for the models.
        thetas (list[float], optional): The theta values to be used for the models. Defaults to None.
        layers (tuple[str], optional): The layers to be used for the models. Defaults to ("embedding", "encoder").
        dim_reduction (int, optional): The dimensionality reduction to be used for the models. Defaults to 2.
        path (str, optional): The path to the directory where the respective model is located. Defaults to "saved_models".
    """
    if len(model_ids) == 1 and model_ids[0] == "timedelta":
        timedelta = True
    elif "timedelta" in model_ids:
        raise ValueError(
            "TimeDelta models cannot be evaluated together with AlertBert models."
        )
    else:
        timedelta = False
        assert thetas is not None, "Theta values must be provided for AlertBert models."

    if len(model_ids) > 1:
        assert len(deltas) == 1, "Only one delta value can be used for multiple models."
        assert len(thetas) == 1, "Only one theta value can be used for multiple models."
        deltas = deltas * len(model_ids)
        thetas = thetas * len(model_ids)
    else:
        if not timedelta:
            assert len(deltas) == len(thetas), (
                "Delta and theta values must have the same length."
            )

    log_to_stdout()
    logging.info(f"Loading data config {aitads_a_config} ...")
    train_data = AITAlertDataset(split="train", configuration=aitads_a_config)
    val_data = AITAlertDataset(split="val", configuration=aitads_a_config)
    label_vocabs = load_ground_truth_label_vocabs(path, aitads_a_config)

    if timedelta:
        logging.info("Evaluating TimeDelta models...")
    else:
        logging.info("Evaluating AlertBert models...")
        reports, model_param_dicts = load_reports(model_ids, path)
        device = get_device("cpu")

        logging.info("Loading data tools...")
        data_tools = load_data_tools(model_ids, model_param_dicts, path, label_vocabs)

        logging.info("Loading models...")
        models = load_models(model_param_dicts, path, data_tools, device)

    logging.info("Setup complete.")

    for key in model_ids:
        for i in range(len(deltas)):
            if timedelta:
                grouping_model_params = get_grouping_model_params(
                    model_id="timedelta",
                    delta=deltas[i],
                )
                logging.info(f"Evaluating delta = {deltas[i]} ...")
                grouping_model = TimeDelta(delta=deltas[i])
            else:
                logging.info(f"Evaluating model {key} ...")
                grouping_model_params = get_grouping_model_params(
                    model_id=key,
                    delta=deltas[i],
                    theta=thetas[i],
                    layers=layers,
                    dim_reduction=dim_reduction,
                )
                grouping_model = AlertBERT(
                    model=MaskedLangModelInferenceWrapper(models[key], layers),
                    collate_fn=data_tools[key]["inf_coll_fn"],
                    dim_reduction=dim_reduction,
                    delta=deltas[i],
                    theta=thetas[i],
                )

            # evaluate model
            results = compute_all_eval_results(
                grouping_model, label_vocabs, train_data, val_data
            )

            # save results
            save_all_eval_results(
                *results,
                grouping_model_params,
                aitads_a_config,
                path,
            )

    logging.info("Done.")


if __name__ == "__main__":
    import gc

    # main(model_ids=["mlm_1l_4h_16d_zero_0k"], aitads_a_config="more-noise-11", deltas=[2.0], thetas=[6.0])

    for config in [
        "original",
        "simul-attacks",
        "more-noise-1",
        "more-noise-2",
        "more-noise-6",
        "more-noise-11",
    ]:
        compute_roc_trajectories(
            model_id="timedelta",
            aitads_a_config=config,
            deltas=[0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
        )
        gc.collect()

        compute_roc_trajectories(
            model_id="mlm_1l_4h_16d_zero_0k",
            aitads_a_config=config,
            deltas=[2.0],
            thetas=[3.0, 6.0, 12.0, 24.0, 48.0],
        )
        gc.collect()

        compute_roc_trajectories(
            model_id="mlm_1l_4h_16d_zero_0k",
            aitads_a_config=config,
            deltas=[4.0],
            thetas=[3.0, 6.0, 12.0, 24.0, 48.0],
        )
        gc.collect()
