import pickle
from collections import Counter
from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

from alertbert.aitads import MultiAlertDataset
from alertbert.models import (
    AbstractDatasetGroupingModel,
    MaskedLangModelInferenceWrapper,
)
from alertbert.preprocessing import Vocabulary

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


if __name__ == "__main__":
    import logging

    from alertbert.aitads import AITAlertDataset
    from alertbert.model_eval_utils import (
        load_data_tools,
        load_ground_truth_label_vocabs,
        load_models,
        load_reports,
    )
    from alertbert.models import AlertBERT, TimeDelta
    from alertbert.utils import get_device, log_to_stdout

    path = "saved_models"
    aitads_a_config = "original"  # ["original", "simul-attacks", "more-noise-1", "more-noise-2", "more-noise-6", "more-noise-11"]

    log_to_stdout()
    logging.info(f"Loading data config {aitads_a_config} ...")
    train_data = AITAlertDataset(split="train", configuration=aitads_a_config)
    val_data = AITAlertDataset(split="val", configuration=aitads_a_config)
    label_vocabs = load_ground_truth_label_vocabs(path, aitads_a_config)

    # execute the following block of code to compute results for MLM based models
    if True:
        logging.info("Evaluating MLMs...")

        # define dim reduction and clustering parameters
        grouping_model_params = {
            "model": {
                "id": None,  # to be left blank here
                "layers": ["embedding", "encoder"],
                "theta": 6.0,
                "delta": 2.0,
                "dim_reduction": 2,
            },
            "data_split": None,  # to be left blank here
        }

        # models to be used
        model_ids = [
            "mlm_1l_4h_zero_0k",
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

            # set up model
            grouping_model = AlertBERT(
                model=MaskedLangModelInferenceWrapper(
                    model, grouping_model_params["model"]["layers"]
                ),
                collate_fn=data_tools[key]["inf_coll_fn"],
                dim_reduction=grouping_model_params["model"]["dim_reduction"],
                delta=grouping_model_params["model"]["delta"],
                theta=grouping_model_params["model"]["theta"],
            )

            # evaluate model
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

            # add model params to results and save results
            # ATTENTION: The grouping_model_params dict is modified in place, so the order of operations is important
            grouping_model_params["model"]["id"] = key

            suffix = "_noise"

            train_stats_noise["model_params"] = grouping_model_params
            train_stats_noise["model_params"]["data_split"] = "train"
            save_results(
                train_stats_noise,
                path,
                f"{grouping_model_params['model']['dim_reduction']}dim"
                + f"_theta_{grouping_model_params['model']['theta']}"
                + f"_delta_{grouping_model_params['model']['delta']}"
                + f"_{aitads_a_config}"
                + suffix,
            )

            val_stats_noise["model_params"] = grouping_model_params
            val_stats_noise["model_params"]["data_split"] = "val"
            save_results(
                val_stats_noise,
                path,
                f"{grouping_model_params['model']['dim_reduction']}dim"
                + f"_theta_{grouping_model_params['model']['theta']}"
                + f"_delta_{grouping_model_params['model']['delta']}"
                + f"_{aitads_a_config}"
                + suffix,
            )

            suffix = "_clean"

            train_stats_clean["model_params"] = grouping_model_params
            train_stats_clean["model_params"]["data_split"] = "train"
            save_results(
                train_stats_clean,
                path,
                f"{grouping_model_params['model']['dim_reduction']}dim"
                + f"_theta_{grouping_model_params['model']['theta']}"
                + f"_delta_{grouping_model_params['model']['delta']}"
                + f"_{aitads_a_config}"
                + suffix,
            )

            val_stats_clean["model_params"] = grouping_model_params
            val_stats_clean["model_params"]["data_split"] = "val"
            save_results(
                val_stats_clean,
                path,
                f"{grouping_model_params['model']['dim_reduction']}dim"
                + f"_theta_{grouping_model_params['model']['theta']}"
                + f"_delta_{grouping_model_params['model']['delta']}"
                + f"_{aitads_a_config}"
                + suffix,
            )
    # execute the following block of code to compute results for time delta models
    if False:
        logging.info("Evaluating TimeDelta models...")

        # define parameters
        grouping_model_params = {
            "model": {
                "id": "timedelta",
                "delta": None,  # to be left blank here
            },
            "data_split": None,  # to be left blank here
        }

        # deltas to be used
        deltas = [1, 2, 4]

        logging.info("Setup complete.")

        for delta in deltas:
            logging.info(f"Evaluating delta = {delta} ...")

            # set up model
            grouping_model = TimeDelta(delta=delta)

            # evaluate model
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

            # add model params to results and save results
            # ATTENTION: The grouping_model_params dict is modified in place, so the order of operations is important
            grouping_model_params["model"]["delta"] = delta

            suffix = "_noise"

            train_stats_noise["model_params"] = grouping_model_params
            train_stats_noise["model_params"]["data_split"] = "train"
            save_results(
                train_stats_noise, path, f"timedelta_{delta}_{aitads_a_config}" + suffix
            )

            val_stats_noise["model_params"] = grouping_model_params
            val_stats_noise["model_params"]["data_split"] = "val"
            save_results(
                val_stats_noise, path, f"timedelta_{delta}_{aitads_a_config}" + suffix
            )

            suffix = "_clean"

            train_stats_clean["model_params"] = grouping_model_params
            train_stats_clean["model_params"]["data_split"] = "train"
            save_results(
                train_stats_clean, path, f"timedelta_{delta}_{aitads_a_config}" + suffix
            )

            val_stats_clean["model_params"] = grouping_model_params
            val_stats_clean["model_params"]["data_split"] = "val"
            save_results(
                val_stats_clean, path, f"timedelta_{delta}_{aitads_a_config}" + suffix
            )

    logging.info("Done.")
