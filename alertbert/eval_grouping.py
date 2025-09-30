import gc
import logging
import pickle
from collections import Counter
from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
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
from alertbert.utils import log_to_stdout, set_up_log

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
            gc.collect()
            # logging.info("Finished scenario!")

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
        return f"timedelta_{grouping_model_params['delta']}_{aitads_a_config}" + suffix
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

timedelta_roc_traj_primary = [2.0**i for i in range(-7, 13)]
timedelta_roc_traj_secondary = [2.0**i * 1.5 for i in range(-7, 12)]
timedelta_roc_traj_all = timedelta_roc_traj_primary + timedelta_roc_traj_secondary

alertbert_deltas = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
alertbert_theta_roc_traj_primary = [2.0**i for i in range(7)]
alertbert_theta_roc_traj_secondary = [2.0**i for i in range(-7, 0)] + [2.0**i for i in range(7, 13)]
alertbert_theta_roc_traj_tertiary = [2.0**i * 1.5 for i in range(6)]
alertbert_theta_roc_traj_quartary = [2.0**i * 1.5 for i in range(-7, 0)] + [2.0**i * 1.5 for i in range(6, 12)]
alertbert_theta_roc_traj_all = (
    alertbert_theta_roc_traj_primary
    + alertbert_theta_roc_traj_secondary
    + alertbert_theta_roc_traj_tertiary
    + alertbert_theta_roc_traj_quartary
)

all_delta_theta_vals = sorted(timedelta_roc_traj_all)


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
    test_mode: bool = False,
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
        test_mode (bool, optional): Whether to compute the results on the test set instead of the training/validation set. Defaults to False.
    """
    if model_id != "timedelta":
        assert thetas is not None, "Theta values must be provided for AlertBert models."
        if len(deltas) > 1 and len(thetas) > 1:
            assert len(deltas) == len(thetas), (
                f"Delta and theta values must have the same length if it is not 1, found {len(deltas)} and {len(thetas)}."
            )
        elif len(deltas) == 1 and len(thetas) > 1:
            deltas = [deltas[0]] * len(thetas)
        elif len(deltas) > 1 and len(thetas) == 1:
            thetas = [thetas[0]] * len(deltas)
    else:
        thetas = [None] * len(deltas)

    set_up_log(f"{path}/eval")
    logging.info(
        f"{model_id:<33} - Checking ROC trajectory for model {model_id} on data config {aitads_a_config} ..."
    )
    if test_mode:
        logging.info(f"{model_id:<33} - Test mode enabled, using test data!")

    not_found_results = []

    for delta, theta in zip(deltas, thetas):
        if theta is not None and theta < delta:
            logging.info(
                f"{model_id:<33} - Adapting delta {delta} for theta {theta} because theta < delta."
            )
            delta = theta
        grouping_model_params = get_grouping_model_params(
            model_id,
            delta,
            theta,
            layers,
            dim_reduction,
            data_split="train" if not test_mode else "test",
        )
        file_name = get_eval_file_name(
            grouping_model_params, aitads_a_config, noise=True
        )
        logging.info(f"{model_id:<33} - Searching for {file_name}")
        try:
            load_results(path=path, model_id=model_id, name=file_name, split="train" if not test_mode else "test")
            logging.info(f"{model_id:<33} - Found!")
        except FileNotFoundError:
            logging.info(f"{model_id:<33} - Not found, will compute results ...")
            not_found_results.append([delta, theta])

    if len(not_found_results) > 0:
        logging.info(
            f"{model_id:<33} - Found {len(not_found_results)} results to compute ..."
        )
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
            test_mode=test_mode,
        )
        logging.info(f"{model_id:<33} - All results computed!")
    else:
        logging.info(f"{model_id:<33} - Nothing found to compute!")


def compute_auc_score(x: Iterable[float], y: Iterable[float]) -> float:
    """Computes the area under the curve (AUC) for the given x and y coordinates of an ROC curve.

    Args:
        x (Iterable[float]): The x coordinates.
        y (Iterable[float]): The y coordinates.

    Returns:
        float: The computed AUC value.
    """
    assert len(x) == len(y)
    if len(x) == 0:
        return np.nan
    s = x[0] * y[0]
    for i in range(1, len(x)):
        s += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return s


def get_relevant_roc_results(
    results: list[dict[str, dict | np.ndarray]],
    label: str = "macro",
) -> np.ndarray[int]:
    """From a list of clustering results this function identifies the results defining the ROC curve.

    Args:
        results (list[dict[str, dict | np.ndarray]]): The list of results dictionaries
            containing the evaluation metrics for each result.
        label (str, optional): The label for which the ROC curve should be computed.

    Returns:
        np.ndarray[int]: An array of indices of the relevant results that define the ROC curve.
    """

    all_valid_results = []  # (idx, tnr, recall)

    for i, result in enumerate(results):
        if label == "macro":
            tnr = result["summary"]["macro"]["macro"]["tnr"][0]
            recall = result["summary"]["macro"]["macro"]["recall"][0]
        else:
            tnr = result["summary"]["lvl1"][label]["tnr"][0]
            recall = result["summary"]["lvl1"][label]["recall"][0]
        if not np.isnan(tnr) and not np.isnan(recall):
            all_valid_results.append((i, tnr, recall))

    if len(all_valid_results) == 0:
        return np.array([], dtype=int)

    relevant_results = set([all_valid_results[0][0]])

    for result in all_valid_results[1:]:
        results_to_remove = set()
        for prev_result in relevant_results:
            if (
                result[1] < all_valid_results[prev_result][1]
                and result[2] < all_valid_results[prev_result][2]
            ):
                break  # this result is not relevant
            elif (
                all_valid_results[prev_result][1] < result[1]
                and all_valid_results[prev_result][2] < result[2]
            ):
                results_to_remove.add(prev_result)
        else:
            relevant_results.add(result[0])
            relevant_results -= results_to_remove

    return np.array(sorted(relevant_results))


def pair_iterator(iterable: Iterable) -> Iterable:
    """Yields each item in the iterable twice."""
    for i in iterable:
        yield i
        yield i


def load_roc_results(
    model_id: str,
    layers: tuple[str],
    dim_reduction: int,
    aitads_a_config: str,
    noise: bool,
    split: str,
    path: str,
) -> list[dict[str, dict | np.ndarray]]:
    """Retreives all computed results for the given model and data configuration.

    Args:
        model_id (str): The id of the model. Can be "timedelta" or "mlm_*".
        layers (tuple[str]): The layers to be used for the AlertBert models.
        dim_reduction (int): The dimensionality reduction to be used for the AlertBert models.
        aitads_a_config (Literal): The configuration of the AIT-ADS-A dataset.
        noise (bool): Whether to load results including or excluding false positive alerts.
        split (str): The data split to load results for. Can be "train" or "val".
        path (str): The path to the directory where the respective model is located.

    Returns:
        list[dict[str, dict | np.ndarray]]: A list of results dictionaries for the given model and data configuration.
    """

    results = []
    seen_results = set()

    for delta in all_delta_theta_vals:
        for theta in all_delta_theta_vals:
            grouping_model_params = get_grouping_model_params(
                model_id,
                delta,
                theta,
                layers,
                dim_reduction,
                data_split=split,
            )
            file_name = get_eval_file_name(
                grouping_model_params, aitads_a_config, noise=noise
            )
            if file_name not in seen_results:
                try:  # noqa: SIM105
                    results.append(
                        load_results(
                            path=path,
                            model_id=model_id,
                            name=file_name,
                            split=split,
                        )
                    )
                except FileNotFoundError:
                    pass
            seen_results.add(file_name)
    return results


def roc_plot(
    model_id: str,
    aitads_a_config: Literal[
        "original",
        "simul-attacks",
        "more-noise-1",
        "more-noise-2",
        "more-noise-6",
        "more-noise-11",
    ],
    layers: tuple[str] = ("embedding", "encoder"),
    dim_reduction: int = 2,
    path: str = "saved_models",
    verbose: bool = False,
    target: str = "hierarchical_event_label",
    plot_macro_only: bool = False,
    label_vocabs: dict[str, Vocabulary] = None,
) -> None:
    """Plots the ROC curves for the given AlertBert or TimeDelta model and data.
    The figure has 4 subplots, one each for training and validation data where the results were computed including/excluding the false positive alerts.

    Args:
        model_id (str): The id of the model. Can be "timedelta" or "mlm_*".
        aitads_a_config (Literal): The configuration of the AIT-ADS-A dataset.
        layers (tuple[str], optional): The layers to be used for the AlertBert models. Defaults to ("embedding", "encoder").
        dim_reduction (int, optional): The dimensionality reduction to be used for the AlertBert models. Defaults to 2.
        path (str, optional): The path to the directory where the respective model is located. Defaults to "saved_models".
        verbose (bool, optional): Whether to print additional information about the relevant results defining the ROC curve. Defaults to False.
    """
    if not plot_macro_only:
        lvl_1_labels = get_low_level_labels(label_vocabs[target], 1)
    cmap = plt.get_cmap("viridis")

    # create the figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10.5), sharex=True, sharey=True)
    title_str = f"ROC plots for: model = {model_id}, data = {aitads_a_config}"
    if model_id != "timedelta":
        title_str += f", {'input' if layers == 1 else 'output'} embeddings, {dim_reduction} dimensions"
    fig.suptitle(title_str)

    for row in range(2):
        for col in range(2):
            split = "train" if not col else "val"
            noise = bool(row)

            # load the results
            results = load_roc_results(model_id, layers, dim_reduction, aitads_a_config, noise, split, path)
            if len(results) == 0:
                continue

            # find relevant results
            relevant_results_idx = get_relevant_roc_results(results)

            tpr_all = np.array(
                [result["summary"]["macro"]["macro"]["recall"][0] for result in results]
            )
            tnr_all = np.array(
                [result["summary"]["macro"]["macro"]["tnr"][0] for result in results]
            )

            tpr_relevant = tpr_all[relevant_results_idx]
            tnr_relevant = tnr_all[relevant_results_idx]

            # sort by tnr
            sort_idx = np.lexsort((tnr_relevant, -1 * tpr_relevant))
            tnr_relevant = tnr_relevant[sort_idx]
            tpr_relevant = tpr_relevant[sort_idx]

            if verbose:
                print(
                    f"relevant results for {split} data, {'incl' if noise else 'excl'} fp alerts: {len(relevant_results_idx)}"
                )
                for i, j in enumerate(sort_idx):
                    current_tnr = tnr_all[relevant_results_idx[j]]
                    current_tpr = tpr_all[relevant_results_idx[j]]

                    # skip the relevant but not interesting results
                    if (
                        i > 0
                        and tnr_all[relevant_results_idx[sort_idx[i - 1]]] >= 0.995
                    ):
                        continue
                    if (
                        i < len(sort_idx) - 1
                        and tpr_all[relevant_results_idx[sort_idx[i + 1]]] >= 0.995
                    ):
                        continue

                    print(
                        f"  - delta = {results[relevant_results_idx[j]]['model_params']['delta']}, theta = {results[relevant_results_idx[j]]['model_params']['theta'] if model_id != 'timedelta' else None}, tnr = {current_tnr:.3f}, tpr = {current_tpr:.3f}"
                    )
                print()

            # transform to step functions
            tnr_relevant = np.array(list(pair_iterator(tnr_relevant)))[:-1]
            tpr_relevant = np.array(list(pair_iterator(tpr_relevant)))[1:]

            # plot ROC curves
            ax = axs[row, col]
            ax.set_box_aspect(1)
            ax.grid()
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_title(
                f"{split} data, {'incl' if noise else 'excl'} fp alerts, {len(results)} data points, {len(relevant_results_idx)} relevant"
            )
            if row == 1:
                ax.set_xlabel("True Negative Rate")
            if col == 0:
                ax.set_ylabel("True Positive Rate")

            # plot indiviadual results
            ax.scatter(
                [
                    tnr_all[i]
                    for i in range(len(tnr_all))
                    if i not in relevant_results_idx
                ],
                [
                    tpr_all[i]
                    for i in range(len(tpr_all))
                    if i not in relevant_results_idx
                ],
                color="b",
                marker="x",
                label="all macro results",
            )
            ax.scatter(
                [tnr_all[i] for i in relevant_results_idx],
                [tpr_all[i] for i in relevant_results_idx],
                color="r",
                marker="x",
                label="ROC relevant macro results",
            )

            # plot macro roc curve
            ax.plot(
                tnr_relevant,
                tpr_relevant,
                label=f"AUC = {compute_auc_score(tnr_relevant, tpr_relevant):.3f}, macro",
                color="r",
            )
            ax.vlines(
                tnr_relevant[-1],
                0.0,
                tpr_relevant[-1],
                ls="--",
                color="r",
                alpha=0.5,
            )
            ax.hlines(
                tpr_relevant[0],
                0.0,
                tnr_relevant[0],
                ls="--",
                color="r",
                alpha=0.5,
            )

            # individual labels
            if not plot_macro_only:
                for i, label in enumerate(lvl_1_labels):
                    relevant_results_idx = get_relevant_roc_results(results, label)

                    tpr_all = np.array(
                        [
                            result["summary"]["lvl1"][label]["recall"][0]
                            for result in results
                        ]
                    )
                    tnr_all = np.array(
                        [
                            result["summary"]["lvl1"][label]["tnr"][0]
                            for result in results
                        ]
                    )

                    tpr_relevant = tpr_all[relevant_results_idx]
                    tnr_relevant = tnr_all[relevant_results_idx]

                    # sort by tnr
                    sort_idx = np.lexsort((tnr_relevant, -1 * tpr_relevant))
                    tnr_relevant = tnr_relevant[sort_idx]
                    tpr_relevant = tpr_relevant[sort_idx]

                    tnr_relevant = np.array(list(pair_iterator(tnr_relevant)))[:-1]
                    tpr_relevant = np.array(list(pair_iterator(tpr_relevant)))[1:]

                    ax.plot(
                        tnr_relevant,
                        tpr_relevant,
                        label=f"AUC = {compute_auc_score(tnr_relevant, tpr_relevant):.3f}, {label}",
                        color=cmap(i / (len(lvl_1_labels) - 1)),
                        alpha=0.5,
                    )
                    if len(tnr_relevant) == 0:
                        continue
                    ax.hlines(
                        tpr_relevant[0],
                        0.0,
                        tnr_relevant[0],
                        ls="--",
                        color=cmap(i / (len(lvl_1_labels) - 1)),
                        alpha=0.25,
                    )
                    ax.vlines(
                        tnr_relevant[-1],
                        0.0,
                        tpr_relevant[-1],
                        ls="--",
                        color=cmap(i / (len(lvl_1_labels) - 1)),
                        alpha=0.25,
                    )

            ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

def roc_test_plot(
    model_id: str,
    aitads_a_config: Literal[
        "original",
        "simul-attacks",
        "more-noise-1",
        "more-noise-2",
        "more-noise-6",
        "more-noise-11",
    ],
    layers: tuple[str] = ("embedding", "encoder"),
    dim_reduction: int = 2,
    path: str = "saved_models",
    verbose: bool = False,
    target: str = "hierarchical_event_label",
    plot_macro_only: bool = False,
    label_vocabs: dict[str, Vocabulary] = None,
    save_mode: bool = False,
    print_auc_table: bool = True,
) -> None:
    """Plots the ROC curves for the given AlertBert or TimeDelta model and test data.
    The function produces 2 figures, one each for the results computed including/excluding the false positive alerts.

    Args:
        model_id (str): The id of the model. Can be "timedelta" or "mlm_*".
        aitads_a_config (Literal): The configuration of the AIT-ADS-A dataset.
        layers (tuple[str], optional): The layers to be used for the AlertBert models. Defaults to ("embedding", "encoder").
        dim_reduction (int, optional): The dimensionality reduction to be used for the AlertBert models. Defaults to 2.
        path (str, optional): The path to the directory where the respective model is located. Defaults to "saved_models".
        verbose (bool, optional): Whether to print additional information about the relevant results defining the ROC curve. Defaults to False.
        save_mode (bool, optional): Whether to save the figures instead of just displaying them. Defaults to False.
    """
    if not plot_macro_only:
        lvl_1_labels = get_low_level_labels(label_vocabs[target], 1)
        lvl_1_labels.remove("dns_scan")
    cmap = plt.get_cmap("viridis")

    # create the figure
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    axs = [ax1, ax2]
    figs = [fig1, fig2]
    title_str = f"ROC_{model_id}_{aitads_a_config}"
    if model_id != "timedelta":
        title_str += f"_{'input' if layers == 1 else 'output'}_emb_{dim_reduction}_dim"

    for col in range(2):
        split = "test"
        noise = bool(col)
        print(f"{'Including' if noise else 'Excluding'} false positive alerts:")

        # load the results
        results = load_roc_results(
            model_id, layers, dim_reduction, aitads_a_config, noise, split, path
        )
        if len(results) == 0:
            continue

        # find relevant results
        relevant_results_idx = get_relevant_roc_results(results)

        tpr_all = np.array(
            [result["summary"]["macro"]["macro"]["recall"][0] for result in results]
        )
        tnr_all = np.array(
            [result["summary"]["macro"]["macro"]["tnr"][0] for result in results]
        )

        tpr_relevant = tpr_all[relevant_results_idx]
        tnr_relevant = tnr_all[relevant_results_idx]

        # sort by tnr
        sort_idx = np.lexsort((tnr_relevant, -1 * tpr_relevant))
        tnr_relevant = tnr_relevant[sort_idx]
        tpr_relevant = tpr_relevant[sort_idx]

        if verbose:
            print(
                f"relevant results for {split} data, {'incl' if noise else 'excl'} fp alerts: {len(relevant_results_idx)}"
            )
            for i, j in enumerate(sort_idx):
                current_tnr = tnr_all[relevant_results_idx[j]]
                current_tpr = tpr_all[relevant_results_idx[j]]

                # skip the relevant but not interesting results
                if i > 0 and tnr_all[relevant_results_idx[sort_idx[i - 1]]] >= 0.995:
                    continue
                if (
                    i < len(sort_idx) - 1
                    and tpr_all[relevant_results_idx[sort_idx[i + 1]]] >= 0.995
                ):
                    continue

                print(
                    f"  - delta = {results[relevant_results_idx[j]]['model_params']['delta']}, theta = {results[relevant_results_idx[j]]['model_params']['theta'] if model_id != 'timedelta' else None}, tnr = {current_tnr:.3f}, tpr = {current_tpr:.3f}"
                )
            print()

        # transform to step functions
        tnr_relevant = np.array(list(pair_iterator(tnr_relevant)))[:-1]
        tpr_relevant = np.array(list(pair_iterator(tpr_relevant)))[1:]

        # plot ROC curves
        ax = axs[col]
        fig = figs[col]
        ax.set_box_aspect(1)
        ax.grid(alpha=0.5)
        ax.set_xlim(0.0, 1.005)
        ax.set_ylim(0.0, 1.005)
        save_str = f"{title_str}_{'incl' if noise else 'excl'}_noise"
        ax.set_xlabel("True Negative Rate")
        ax.set_ylabel("True Positive Rate")

        # plot macro roc curve
        ax.plot(
            tnr_relevant,
            tpr_relevant,
            label="macro",
            color="r",
            zorder=10,
        )
        ax.vlines(
            tnr_relevant[-1],
            0.0,
            tpr_relevant[-1],
            color="r",
            zorder=10,
        )
        ax.hlines(
            tpr_relevant[0],
            0.0,
            tnr_relevant[0],
            color="r",
            zorder=10,
        )
        if print_auc_table:
            print(f"model & macro & {' & '.join(lvl_1_labels)} \\\\")
            table_row = (
                f"{model_id} & {compute_auc_score(tnr_relevant, tpr_relevant):.5f} "
            )

        # individual labels
        if not plot_macro_only:
            for i, label in enumerate(lvl_1_labels):
                relevant_results_idx = get_relevant_roc_results(results, label)

                tpr_all = np.array(
                    [
                        result["summary"]["lvl1"][label]["recall"][0]
                        for result in results
                    ]
                )
                tnr_all = np.array(
                    [result["summary"]["lvl1"][label]["tnr"][0] for result in results]
                )

                tpr_relevant = tpr_all[relevant_results_idx]
                tnr_relevant = tnr_all[relevant_results_idx]

                # sort by tnr
                sort_idx = np.lexsort((tnr_relevant, -1 * tpr_relevant))
                tnr_relevant = tnr_relevant[sort_idx]
                tpr_relevant = tpr_relevant[sort_idx]

                tnr_relevant = np.array(list(pair_iterator(tnr_relevant)))[:-1]
                tpr_relevant = np.array(list(pair_iterator(tpr_relevant)))[1:]

                ax.plot(
                    tnr_relevant,
                    tpr_relevant,
                    label=label,
                    color=cmap(i / (len(lvl_1_labels) - 1)),
                )
                if len(tnr_relevant) == 0:
                    continue
                ax.hlines(
                    tpr_relevant[0],
                    0.0,
                    tnr_relevant[0],
                    color=cmap(i / (len(lvl_1_labels) - 1)),
                )
                ax.vlines(
                    tnr_relevant[-1],
                    0.0,
                    tpr_relevant[-1],
                    color=cmap(i / (len(lvl_1_labels) - 1)),
                )
                if print_auc_table:
                    table_row += (
                        f"& {compute_auc_score(tnr_relevant, tpr_relevant):.5f} "
                    )

            ax.legend(loc="lower left")

        fig.tight_layout()
        if save_mode:
            fig.savefig(f"../paper_figures/{save_str}.pdf")
        if print_auc_table:
            print(table_row + " \\\\")
            print()

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


# TODO: this function for test data
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
                        f"""{
                            train_results['summary'][level_str][label][m][0]:<5.3f
                        }±{
                            train_results['summary'][level_str][label][m][1]:<5.3f
                        } | {
                            val_results['summary'][level_str][label][m][0]:<5.3f
                        }±{
                            val_results['summary'][level_str][label][m][1]:<5.3f
                        }"""
                        for m in used_metrics
                    ]
                )
            )


# TODO: this function for test data
def pprint_eval_diff(
    train_results1: dict[str, dict | np.ndarray],
    val_results1: dict[str, dict | np.ndarray],
    train_results2: dict[str, dict | np.ndarray],
    val_results2: dict[str, dict | np.ndarray],
    target_vocab: Vocabulary,
    excluded_label: str = "-",
    exclude_raw_metrics: bool = True,
    hierarchical_label_levels: Iterable[int] = [0, 1],
    test_mode: bool = False,
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
            if not test_mode:
                print(
                    f"{label:<{label_str_len}} | "
                    + " | ".join(
                        [
                            f"""{
                                train_results2['summary'][level_str][label][m][0]
                                - train_results1['summary'][level_str][label][m][0]:< 7.3f
                            } | {
                                val_results2['summary'][level_str][label][m][0]
                                - val_results1['summary'][level_str][label][m][0]:< 7.3f
                            }"""
                            for m in used_metrics
                        ]
                    )
                )
            else:
                pass


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
    train_data: AITAlertDataset = None,
    val_data: AITAlertDataset = None,
    test_data: AITAlertDataset = None,
) -> tuple[dict, dict, dict, dict]:
    """Computes all evaluation results for the given model and data.

    Args:
        grouping_model (AbstractDatasetGroupingModel): The alert grouping model to be evaluated.
        label_vocabs (dict): The vocabularies containing the target labels.
        train_data (AITAlertDataset, optional): The training dataset to be evaluated.
        val_data (AITAlertDataset, optional): The validation dataset to be evaluated.
        test_data (AITAlertDataset, optional): The test dataset to be evaluated. Defaults to None.
    """
    results = {}
    if train_data:
        results["train"] = {}
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
        results["train"]["noise"] = train_stats_noise
        results["train"]["clean"] = train_stats_clean
    
    if val_data:
        results["val"] = {}
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
        results["val"]["noise"] = val_stats_noise
        results["val"]["clean"] = val_stats_clean
    
    if test_data:
        results["test"] = {}
        test_stats_noise, cont_matrices = eval_alert_grouping(
            model=grouping_model,
            target_vocab=label_vocabs["hierarchical_event_label"],
            data=test_data,
            ignore_excluded_macro_label=False,
        )
        test_stats_clean, _ = eval_alert_grouping(
            target_vocab=label_vocabs["hierarchical_event_label"],
            contingency_matrices=cont_matrices,
        )
        results["test"]["noise"] = test_stats_noise
        results["test"]["clean"] = test_stats_clean
    
    return results


def save_all_eval_results(
    results: dict,
    grouping_model_params: dict,
    aitads_a_config: str,
    path: str,
) -> None:
    """Saves all evaluation results to pickle files.

    Args:
        results (dict): The dict containing the noisy/clean training/val/test results.
        aitads_a_config (str): The configuration of the AIT-ADS-A dataset.
        path (str): The path to the directory where the respective model is located.
    """
    if "train" in results:
        results["train"]["noise"]["model_params"] = grouping_model_params
        results["train"]["noise"]["model_params"]["data_split"] = "train"
        save_results(
            results["train"]["noise"],
            path,
            get_eval_file_name(grouping_model_params, aitads_a_config, noise=True),
        )

        results["train"]["clean"]["model_params"] = grouping_model_params
        results["train"]["clean"]["model_params"]["data_split"] = "train"
        save_results(
            results["train"]["clean"],
            path,
            get_eval_file_name(grouping_model_params, aitads_a_config, noise=False),
        )

    if "val" in results:
        results["val"]["noise"]["model_params"] = grouping_model_params
        results["val"]["noise"]["model_params"]["data_split"] = "val"
        save_results(
            results["val"]["noise"],
            path,
            get_eval_file_name(grouping_model_params, aitads_a_config, noise=True),
        )

        results["val"]["clean"]["model_params"] = grouping_model_params
        results["val"]["clean"]["model_params"]["data_split"] = "val"
        save_results(
            results["val"]["clean"],
            path,
            get_eval_file_name(grouping_model_params, aitads_a_config, noise=False),
        )

    if "test" in results:
        results["test"]["noise"]["model_params"] = grouping_model_params
        results["test"]["noise"]["model_params"]["data_split"] = "test"
        save_results(
            results["test"]["noise"],
            path,
            get_eval_file_name(grouping_model_params, aitads_a_config, noise=True),
        )

        results["test"]["clean"]["model_params"] = grouping_model_params
        results["test"]["clean"]["model_params"]["data_split"] = "test"
        save_results(
            results["test"]["clean"],
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
    test_mode: bool = False,
) -> None:
    """Main function for evaluating alert grouping models.
    Loads the specified models and evaluates them on the training and validation (or test) sets of the specified augmentation of the AIT Alert dataset.
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
        test_mode (bool, optional): Whether to use test data instead of training/validation data. Defaults to False.
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

    write_logs = False
    log_to_stdout() if write_logs else None
    logging.info(f"Loading data config {aitads_a_config} ...") if write_logs else None
    if not test_mode:
        train_data = AITAlertDataset(split="train", configuration=aitads_a_config)
        val_data = AITAlertDataset(split="val", configuration=aitads_a_config)
        test_data = None
    else:
        train_data = None
        val_data = None
        test_data = AITAlertDataset(split="test", configuration=aitads_a_config)
    label_vocabs = load_ground_truth_label_vocabs(path, aitads_a_config)

    if timedelta:
        logging.info("Evaluating TimeDelta models...") if write_logs else None
    else:
        logging.info("Evaluating AlertBert models...") if write_logs else None
        reports, model_param_dicts = load_reports(model_ids, path)
        device = "cpu"

        logging.info("Loading data tools...") if write_logs else None
        data_tools = load_data_tools(model_ids, model_param_dicts, path, label_vocabs)

        logging.info("Loading models...") if write_logs else None
        models = load_models(model_param_dicts, path, data_tools, device)

    logging.info("Setup complete.") if write_logs else None

    for key in model_ids:
        for i in range(len(deltas)):
            if timedelta:
                grouping_model_params = get_grouping_model_params(
                    model_id="timedelta",
                    delta=deltas[i],
                )
                logging.info(
                    f"Evaluating delta = {deltas[i]} ..."
                ) if write_logs else None
                grouping_model = TimeDelta(delta=deltas[i])
            else:
                logging.info(f"Evaluating model {key} ...") if write_logs else None
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
                grouping_model, label_vocabs, train_data, val_data, test_data
            )

            # save results
            save_all_eval_results(
                results,
                grouping_model_params,
                aitads_a_config,
                path,
            )

    logging.info("Done.") if write_logs else None


if __name__ == "__main__":
    data_configs = [
        "simul-attacks",
        "more-noise-2",
        "more-noise-6",
        "more-noise-11",
        "original",
        "more-noise-1",
    ]

    test_mode = False

    def eval_run_td() -> None:
        for config in data_configs:
            compute_roc_trajectories(
                model_id="timedelta",
                aitads_a_config=config,
                deltas=timedelta_roc_traj_all,
                test_mode=test_mode,
            )
            gc.collect()

    eval_run_td()

    def model_config_generator() -> Iterable[tuple[str, str]]:
        for config in data_configs:
            for model_id in [
                "mlm_1l_4h_16d_zero_0k",
                f"mlm_1l_4h_16d_{config}_1_60k",
                f"mlm_1l_2h_16d_{config}_1_60k",
                f"mlm_1l_1h_16d_{config}_1_60k",
            ]:
                yield model_id, config

    def eval_run_ab(theta_traj: list[float], deltas: list[float]) -> None:
        Parallel(n_jobs=8)(
            delayed(compute_roc_trajectories)(
                model_id=model_id,
                aitads_a_config=config,
                deltas=deltas,
                thetas=theta_traj,
                test_mode=test_mode,
            )
            for model_id, config in model_config_generator()
        )

    # eval_run_ab(all_delta_theta_vals, all_delta_theta_vals)

    for thetas in [
        alertbert_theta_roc_traj_primary,
        alertbert_theta_roc_traj_secondary,
        alertbert_theta_roc_traj_tertiary,
        alertbert_theta_roc_traj_quartary,
    ]:
        for delta in alertbert_deltas:
            eval_run_ab(thetas, [delta])
