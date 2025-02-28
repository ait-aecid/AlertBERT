import json
import logging

import torch
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from alertbert.aitads import AITAlertDataset, AlertSequenceBatchSampler
from alertbert.models import (
    MaskedLangModelEvalWrapper,
    MaskedLanguageModel,
)
from alertbert.preprocessing import (
    MaskedLangModelingSequenceCollate,
    default_collate_fn,
    load_feature_vocabs,
)
from alertbert.utils import get_device, log_to_stdout

"""This module contains functions for evaluating masked language models.
If executed as a script, it will load a trained model and evaluate it on the training and validation sets of the AIT Alert dataset.
"""


def top_k_accuracy(ranks: torch.IntTensor, k: int) -> float:
    """Calculates the top-k accuracy.

    Parameters:
    - ranks (torch.IntTensor): A tensor containing the ranks of the predictions.
    - k (int): The value of k for top-k accuracy.

    Returns:
    - float: The top-k accuracy.

    """
    return (ranks <= k).mean(dtype=torch.float64).item()


def classification_report(stats: dict[str, torch.tensor]) -> dict[str, float]:
    """Computes various classification metrics from statistics produced by eval_masked_lang_model.

    Parameters:
    - stats (dict[str, torch.tensor]): The statistics produced by eval_masked_lang_model.

    Returns:
    - dict[str, float]: A dictionary containing the computed metrics.

    """
    y_true = stats["true"]
    y_pred = stats["pred"]
    rank = stats["rank"]
    return {
        "loss": stats["loss"],
        "accuracy": stats["corr"],
        "top_2_accuracy": top_k_accuracy(rank, 2),
        "top_3_accuracy": top_k_accuracy(rank, 3),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0.0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0.0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0.0),
    }


def eval_masked_lang_model(
    model: MaskedLangModelEvalWrapper,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
) -> dict[str, float | dict[str, torch.tensor]]:
    """Computes evaluation statistics for a masked language model.

    Parameters:
    - model (MaskedLangModelEvalWrapper): The model to evaluate.
    - loader (DataLoader): The data loader.
    - device (torch.device): The device to use for computation.
    - epochs (int): The number of epochs to evaluate.
        As the targets for prediction are randomly chosen every epoch a larger number of epochs is recommended for consistent evaluation.

    Returns:
    - dict[str, float | dict[str, torch.tensor]]: A dictionary containing the evaluation statistics.

    """
    stats = {  # dictionary to store statistics for each target of the model
        t: {
            "loss": [],  # average loss
            "corr": [],  # accuracy
            "rank": [],  # rank of each true label in the prediction
            "pred": [],  # predicted labels
            "true": [],  # true labels
            "size": [],  # number of samples
        }
        for t in model.module["head"].targets
    }
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    with torch.no_grad():
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                batch = model(batch)
                for t in model.module["head"].targets:
                    true = batch[f"{t}_true"]
                    stats[t]["true"].append(true.cpu())
                    stats[t]["size"].append(len(true))
                    out = batch[f"{t}_pred_mask"]
                    loss_value = loss_fn(out, true)
                    stats[t]["loss"].append(loss_value.item())
                    pred = out.argmax(dim=1)
                    stats[t]["pred"].append(pred.cpu())
                    corr = (pred == true).sum()
                    stats[t]["corr"].append(corr.item())
                    rank = torch.sum(
                        out.t() >= out[(torch.arange(len(true)), true)], dim=0
                    )
                    stats[t]["rank"].append(rank.cpu())
    for t in model.module["head"].targets:
        # aggregate statistics
        stats[t]["size"] = sum(stats[t]["size"])
        stats[t]["loss"] = sum(stats[t]["loss"]) / stats[t]["size"]
        stats[t]["corr"] = sum(stats[t]["corr"]) / stats[t]["size"]
        stats[t]["rank"] = torch.cat(stats[t]["rank"])
        stats[t]["pred"] = torch.cat(stats[t]["pred"])
        stats[t]["true"] = torch.cat(stats[t]["true"])
    # add average loss across all targets
    stats["total_loss"] = sum(
        stats[t]["loss"] for t in model.module["head"].targets
    ) / len(model.module["head"].targets)
    return stats


if __name__ == "__main__":
    model_id = "mlm_1l_4h_base_3-1_60k"
    path = "saved_models"
    with open(path + f"/{model_id}/report.json") as f:
        report = json.load(f)
    params = report["params"]

    log_to_stdout()

    device = get_device()

    logging.info("Loading data...")
    train_data = AITAlertDataset(split="train", configuration=params["augment"])
    val_data = AITAlertDataset(split="val", configuration=params["augment"])

    collate_function_map = load_feature_vocabs(
        path=f"{path}/{model_id}",
        features=set(params["features"]) | set(params["targets"]),
        min_freq=params["min_freq"],
    )

    collate_function_map[params["encoding"]] = default_collate_fn

    collate_function = MaskedLangModelingSequenceCollate(
        collate_function_map,
        params["target_ratio"],
        params["mask_ratio"],
        params["perturb_ratio"],
    )

    train_sampler = AlertSequenceBatchSampler(
        train_data,
        context_size=params["context_size"],
        batch_size=16,
        drop_last=False,
        shuffle=False,
    )
    val_sampler = AlertSequenceBatchSampler(
        val_data,
        context_size=params["context_size"],
        batch_size=16,
        drop_last=False,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        collate_fn=collate_function,
    )
    val_loader = DataLoader(
        val_data,
        batch_sampler=val_sampler,
        collate_fn=collate_function,
    )

    logging.info("Building model...")
    model = MaskedLanguageModel(params=params, vocabs=collate_function_map)
    model.load_state_dict(
        torch.load(path + f"/{model_id}/model.pt", weights_only=True), strict=False
    )
    model.to(device)
    model.eval()
    model_ev = MaskedLangModelEvalWrapper(model)

    logging.info("Evaluating...")
    train_stats = eval_masked_lang_model(model_ev, train_loader, device, epochs=10)
    val_stats = eval_masked_lang_model(model_ev, val_loader, device, epochs=10)

    report = {
        "training": {
            t: classification_report(train_stats[t]) for t in params["targets"]
        },
        "validation": {
            t: classification_report(val_stats[t]) for t in params["targets"]
        },
    }
    print(report)
    print()

    results = f"Model: mlm_{params['id']} Results: train loss = {train_stats['total_loss']:1.05f}, val loss = {val_stats['total_loss']:1.05f}"
    for t in params["targets"]:
        results += f", {t: >5} train acc = {train_stats[t]['corr']:1.05f}, {t: >5} val acc = {val_stats[t]['corr']:1.05f}"
        results += f", {t: >5} train f1 = {report['training'][t]['macro_f1']:1.05f}, {t: >5} val f1 = {report['validation'][t]['macro_f1']:1.05f}"
    results += f"; params = {params}"
    print(results)
