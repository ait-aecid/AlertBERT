import json
import logging
import os
from datetime import datetime as dt
from math import ceil, floor
from typing import Any

import torch
from torch.utils.data import DataLoader

from alertbert.aitads import (
    AITAlertDataset,
    AlertSequenceBatchSampler,
    aitads_train_external_mail_hosts,
)
from alertbert.eval_mlm import classification_report, eval_masked_lang_model
from alertbert.models import (
    MaskedLangModelEvalWrapper,
    MaskedLangModelTrainWrapper,
    MaskedLanguageModel,
    MultiTargetLoss,
)
from alertbert.preprocessing import (
    MaskedLangModelingSequenceCollate,
    build_feature_vocabs,
    default_collate_fn,
)
from alertbert.utils import OptimWrapper, get_device, log_to_stdout, set_up_log

"""This module contains functions for training masked language models.
If executed as a script, it trains a masked language model on the AIT Alert dataset according to the parameters specified after this comment.
The training routine of the script is encapsulated in the main function, which takes a dictionary of parameters as input and serves the purpose to make the training of models possible from other scripts or modules.
Model checkpoints are saved every 10 epochs during the specified save intervals, and the best performing models of every save interval are retained at the end of the training.
Each model is saved in a separate directory under the specified path with the following files:
- model.pt: the model state dictionary
- vocab_{feature}.json: the vocabulary for each feature in the model
- report.json: a report containing the model name, timestamp, training and validation results, and the training parameters
"""

########################################################################################

"""The following variables and the params dictionary define the parameters for training a masked language model on the AIT Alert dataset.

The meaning of each variable defined below is as follows:
    - context_size (int): The size of the context window.
    - layers (int): The number of layers in the transformer encoder.
    - heads (int): The number of attention heads in each layer of the transformer encoder.
    - dim_per_head (int): The dimension of each attention head in the transformer encoder.
    - ffw_factor (int): The factor by which to multiply the dimension of the model to get the dimension of the feedforward layer in the transformer encoder.
    - encoding_type (str): The type of encoding to use for the transformer encoder: "learned" for learned positional encoding or "rotary" for rotary encoding.
    - rotary_max_exp (int): The maximum exponent of the frequencies to use for the rotary encoding. For positional encoding this should be log2(context_size), 
        for time encoding it should be log2 of the maximal reasonable timespan in a context window, e.g. 14 for overnight context windows.
    - rotary_cutoff (float): The cutoff ratio for the frequencies to use for the rotary encoding.
    - gamma (float | None): The gamma parameter for the focal loss function, if used. If None, the cross-entropy loss function is used.
    - lo (str): Indicator of the loss function used: "ce" for cross-entropy or "fo" for focal loss.
    - save_intervals (list[tuple[int, int]]): A list of tuples specifying the number of model updates at which to start and end each save interval.

The params dictionary contains the following parameters:
    # data params
    - augment (Literal | None): The configuration of AIT-ADS-A to use. If None, the original AIT Alert dataset is used.
    - context_size (int): The size of the context window.
    - batch_size (int): The batch size for training.
    - features (list[Literal["short", "host"]]): The list of features to include in the model.
    - targets (list[Literal["short", "host"]]): The list of target variables to predict.
    - sampling (Literal["index", "time"]): The sampling method to use for creating batches. The option "time" is deprecated.
    - min_freq (int): The minimum frequency of a token to be included in the vocabulary.
    # model params
    - d_model (int): The dimension of the model.
    - nhead (int): The number of attention heads in each layer of the model.
    - num_layers (int): The number of layers in the model.
    - dim_feedforward (int): The dimension of the feedforward layer in the model.
    - activation (Literal["relu", "gelu"]): The activation function to use in the feedforward layer.
    - gated_activation (bool): Whether to use a gated activation function in the feedforward layer.
    - encoding (Literal["position", "raw_time"]): The encoding method to use.
    - enc_type (Literal["learned", "rotary"]): The type of encoding to use. The option "learned" is only available for the "position" encoding. Sinusoidal encoding is currently not implemented.
    - encoding_freqs (list[float] | None): The frequencies to use for the rotary/sinusoidal encoding. The length of the list must be at most d_model//nheads//2, the remaining frequencies are set to 0.
    - biases (bool): Whether to include biases in the model.
    - head_bias (bool): Whether to use biases in the prediction head.
    - tie_weights (bool): Whether to tie the weights of the input and output embeddings.
    - emb_init_std (float): The standard deviation of the normal distribution to use for initializing the embeddings.
    # training params
    - updates (int): The number of updates for which to train the model. In the saved parameters, this is updated to the number at which the best model was saved.
    - optimizer (Literal["sgd", "adam"]): The optimizer to use for training.
    - scheduler (Literal["schedulefree", "linear"]): The scheduler to use for training.
    - lr (float): The learning rate for the optimizer.
    - warm_up_steps (int | float): The number of warm-up steps for the learning rate scheduler.
    - decay (float): The weight decay for the optimizer.
    - momentum (float): The momentum for the optimizer.
    - gamma (float | None): The gamma parameter for the focal loss function, if used. If None, the cross-entropy loss function is used.
    - class_balance (float): Inverse softmax temperature to be applied to class frequencies to obtain class balancing weights for the loss function.
        If 0, no class balancing is applied, positive values emphasize underrepresented classes, and negative values emphasize overrepresented classes.
    - target_ratio (float): The ratio of target tokens to mask in the input sequence.
    - mask_ratio (float): The ratio of target tokens to replace with the mask token in the input sequence.
    - perturb_ratio (float): The ratio of target tokens to replace with a random token in the input sequence.
    # file params
    - path (str): The path to save the model checkpoints.
    - log (str): The name of the log file to use.
    - id (str): The identifier for the model.
"""

context_size = 2**12
layers = 1
heads = 4
dim_per_head = 16
ffw_factor = 4
encoding_type = "rotary"
rotary_max_exp = 14
rotary_cutoff = 0.75
gamma = None
lo = "fo" if gamma else "ce"
save_intervals = [(18000, 20000), (38000, 40000), (58000, 60000)]

params = {
    # data
    "augment": "original",
    "context_size": context_size,
    "batch_size": 16,
    "features": ["short", "host"],
    "targets": ["short", "host"],
    "sampling": "index",
    "min_freq": 10,
    # model
    "d_model": heads * dim_per_head,
    "nhead": heads,
    "num_layers": layers,
    "dim_feedforward": heads * dim_per_head * ffw_factor,
    "activation": "gelu",
    "gated_activation": True,
    "encoding": "raw_time",
    "enc_type": encoding_type,
    "encoding_freqs": [
        2 ** (-i * 2.0 / dim_per_head * rotary_max_exp) for i in range(int(dim_per_head // 2 * rotary_cutoff))
    ]
    if encoding_type == "rotary"
    else None,
    "biases": False,
    "head_bias": False,
    "tie_weights": True,
    "emb_init_std": (heads * dim_per_head) ** -0.5,
    # training
    "updates": save_intervals[-1][1],
    "optimizer": "adam",
    "scheduler": "linear",
    "lr": 5e-3,
    "warm_up_steps": 200,
    "decay": 0.1,
    "momentum": 0.9,
    "gamma": gamma,
    "class_balance": 2.0,
    "target_ratio": 0.2,
    "mask_ratio": 0.8,
    "perturb_ratio": 0.1,
    # files
    "path": "saved_models",
    "log": "train",
    "id": f"{layers}l_{heads}h_nano",
}

########################################################################################


def train_model(
    model: MaskedLangModelTrainWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Trains the given model on the given data loader using the given optimizer.

    Parameters:
    - model (MaskedLangModelTrainWrapper): The model to train.
    - loader (DataLoader): The data loader to use for training.
    - optimizer (torch.optim.optimizer.Optimizer): The optimizer to use for training.
    - device (torch.device): The device on which to perform the training.

    Returns:
    - tuple[float, float]: A tuple containing the mean and standard deviation of the losses obtained during training

    """
    model.train()
    losses = torch.empty(len(loader))
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        batch = model(batch)
        optimizer.zero_grad()
        batch["loss"].backward()
        optimizer.step()
        losses[i] = batch["loss"].item()
    return losses.mean(), losses.std()


def main(params: dict[str, Any], save_intervals: list[tuple[int, int]]) -> None:
    """This function encapsulates the training routine for a masked language model on the AIT Alert dataset.
    For more information plaese refer to the module docstring.
    """

    # set up environment

    if params["log"]:
        set_up_log(f"{params['path']}/{params['log']}")
    else:
        log_to_stdout()

    logging.info("Run id: " + params["id"])

    device = get_device()

    # load data

    logging.info("Loading data...")
    if params["augment"]:
        train_data = AITAlertDataset(split="train", configuration=params["augment"])
        val_data = AITAlertDataset(split="val", configuration=params["augment"])
    else:
        train_data = AITAlertDataset(split="train", flavour="original")
        val_data = AITAlertDataset(split="val", flavour="original")

    collate_function_map = build_feature_vocabs(
        dataset=train_data,
        features=set(params["features"]) | set(params["targets"]),
        min_freq=params["min_freq"],
    )
    if "host" in collate_function_map:
        collate_function_map["host"].remove(aitads_train_external_mail_hosts)

    if params["encoding"] == "raw_time" and params["enc_type"] == "learned":
        raise ValueError("Time encoding is not available for learned encoding.")
    else:
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
        batch_size=params["batch_size"],
        sampling_method=params["sampling"],
    )
    val_sampler = AlertSequenceBatchSampler(
        val_data,
        context_size=params["context_size"],
        batch_size=params["batch_size"],
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

    # build model

    logging.info("Building model...")
    model = MaskedLanguageModel(params=params, vocabs=collate_function_map)
    model.to(device)

    # calculate epochs from updates
    save_intervals_epochs = [(floor(s / len(train_loader)), ceil(e / len(train_loader))) for s, e in save_intervals]
    epochs = save_intervals_epochs[-1][1]
    updates_per_epoch = len(train_loader)

    save_interval_index = 0
    save_interval_start, save_interval_end = save_intervals_epochs[save_interval_index]
    saving = False

    logging.info(f"Specified number of {params['updates']} updates amounts to {epochs} epochs with {updates_per_epoch} updates per epoch.")
    logging.info(f"Save intervals in epochs: {save_intervals_epochs}")


    if params["scheduler"] == "schedulefree":
        if params["optimizer"] == "adam":
            from schedulefree import AdamWScheduleFree

            optimizer = AdamWScheduleFree(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                betas=(params["momentum"], 0.999),
                warmup_steps=params["warm_up_steps"],
            )
        elif params["optimizer"] == "sgd":
            from schedulefree import SGDScheduleFree

            optimizer = SGDScheduleFree(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                momentum=params["momentum"],
                warmup_steps=params["warm_up_steps"],
            )
    elif params["scheduler"] == "linear":
        if params["optimizer"] == "adam":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                betas=(params["momentum"], 0.999),
            )
        elif params["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
                momentum=params["momentum"],
            )
        optimizer = OptimWrapper(
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params["lr"],
                total_steps=epochs * updates_per_epoch,
                pct_start=float(params["warm_up_steps"]) / (epochs * updates_per_epoch),
                anneal_strategy="linear",
                cycle_momentum=False,
                div_factor=1e3,
                final_div_factor=1e4,
            )
        )

    class_weights = {
        t: torch.softmax(
            collate_function_map[t].get_frequencies().to(device)
            * params["class_balance"]
            * -1.0,
            dim=0,
        )
        for t in params["targets"]
    }

    if params["gamma"] is not None:
        # from kornia.losses import FocalLoss
        raise NotImplementedError(
            "Focal loss is currently disabled due to a warning from the kornia package."
        )
        # if fixed remove the comment below

    loss_fn = MultiTargetLoss(
        [
            (
                torch.nn.CrossEntropyLoss(weight=class_weights[t])
                if True  # params["gamma"] is None
                else FocalLoss(  # noqa: F821
                    alpha=None,
                    gamma=params["gamma"],
                    reduction="mean",
                    weight=class_weights[t],
                )
            )
            for t in params["targets"]
        ]
    )

    model_tr = MaskedLangModelTrainWrapper(model, loss_fn)
    model_ev = MaskedLangModelEvalWrapper(model)

    # train model

    logging.info("Training...")

    for e in range(epochs):
        model.train()
        optimizer.train()  # necessary for schedulefree optimizer, ignored via OptimWrapper for other optimizers

        mean, std = train_model(model_tr, train_loader, optimizer, device)
        if (e < 20) or (e % 10 == 9):
            logging.info(f"Epoch {e + 1:4d}: train loss = {mean:1.05f} ± {std:1.05f}")

        if e + 1 == save_interval_start:
            logging.info("Beginning of save interval.")
            saving = True
            best_val_loss = float("inf")
            best_train_stats = None
            best_val_stats = None

        if ((e % 10 == 9) and saving) or (e % 100 == 99):
            # model evaluation every 100 epochs and every 10 epochs during save intervals
            optimizer.eval()  # necessary for schedulefree optimizer, ignored via OptimWrapper for other optimizers

            if params["scheduler"] == "schedulefree":
                # flush optimizer momentum (see https://github.com/facebookresearch/schedule_free/blob/main/README.md#caveats)
                with torch.no_grad():
                    for batch in train_loader:
                        batch = batch.to(device)
                        model_tr(batch)

            model.eval()

            tr_stats = eval_masked_lang_model(model_ev, train_loader, device, epochs=3)
            val_stats = eval_masked_lang_model(model_ev, val_loader, device, epochs=5)
            for t in params["targets"]:
                logging.info(
                    f"Evaluation: target = {t}, train loss = {tr_stats[t]['loss']:1.05f}, train acc = {tr_stats[t]['corr']:1.05f}; val loss = {val_stats[t]['loss']:1.05f}, val acc = {val_stats[t]['corr']:1.05f}"
                )

            if saving and val_stats["total_loss"] < best_val_loss:
                # create a checkpoint for best performing models
                best_val_loss = val_stats["total_loss"]
                best_train_stats = tr_stats
                best_val_stats = val_stats
                params["updates"] = (e + 1) * updates_per_epoch
                logging.info("Creating checkpoint...")
                model_name = f"mlm_{params['id']}_{save_intervals[save_interval_index][1]//1000}k"
                save_location = f"{params['path']}/{model_name}"
                os.makedirs(save_location, exist_ok=True)
                torch.save(model.state_dict(), save_location + "/model.pt")
                for f in set(params["features"]) | set(params["targets"]):
                    collate_function_map[f].save(save_location + f"/vocab_{f}.json")
                report = {
                    "model": model_name,
                    "timestamp": str(dt.now()),
                    "epochs": e + 1,
                    "training": {
                        t: classification_report(tr_stats[t]) for t in params["targets"]
                    },
                    "validation": {
                        t: classification_report(val_stats[t])
                        for t in params["targets"]
                    },
                    "params": params,
                }
                with open(save_location + "/report.json", "w") as f:
                    json.dump(report, f, indent=4)

        if e + 1 == save_interval_end:
            logging.info("End of save interval.")
            saving = False
            save_interval_index += 1
            if save_interval_index < len(save_intervals_epochs):
                save_interval_start, save_interval_end = save_intervals_epochs[
                    save_interval_index
                ]

            # log results to results.log
            results = f"{str(dt.now())} Model: {model_name} Results: train loss = {best_train_stats['total_loss']:1.05f}, val loss = {best_val_stats['total_loss']:1.05f}"
            for t in params["targets"]:
                results += f", {t: >5} train acc = {best_train_stats[t]['corr']:1.05f}, {t: >5} val acc = {best_val_stats[t]['corr']:1.05f}"
                results += f", {t: >5} train f1 = {report['training'][t]['macro_f1']:1.05f}, {t: >5} val f1 = {report['validation'][t]['macro_f1']:1.05f}"
            results += f"; params = {params}"
            with open(params["path"] + "/results.log", "a") as f:
                print(results, file=f)

    logging.info("Done.")


if __name__ == "__main__":
    main(params, save_intervals)
