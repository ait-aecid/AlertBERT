import json
import logging
import os
from datetime import datetime as dt
from math import ceil, floor

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
    MaskedLangModelParams,
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
If executed as a script, it trains a masked language model on the AIT Alert dataset according to the parameters specified in a MaskedLangModelParams object.
The training routine of the script is encapsulated in the main function, which takes a Params object as input and 
serves the purpose to make the training of models possible from other scripts or modules.
Model checkpoints are saved every 10 epochs during the specified save intervals and the best performing models 
of every save interval are retained at the end of the training.
Each model is saved in a separate directory under the specified path with the following files:
- model.pt: the model state dictionary
- vocab_{feature}.json: the vocabulary for each feature in the model
- report.json: a report containing the model name, timestamp, training and validation results, and the training parameters
"""


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


def main(params: MaskedLangModelParams) -> None:
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

    if params["encoding"] == "raw_time" and params["encoding_type"] == "learned":
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
    save_intervals = params.save_intervals
    save_intervals_epochs = [
        (floor(s / len(train_loader)), ceil(e / len(train_loader)))
        for s, e in save_intervals
    ]
    epochs = save_intervals_epochs[-1][1]
    updates_per_epoch = len(train_loader)

    save_interval_index = 0
    save_interval_start, save_interval_end = save_intervals_epochs[save_interval_index]
    saving = False

    logging.info(
        f"Specified number of {params['updates']} updates amounts to {epochs} epochs with {updates_per_epoch} updates per epoch."
    )
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
                    f"Evaluation: target = {t}, train loss = {
                        tr_stats[t]['loss']:1.05f
                    }, train acc = {tr_stats[t]['corr']:1.05f}; val loss = {
                        val_stats[t]['loss']:1.05f
                    }, val acc = {val_stats[t]['corr']:1.05f}"
                )

            if saving and val_stats["total_loss"] < best_val_loss:
                # create a checkpoint for best performing models
                best_val_loss = val_stats["total_loss"]
                best_train_stats = tr_stats
                best_val_stats = val_stats
                params["updates"] = (e + 1) * updates_per_epoch
                logging.info("Creating checkpoint...")
                model_name = f"mlm_{params['id']}_{save_intervals[save_interval_index][1] // 1000}k"
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
                    "params": params.dict,
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
    main(MaskedLangModelParams("default_params"))
