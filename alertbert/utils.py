import logging
import sys

import torch

"""This module contains utility functions for deep learning tasks."""


def get_device(d: str = None) -> torch.device:
    """Get the torch.device to be used for computation.

    Args:
        d (str, optional): The device to be used. If None, the device is
            selected based on availability. Defaults to None.

    Returns:
        torch.device: The device to be used for computation.
    """
    if d is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    elif d == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    elif d == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif d == "cpu":
        dev = torch.device("cpu")
    else:
        raise ValueError(f"device not available: {d}")
    logging.info(f"Using device: {dev}")
    return dev


def set_up_log(filename: str) -> None:
    """Set up a log file.

    Args:
        filename (str): The name of the log file.

    """
    # check if logging is already configured
    if logging.getLogger().hasHandlers():
        print(f"Logging already configured to {logging.getLogger().handlers[0].baseFilename}")
        return
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
        filename=filename + ".log",
    )
    logging.info(f"Logging to {filename}.log")


def log_to_stdout() -> None:
    """Configures logging to stdout."""
    # check if logging is already configured
    if logging.getLogger().hasHandlers():
        print(f"Logging already configured to {logging.getLogger().handlers[0].baseFilename}")
        return
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    logging.info("Logging to stdout")

def count_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a Pytorch Module.
    Important: Does not account for tied weights!"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class OptimWrapper:
    """Wrapper to unify the interfaces of Pytorch and schedulefree optimizers."""
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer
    
    def step(self) -> None:
        self.optimizer.step()
        self.scheduler.step()
    
    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
    
    def train(self) -> None:
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
    
    def eval(self) -> None:
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

def print_to_log(message: str) -> None:
    """Prints a message to the log file."""
    with open("saved_models/results.log", "a") as log_file:
        print(f"\n# {message}", file=log_file)
