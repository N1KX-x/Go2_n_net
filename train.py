#!/usr/bin/env python3
"""Train the n_net on data/segments.csv (run dataset.py first).

    python3 train.py

Keeps the network with the lowest validation loss and saves it, with its
normalization, the current model's gains and offsets and the split, to
checkpoints/n_net.pt. Also draws results/loss_curve.png.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
import dataset
from model import NNet, pick_device


class Learner:
    """Training loop after the article's Learner: train_epoch, validate_epoch, fit."""

    def __init__(self, net: NNet, train_loader: DataLoader, valid_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, device: torch.device):
        self.net, self.optimizer, self.device = net, optimizer, device
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.loss_fn = nn.MSELoss()

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x.to(self.device), y.to(self.device)
        return self.loss_fn(self.net(x), y / self.net.target_scale)

    def train_epoch(self) -> float:
        self.net.train()
        total, count = 0.0, 0
        for x, y in self.train_loader:
            loss = self._loss(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        return total / count

    @torch.no_grad()
    def validate_epoch(self) -> float:
        self.net.eval()
        total, count = 0.0, 0
        for x, y in self.valid_loader:
            total += self._loss(x, y).item() * len(x)
            count += len(x)
        return total / count

    def fit(self, max_epochs: int, patience: int, verbose: bool = True) -> dict:
        """Train until the validation loss stops improving; keep the best weights."""
        say = print if verbose else (lambda *args: None)
        history = {"train": [], "valid": []}
        best_loss, best_epoch, best_state = self.validate_epoch(), 0, self._state()
        say(f"epoch   0  valid {best_loss:.4f}  (current model alone)")
        for epoch in range(1, max_epochs + 1):
            history["train"].append(self.train_epoch())
            history["valid"].append(self.validate_epoch())
            if history["valid"][-1] < best_loss:
                best_loss, best_epoch, best_state = history["valid"][-1], epoch, self._state()
            if epoch % 20 == 0 or epoch == 1:
                say(f"epoch {epoch:3d}  train {history['train'][-1]:.4f}  "
                    f"valid {history['valid'][-1]:.4f}  best {best_loss:.4f} @ {best_epoch}")
            if epoch - best_epoch >= patience:
                say(f"stopped at epoch {epoch}: no better validation loss for {patience} epochs")
                break
        self.net.load_state_dict(best_state)
        return {"history": history, "best_epoch": best_epoch, "best_valid_loss": best_loss}

    def _state(self) -> dict:
        return {name: value.detach().clone() for name, value in self.net.state_dict().items()}


def loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=config.BATCH_SIZE, shuffle=shuffle)


def plot_history(history: dict, best_epoch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train"], label="train")
    ax.plot(epochs, history["valid"], label="validation")
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"kept (epoch {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (scaled correction)")
    ax.set_title("n_net training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_ROOT / "loss_curve.png", dpi=150)
    plt.close(fig)


def train_network(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
                  y_valid: np.ndarray, device: torch.device, seed: int = config.TRAIN_SEED,
                  verbose: bool = True) -> tuple[NNet, dict]:
    """A new n_net trained with the settings in config.py, and its training history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = NNet(x_train.shape[1], config.HIDDEN_SIZES, config.DROPOUT)
    net.set_normalization(torch.from_numpy(x_train), torch.from_numpy(y_train))
    net.to(device)
    if config.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=config.LEARNING_RATE,
                                    weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)
    learner = Learner(net, loader(x_train, y_train, True), loader(x_valid, y_valid, False),
                      optimizer, device)
    return net, learner.fit(config.MAX_EPOCHS, config.EARLY_STOP_PATIENCE, verbose)


def main() -> None:
    rows = dataset.load_segments()
    split = json.loads(config.SPLIT_JSON.read_text(encoding="utf-8"))
    params = json.loads(config.MODEL_PARAMS_JSON.read_text(encoding="utf-8"))
    x_train, y_train = dataset.arrays(dataset.rows_in(rows, split["train"]))
    x_valid, y_valid = dataset.arrays(dataset.rows_in(rows, split["valid"]))
    device = pick_device(config.DEVICE)
    print(f"{len(x_train)} train / {len(x_valid)} validation segments, device {device}")
    started = time.time()
    net, result = train_network(x_train, y_train, x_valid, y_valid, device)
    print(f"kept epoch {result['best_epoch']}, validation loss {result['best_valid_loss']:.4f} "
          f"({time.time() - started:.0f} s)")

    config.CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {name: value.cpu() for name, value in net.state_dict().items()},
        "feature_names": dataset.FEATURE_NAMES,
        "target_names": dataset.TARGET_NAMES,
        "hidden_sizes": tuple(config.HIDDEN_SIZES),
        "dropout": config.DROPOUT,
        "model_params": params,
        "split": split,
        "settings": {name: getattr(config, name) for name in (
            "OPTIMIZER", "LEARNING_RATE", "WEIGHT_DECAY", "BATCH_SIZE", "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE", "TRAIN_SEED", "ONLY_PLANNER_LIMITS")},
        **result,
    }, config.CHECKPOINT)
    plot_history(result["history"], result["best_epoch"])
    print(f"Saved {config.CHECKPOINT} and {config.RESULTS_ROOT / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
