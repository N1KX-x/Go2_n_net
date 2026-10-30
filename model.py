"""The n_net: a small MLP that corrects the current model's motion over one segment."""

from __future__ import annotations

import torch
from torch import nn


class NNet(nn.Module):
    """Linear -> ReLU -> Dropout blocks, then a linear output (the article's MLP, for regression).

    Takes raw features (dataset.FEATURE_NAMES) and returns the correction to the
    current model's (dx, dy, dtheta), scaled by `target_scale`. Inputs are
    standardized inside, with statistics from the training set, so a saved
    network needs nothing else to run. The output layer starts at zero: before
    training, the n_net is exactly the current model.
    """

    def __init__(self, n_inputs: int, hidden_sizes=(64, 32), dropout: float = 0.2,
                 n_outputs: int = 3):
        super().__init__()
        layers, width = [], n_inputs
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(width, n_outputs)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("input_mean", torch.zeros(n_inputs))
        self.register_buffer("input_std", torch.ones(n_inputs))
        self.register_buffer("target_scale", torch.ones(n_outputs))

    def set_normalization(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Input mean/std and target scale from the training set (a constant input keeps std 1)."""
        std = x.std(dim=0)
        self.input_mean.copy_(x.mean(dim=0))
        self.input_std.copy_(torch.where(std > 1e-8, std, torch.ones_like(std)))
        self.target_scale.copy_(y.pow(2).mean(dim=0).sqrt().clamp_min(1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scaled correction; the loss is taken on this."""
        return self.output(self.body((x - self.input_mean) / self.input_std))

    @torch.no_grad()
    def correction(self, x: torch.Tensor) -> torch.Tensor:
        """Correction in metres and radians, with dropout off."""
        training = self.training
        self.eval()
        result = self(x) * self.target_scale
        self.train(training)
        return result


def pick_device(name: str = "auto") -> torch.device:
    """cuda, then Apple's mps, then cpu, unless `name` says which."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[NNet, dict]:
    """A trained n_net (on `device`, eval mode) and everything saved with it."""
    saved = torch.load(path, map_location=device, weights_only=False)
    net = NNet(len(saved["feature_names"]), saved["hidden_sizes"], saved["dropout"])
    net.load_state_dict(saved["state_dict"])
    return net.to(device).eval(), saved
