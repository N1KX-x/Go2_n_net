#!/usr/bin/env python3
"""Predict where the robot goes: the current model plus the trained n_net's correction.

    python3 predict.py PLAN.csv [--start X Y THETA]

PLAN.csv has columns k, vx, vy, omega, dt (a plan.csv or a run's commands.csv).
Prints the pose at every segment boundary from the current model and from the
current model + n_net. From other code:

    from predict import CorrectedModel
    model = CorrectedModel()
    poses = model.rollout(commands, (x, y, theta))   # like core.nominal_rollout
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import config
import dataset
from core import Command, load_candidate_csv, motion_kind
from model import load_checkpoint


class CorrectedModel:
    """core.nominal_step with the gains and offsets the n_net was trained on, plus its correction."""

    def __init__(self, checkpoint: Path = config.CHECKPOINT, device: str = "cpu"):
        self.device = torch.device(device)
        self.net, self.saved = load_checkpoint(checkpoint, self.device)
        self.model = dataset.model_namespace(self.saved["model_params"])

    def corrections(self, x: np.ndarray) -> np.ndarray:
        """Corrections (dx, dy, dtheta) for rows of features (dataset.FEATURE_NAMES)."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        return self.net.correction(x).cpu().numpy().astype(float)

    def motion(self, command: Command, previous: Command | None) -> tuple[float, float, float]:
        """Motion over one segment, in the robot frame at its start."""
        if motion_kind(command) is None:
            raise ValueError(f"Segment {command.k}: the current model does not cover this command")
        nominal = dataset.nominal_motion(command, previous, self.model)
        correction = self.corrections([dataset.features(command, previous, nominal)])[0]
        return tuple(n + c for n, c in zip(nominal, correction))

    def step(self, pose, command: Command, previous: Command | None = None):
        return dataset.compose(pose, self.motion(command, previous))

    def rollout(self, commands, initial_pose) -> list[tuple[float, float, float]]:
        """Poses at every segment boundary, starting with `initial_pose`."""
        poses, previous = [tuple(initial_pose)], None
        for command in commands:
            poses.append(self.step(poses[-1], command, previous))
            previous = command
        return poses

    def changed_parameters(self) -> list[str]:
        """Gains and offsets that differ between the repo's config.py now and the training data."""
        now = dataset.read_repo_model()["values"]
        trained = self.saved["model_params"]["values"]
        return sorted(name for name in set(now) | set(trained) if now.get(name) != trained.get(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--start", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "THETA"))
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT)
    args = parser.parse_args()

    model = CorrectedModel(args.checkpoint)
    changed = model.changed_parameters()
    if changed:
        print(f"Warning: config.py changed since training ({', '.join(changed)}); "
              "rebuild the dataset and retrain")
    commands = load_candidate_csv(args.plan)
    nominal, previous = [tuple(args.start)], None
    for command in commands:
        nominal.append(dataset.compose(nominal[-1],
                                       dataset.nominal_motion(command, previous, model.model)))
        previous = command
    corrected = model.rollout(commands, args.start)
    print(f"{'k':>3s} {'model x':>8s} {'model y':>8s} {'model th':>9s}   "
          f"{'n_net x':>8s} {'n_net y':>8s} {'n_net th':>9s}")
    for k, (a, b) in enumerate(zip(nominal, corrected)):
        print(f"{k:3d} {a[0]:8.3f} {a[1]:8.3f} {math.degrees(a[2]):8.1f}d   "
              f"{b[0]:8.3f} {b[1]:8.3f} {math.degrees(b[2]):8.1f}d")


if __name__ == "__main__":
    main()
