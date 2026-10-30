#!/usr/bin/env python3
"""Compare the current model, current model + linear correction, and current model + n_net.

    python3 evaluate.py

Per segment: RMS position and heading error from the true start pose of each
segment. Whole runs (test runs whose every segment is usable): roll each model
forward from the run's start and report E (largest boundary distance) as in
phase0analyze. The 2026-10-21 pilots fitted the current gains, so on those the
current model is in-sample; the other sessions are new to every model.

Writes results/evaluation.txt, results/test_runs.csv and results/test_rollouts.png.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np

import config
import dataset
from predict import CorrectedModel


class Ridge:
    """Linear correction on the same inputs, as a baseline the n_net has to beat."""

    def __init__(self, x: np.ndarray, y: np.ndarray, lam: float):
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        a = self._design(x)
        penalty = lam * np.eye(a.shape[1])
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(a.T @ a + penalty, a.T @ y)

    def _design(self, x: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((len(x), 1)), (x - self.mean) / self.std])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._design(x) @ self.coef


def segment_errors(rows: list[dict], methods: dict) -> dict[str, tuple[float, float]]:
    """RMS position error (cm) and heading error (deg) per method over used segments."""
    x, y = dataset.arrays(rows)
    result = {}
    for name, correct in methods.items():
        error = y - correct(x)
        result[name] = (100 * math.sqrt(np.mean(error[:, 0]**2 + error[:, 1]**2)),
                        math.degrees(math.sqrt(np.mean(error[:, 2]**2))))
    return result


def rollout_errors(segments: list[dict], correction: np.ndarray) -> list[float]:
    """Boundary distances of one run rolled forward from its true start pose."""
    pose = (segments[0]["start_x"], segments[0]["start_y"], segments[0]["start_theta"])
    errors = []
    for row, c in zip(segments, correction):
        nominal = (row["nominal_dx"], row["nominal_dy"], row["nominal_dtheta"])
        pose = dataset.compose(pose, [n + d for n, d in zip(nominal, c)])
        errors.append(math.hypot(pose[0] - row["end_x"], pose[1] - row["end_y"]))
    return errors


def plot_rollouts(runs: dict, methods: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(runs)
    columns = 4
    lines = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(lines, columns, figsize=(4*columns, 3.6*lines), squeeze=False)
    for ax, name in zip(axes.flat, names):
        segments = runs[name]
        x, _ = dataset.arrays(segments)
        ax.plot([segments[0]["start_x"]] + [r["end_x"] for r in segments],
                [segments[0]["start_y"]] + [r["end_y"] for r in segments],
                "k.-", label="mocap", linewidth=1.5)
        for label, correct in methods.items():
            pose = (segments[0]["start_x"], segments[0]["start_y"], segments[0]["start_theta"])
            path = [pose]
            for row, c in zip(segments, correct(x)):
                nominal = (row["nominal_dx"], row["nominal_dy"], row["nominal_dtheta"])
                path.append(dataset.compose(path[-1], [n + d for n, d in zip(nominal, c)]))
            ax.plot([p[0] for p in path], [p[1] for p in path], ".-", linewidth=1, label=label)
        ax.set_title(name, fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.RESULTS_ROOT / "test_rollouts.png", dpi=120)
    plt.close(fig)


def main() -> None:
    model = CorrectedModel()
    saved = model.saved
    rows = dataset.load_segments()
    built_with = json.loads(config.MODEL_PARAMS_JSON.read_text(encoding="utf-8"))
    if built_with != saved["model_params"]:
        raise SystemExit("data/segments.csv was rebuilt with other gains than the n_net was "
                         "trained on; run train.py again")
    split = saved["split"]
    parts = {part: dataset.rows_in(rows, runs) for part, runs in split.items()}
    ridge = Ridge(*dataset.arrays(parts["train"]), config.RIDGE_LAMBDA)
    methods = {"current model": lambda x: np.zeros((len(x), 3)),
               "+ linear": ridge,
               "+ n_net": model.corrections}

    lines = [f"n_net {config.CHECKPOINT.name}: epoch {saved['best_epoch']}, "
             f"gains from repo commit {saved['model_params']['repo_commit']}"]
    changed = model.changed_parameters()
    if changed:
        lines.append(f"Note: the repo's config.py has changed since ({', '.join(changed)})")
    lines += ["", "Per segment, RMS error from the true start pose: position cm / heading deg",
              f"{'':34s}" + "".join(f"{name:>20s}" for name in methods)]

    def add(label: str, part_rows: list[dict]) -> None:
        if not any(row["use"] for row in part_rows):
            return
        n = sum(int(row["use"]) for row in part_rows)
        errors = segment_errors(part_rows, methods)
        lines.append(f"{label:27s} {n:5d}  " + "".join(
            f"{errors[name][0]:12.1f} / {errors[name][1]:4.1f}" for name in methods))

    for part in ("train", "valid", "test"):
        add(part, parts[part])
    lines.append("test by session:")
    for session in sorted({row["session"] for row in parts["test"]}):
        add(f"  {session}", [row for row in parts["test"] if row["session"] == session])
    if not config.ONLY_PLANNER_LIMITS:
        add("test, planner limits only", [row for row in parts["test"] if row["in_limits"]])

    by_run = defaultdict(list)
    for row in parts["test"]:
        by_run[dataset.run_id(row)].append(row)
    complete = {name: sorted(segments, key=lambda r: r["k"])
                for name, segments in sorted(by_run.items())
                if all(row["use"] for row in segments)}
    table = []
    for name, segments in complete.items():
        x, _ = dataset.arrays(segments)
        record = {"run": name, "segments": len(segments)}
        for label, correct in methods.items():
            errors = rollout_errors(segments, correct(x))
            record[f"E {label}"] = max(errors)
            record[f"sq {label}"] = sum(e*e for e in errors)
        table.append(record)

    lines += ["", f"Whole test runs rolled forward ({len(complete)} of {len(by_run)} test runs are "
              "complete): mean E / q90 E / RMS of all boundaries, metres",
              f"{'':34s}" + "".join(f"{name:>20s}" for name in methods)]
    sessions = sorted({name.split("/")[0] for name in complete})
    for session in sessions + ["all"]:
        mine = [r for r in table if session == "all" or r["run"].startswith(session + "/")]
        count = sum(r["segments"] for r in mine)
        cells = []
        for label in methods:
            values = [r[f"E {label}"] for r in mine]
            rms = math.sqrt(sum(r[f"sq {label}"] for r in mine) / count)
            cells.append(f"{np.mean(values):6.2f} /{np.quantile(values, 0.9):5.2f} /{rms:5.2f}")
        lines.append(f"{session:27s} {len(mine):3d} runs" + "".join(f"{c:>20s}" for c in cells))

    config.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    (config.RESULTS_ROOT / "evaluation.txt").write_text(report, encoding="utf-8")
    with (config.RESULTS_ROOT / "test_runs.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["run", "segments"] + [f"E_m {label}" for label in methods])
        for r in table:
            writer.writerow([r["run"], r["segments"]] + [f"{r[f'E {label}']:.4f}" for label in methods])
    plot_rollouts(complete, methods)
    print(report, end="")
    print(f"Wrote {config.RESULTS_ROOT / 'evaluation.txt'}, test_runs.csv, test_rollouts.png")


if __name__ == "__main__":
    main()
