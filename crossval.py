#!/usr/bin/env python3
"""Cross-validation by run: how stable the n_net's improvement is across splits.

    python3 crossval.py
    python3 crossval.py --exclude --name crossval_all       # every session

Each repeat deals every session's runs into CV_FOLDS folds. Each fold is the
test set once; VALID_SHARE of each session's other runs are the validation set
(early stopping) and the rest train the n_net and the linear correction. So
every run is tested once per repeat, by networks that never saw it. The
current model's gains were fitted on the 2026-10-21 pilots, so on those the
current model is in-sample and the others are not.

Writes results/crossval.txt and results/crossval_runs.csv (or --name).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch

import config
import dataset
from evaluate import Ridge, rollout_errors
from model import pick_device
from train import train_network


METHODS = ("current model", "+ linear", "+ n_net")


def make_splits(rows: list[dict], repeat: int) -> list[dict[str, list[str]]]:
    """CV_FOLDS train/valid/test splits of whole runs, dealt separately in each session.

    Each session draws from its own seed, so adding or leaving out a session
    does not move the other sessions' runs between folds.
    """
    by_session = defaultdict(list)
    for run in sorted({dataset.run_id(row) for row in rows if row["use"]}):
        by_session[run.split("/")[0]].append(run)
    rngs = {session: random.Random(f"{config.CV_SEED}/{repeat}/{session}") for session in by_session}
    fold_of = {}
    for session, runs in by_session.items():
        rngs[session].shuffle(runs)
        offset = rngs[session].randrange(config.CV_FOLDS)   # small sessions do not all start in fold 0
        fold_of.update({run: (i + offset) % config.CV_FOLDS for i, run in enumerate(runs)})
    splits = []
    for fold in range(config.CV_FOLDS):
        split = {"train": [], "valid": [], "test": []}
        for session, runs in by_session.items():
            rest = [run for run in runs if fold_of[run] != fold]
            rngs[session].shuffle(rest)
            n_valid = round(config.VALID_SHARE * len(runs))
            split["test"] += [run for run in runs if fold_of[run] == fold]
            split["valid"] += rest[:n_valid]
            split["train"] += rest[n_valid:]
        splits.append(split)
    return splits


def run_fold(rows: list[dict], split: dict, device: torch.device, seed: int):
    """Train on one split; per-segment errors and whole-run E of its test runs."""
    x_train, y_train = dataset.arrays(dataset.rows_in(rows, split["train"]))
    x_valid, y_valid = dataset.arrays(dataset.rows_in(rows, split["valid"]))
    net, result = train_network(x_train, y_train, x_valid, y_valid, device, seed, verbose=False)
    ridge = Ridge(x_train, y_train, config.RIDGE_LAMBDA)
    corrections = {
        "current model": lambda x: np.zeros((len(x), 3)),
        "+ linear": ridge,
        "+ n_net": lambda x: net.correction(torch.as_tensor(x, device=device)).cpu().numpy(),
    }
    test_rows = dataset.rows_in(rows, split["test"])
    used = [row for row in test_rows if row["use"]]
    x, y = dataset.arrays(test_rows)
    segments = [{"session": row["session"], "in_limits": row["in_limits"]} for row in used]
    for name, correct in corrections.items():
        error = y - correct(x)
        for segment, e in zip(segments, error):
            segment[name] = (float(e[0]**2 + e[1]**2), float(e[2]**2))

    by_run = defaultdict(list)
    for row in test_rows:
        by_run[dataset.run_id(row)].append(row)
    runs = []
    for name, run_rows in sorted(by_run.items()):
        run_rows.sort(key=lambda r: r["k"])
        if not all(row["use"] for row in run_rows):
            continue
        x_run, _ = dataset.arrays(run_rows)
        record = {"run": name, "session": name.split("/")[0], "segments": len(run_rows)}
        for method, correct in corrections.items():
            errors = rollout_errors(run_rows, correct(x_run))
            record[method] = max(errors)
            record[f"sq {method}"] = sum(e*e for e in errors)
        runs.append(record)
    return segments, runs, result["best_epoch"]


def segment_rms(segments: list[dict], method: str) -> tuple[float, float]:
    return (100*math.sqrt(np.mean([s[method][0] for s in segments])),
            math.degrees(math.sqrt(np.mean([s[method][1] for s in segments]))))


def spread(values: list[float]) -> str:
    return (f"mean {np.mean(values):5.2f}  sd {np.std(values, ddof=1):4.2f}  "
            f"min {min(values):5.2f}  max {max(values):5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exclude", nargs="*", metavar="SESSION",
                        help="sessions to leave out (default: config.EXCLUDE_SESSIONS; "
                             "give none to use every session)")
    parser.add_argument("--name", default="crossval",
                        help="results/NAME.txt and results/NAME_runs.csv")
    args = parser.parse_args()
    rows = dataset.load_segments(exclude=args.exclude)
    device = pick_device(config.DEVICE)
    folds, all_segments, all_runs = [], [], []
    started = time.time()
    for repeat in range(config.CV_REPEATS):
        for fold, split in enumerate(make_splits(rows, repeat)):
            segments, runs, epoch = run_fold(rows, split, device, config.TRAIN_SEED + fold)
            for run in runs:
                run.update(repeat=repeat, fold=fold)
            folds.append({"repeat": repeat, "fold": fold, "epoch": epoch,
                          "segments": segments, "runs": runs})
            all_segments += segments
            all_runs += runs
            seg = {m: segment_rms(segments, m)[0] for m in METHODS}
            mean_e = {m: np.mean([r[m] for r in runs]) for m in METHODS}
            print(f"repeat {repeat} fold {fold}: {len(runs):2d} runs, epoch {epoch:3d}, "
                  f"segment cm " + " / ".join(f"{seg[m]:4.1f}" for m in METHODS)
                  + "   mean E m " + " / ".join(f"{mean_e[m]:4.2f}" for m in METHODS)
                  + f"   ({time.time() - started:.0f} s)", flush=True)

    lines = [f"Cross-validation by run: {config.CV_FOLDS} folds x {config.CV_REPEATS} repeats "
             f"= {len(folds)} trainings; columns are " + " / ".join(METHODS),
             "Sessions: " + ", ".join(sorted({row["session"] for row in rows})), "",
             "Per fold (test runs only)",
             f"{'':12s}{'runs':>5s}{'epoch':>6s}   {'segment error cm':>24s}   {'mean E m':>20s}"
             f"   {'q90 E m':>20s}"]
    for f in folds:
        seg = [segment_rms(f["segments"], m)[0] for m in METHODS]
        mean_e = [np.mean([r[m] for r in f["runs"]]) for m in METHODS]
        q90 = [np.quantile([r[m] for r in f["runs"]], 0.9) for m in METHODS]
        lines.append(f"r{f['repeat']} fold {f['fold']}  {len(f['runs']):4d}{f['epoch']:6d}   "
                     + " / ".join(f"{v:6.1f}" for v in seg) + "   "
                     + " / ".join(f"{v:5.2f}" for v in mean_e) + "   "
                     + " / ".join(f"{v:5.2f}" for v in q90))

    lines += ["", "Ratio to the current model across the folds (below 1 = better)"]
    for method in ("+ linear", "+ n_net"):
        seg_ratio = [segment_rms(f["segments"], method)[0] / segment_rms(f["segments"], METHODS[0])[0]
                     for f in folds]
        e_ratio = [np.mean([r[method] for r in f["runs"]]) / np.mean([r[METHODS[0]] for r in f["runs"]])
                   for f in folds]
        lines.append(f"{method:9s} segment error  {spread(seg_ratio)}")
        lines.append(f"{'':9s} mean E         {spread(e_ratio)}")

    lines += ["", f"Pooled over all folds and repeats (each run counted {config.CV_REPEATS}x)",
              f"{'':28s}{'segments':>9s}   {'segment cm':>20s}   {'heading deg':>17s}"
              f"   {'runs':>5s}   {'mean E m':>20s}   {'q90 E m':>20s}"]
    groups = [(s, s) for s in sorted({s["session"] for s in all_segments})] + [("all", None)]
    if not config.ONLY_PLANNER_LIMITS:
        groups.insert(-1, ("planner limits only", "limits"))
    for label, key in groups:
        segs = [s for s in all_segments
                if key is None or (key == "limits" and s["in_limits"]) or s["session"] == key]
        runs = [r for r in all_runs if key in (None, "limits") or r["session"] == key]
        line = (f"{label:28s}{len(segs):9d}   "
                + " / ".join(f"{segment_rms(segs, m)[0]:5.1f}" for m in METHODS) + "   "
                + " / ".join(f"{segment_rms(segs, m)[1]:4.1f}" for m in METHODS))
        if runs and key != "limits":
            line += (f"   {len(runs):5d}   " + " / ".join(f"{np.mean([r[m] for r in runs]):5.2f}"
                                                        for m in METHODS)
                     + "   " + " / ".join(f"{np.quantile([r[m] for r in runs], 0.9):5.2f}"
                                          for m in METHODS))
        lines.append(line)

    better = sum(r["+ n_net"] < r["current model"] for r in all_runs)
    better_linear = sum(r["+ n_net"] < r["+ linear"] for r in all_runs)
    lines += ["", f"Whole runs where + n_net has lower E: than the current model {better} of "
              f"{len(all_runs)}, than + linear {better_linear} of {len(all_runs)}",
              "Largest increases of E with + n_net over the current model:"]
    for r in sorted(all_runs, key=lambda r: r["current model"] - r["+ n_net"])[:5]:
        lines.append(f"  {r['run']:40s} r{r['repeat']}  {r['current model']:.2f} -> {r['+ n_net']:.2f} m")

    config.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    (config.RESULTS_ROOT / f"{args.name}.txt").write_text(report, encoding="utf-8")
    with (config.RESULTS_ROOT / f"{args.name}_runs.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["repeat", "fold", "run", "segments"] + [f"E_m {m}" for m in METHODS])
        for r in all_runs:
            writer.writerow([r["repeat"], r["fold"], r["run"], r["segments"]]
                            + [f"{r[m]:.4f}" for m in METHODS])
    print()
    print(report, end="")
    print(f"Wrote {config.RESULTS_ROOT / f'{args.name}.txt'} and {args.name}_runs.csv "
          f"({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
