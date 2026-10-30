#!/usr/bin/env python3
"""Turn the raw runs into one table of segments for the n_net.

    python3 dataset.py

For every commanded segment it stores the command, the command before it, the
motion the current model predicts (core.nominal_step with the gains and offsets
in the robot repo's config.py) and the motion mocap measured, both in the
robot's frame at the start of the segment. The n_net learns the difference.

Writes data/segments.csv, data/model_params.json (the gains and offsets used,
so later config.py changes cannot silently change the targets) and
data/split.json (which whole runs are train, validation and test).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import config

sys.path.append(str(config.REPO_ROOT))
from core import Command, load_sent_commands, motion_kind, nominal_step, quaternion_to_theta_raw  # noqa: E402


KINDS = ("vx", "vy", "omega", "vx+vy", "vx+omega")
FEATURE_NAMES = ("vx", "vy", "omega", "dt", "prev_vx", "prev_vy", "prev_omega",
                 "nominal_dx", "nominal_dy", "nominal_dtheta") + tuple(f"kind_{k}" for k in KINDS)
TARGET_NAMES = ("residual_dx", "residual_dy", "residual_dtheta")
# The planner's command limits, read from the repo's config.py with the model.
LIMIT_NAMES = ("VX_MAX_MPS", "VY_MAX_MPS", "OMEGA_MAX_RADPS", "DIAG_SPEED_SUM_MAX_MPS",
               "ARC_VX_MAX_MPS", "ARC_OMEGA_MAX_RADPS")


# The current model ---------------------------------------------------------

def read_repo_model() -> dict:
    """Every MODEL_* gain, offset and lag plus the command limits in the repo's config.py."""
    spec = importlib.util.spec_from_file_location("repo_config", config.REPO_ROOT / "config.py")
    repo_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repo_config)
    values = {name: getattr(repo_config, name) for name in dir(repo_config)
              if name.startswith("MODEL_") or name in LIMIT_NAMES}
    try:
        commit = subprocess.run(["git", "-C", str(config.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"repo_commit": commit, "values": values}


def model_namespace(params: dict) -> SimpleNamespace:
    """What core.nominal_step expects as its cfg."""
    return SimpleNamespace(**params["values"])


def nominal_motion(command: Command, previous: Command | None, model) -> tuple[float, float, float]:
    """The current model's motion over one segment, in the robot frame at its start."""
    return nominal_step(0.0, 0.0, 0.0, command, model, previous=previous)


def compose(pose: tuple[float, float, float], motion) -> tuple[float, float, float]:
    """Pose after `motion` (dx, dy, dtheta in the robot frame) from `pose` (world frame)."""
    x, y, theta = pose
    dx, dy, dtheta = motion
    c, s = math.cos(theta), math.sin(theta)
    return x + c*dx - s*dy, y + s*dx + c*dy, theta + dtheta


def features(command: Command, previous: Command | None, nominal) -> list[float]:
    """The n_net's inputs, in FEATURE_NAMES order. No previous command means standstill."""
    previous = previous or Command(-1, 0.0, 0.0, 0.0, 0.0)
    kind = motion_kind(command)
    return [command.vx, command.vy, command.omega, command.dt,
            previous.vx, previous.vy, previous.omega, *nominal,
            *(1.0 if kind == k else 0.0 for k in KINDS)]


def in_limits(command: Command, limits: dict) -> bool:
    """Whether the planner could send this command under the repo's limits."""
    kind = motion_kind(command)
    if kind is None:
        return False
    if command.vx > limits["VX_MAX_MPS"] + 1e-9 or abs(command.vy) > limits["VY_MAX_MPS"] + 1e-9 \
            or abs(command.omega) > limits["OMEGA_MAX_RADPS"] + 1e-9:
        return False
    if kind == "vx+vy":
        return command.vx + abs(command.vy) <= limits["DIAG_SPEED_SUM_MAX_MPS"] + 1e-9
    if kind == "vx+omega":
        return (command.vx <= limits["ARC_VX_MAX_MPS"] + 1e-9
                and abs(command.omega) <= limits["ARC_OMEGA_MAX_RADPS"] + 1e-9)
    return True


# Raw files ------------------------------------------------------------------

def load_commands(path: Path, mocap_format: str) -> list[Command]:
    if mocap_format == "current":
        return load_sent_commands(path)
    # 2026-09-25: k,t_send,v,omega,dt[,vy]
    with path.open(newline="", encoding="utf-8-sig") as source:
        return [Command(int(row["k"]), float(row["v"]), float(row.get("vy") or 0.0),
                        float(row["omega"]), float(row["dt"]), float(row["t_send"]))
                for row in csv.DictReader(source)]


def load_mocap(path: Path, mocap_format: str, yaw_offset: float):
    """Tracked samples as arrays: time, body-centre x and y, and unwrapped corrected heading."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    samples = []
    for row in rows:
        if mocap_format == "current":
            if row["tracked"].strip().lower() not in {"1", "true", "yes"}:
                continue
            sample = (float(row["t"]), float(row["x"]), float(row["y"]), float(row["theta_raw"]))
        else:
            if row["tracking_valid"].strip().lower() not in {"1", "true", "yes"}:
                continue
            sample = (float(row["unix_time"]), float(row["x"]), -float(row["z"]),
                      quaternion_to_theta_raw(*(float(row[q]) for q in ("qx", "qy", "qz", "qw"))))
        if all(math.isfinite(value) for value in sample):
            samples.append(sample)
    if len(samples) < 2:
        raise ValueError(f"{path} has fewer than two tracked samples")
    data = np.array(sorted(samples))
    data = data[np.concatenate(([True], np.diff(data[:, 0]) > 0))]
    times, marker_x, marker_y = data[:, 0], data[:, 1], data[:, 2]
    theta = np.unwrap(data[:, 3]) + yaw_offset
    c, s = np.cos(theta), np.sin(theta)
    body_x = marker_x + c*config.MARKER_TO_BODY_X_M - s*config.MARKER_TO_BODY_Y_M
    body_y = marker_y + s*config.MARKER_TO_BODY_X_M + c*config.MARKER_TO_BODY_Y_M
    return times, body_x, body_y, theta


def _tracked(times: np.ndarray, start: float, end: float) -> bool:
    if times[0] > start or times[-1] < end:
        return False
    inside = (times[1:] > start) & (times[:-1] < end)
    return not np.any(np.diff(times)[inside] > config.MAX_TRACKING_GAP_S)


def run_segments(session: str, run_dir: Path, params: dict) -> list[dict]:
    """One row per commanded segment of one run."""
    run = run_dir.name
    mocap_format = config.MOCAP_FORMAT[session]
    commands = load_commands(run_dir / "commands.csv", mocap_format)
    times, body_x, body_y, theta = load_mocap(run_dir / "mocap.csv", mocap_format,
                                              config.yaw_offset(session, run))
    model = model_namespace(params)
    # Segment boundaries follow the plan from the first send time, as in core.trace_execution.
    boundaries = commands[0].t_send + np.concatenate(([0.0], np.cumsum([c.dt for c in commands])))
    rows = []
    for index, command in enumerate(commands):
        previous = commands[index-1] if index else None
        start, end = boundaries[index], boundaries[index+1]
        covered = motion_kind(command) is not None
        tracked = _tracked(times, start, end)
        planner_ok = in_limits(command, params["values"])
        excluded = f"{session}/{run}" in config.EXCLUDE_RUNS
        row = {"session": session, "run": run, "k": index,
               "vx": command.vx, "vy": command.vy, "omega": command.omega, "dt": command.dt,
               "prev_vx": previous.vx if previous else 0.0,
               "prev_vy": previous.vy if previous else 0.0,
               "prev_omega": previous.omega if previous else 0.0,
               "kind": motion_kind(command) or "none",
               "tracked": int(tracked), "covered": int(covered), "in_limits": int(planner_ok),
               "use": int(tracked and covered and not excluded
                          and (planner_ok or not config.ONLY_PLANNER_LIMITS))}
        if tracked:
            pose0 = tuple(float(np.interp(start, times, v)) for v in (body_x, body_y, theta))
            pose1 = tuple(float(np.interp(end, times, v)) for v in (body_x, body_y, theta))
            c, s = math.cos(pose0[2]), math.sin(pose0[2])
            wx, wy = pose1[0] - pose0[0], pose1[1] - pose0[1]
            actual = (c*wx + s*wy, -s*wx + c*wy, pose1[2] - pose0[2])
        else:
            pose0 = pose1 = actual = (math.nan,)*3
        nominal = nominal_motion(command, previous, model) if covered else (math.nan,)*3
        row.update(zip(("start_x", "start_y", "start_theta"), pose0))
        row.update(zip(("end_x", "end_y", "end_theta"), pose1))
        row.update(zip(("actual_dx", "actual_dy", "actual_dtheta"), actual))
        row.update(zip(("nominal_dx", "nominal_dy", "nominal_dtheta"), nominal))
        rows.append(row)
    return rows


# Table, arrays and split ------------------------------------------------------

def load_segments(path: Path = config.SEGMENTS_CSV, exclude=None) -> list[dict]:
    """Segment rows, without the sessions in `exclude` (default config.EXCLUDE_SESSIONS)."""
    exclude = set(config.EXCLUDE_SESSIONS if exclude is None else exclude)
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["session"] not in exclude]
    for row in rows:
        for key, value in row.items():
            if key not in ("session", "run", "kind"):
                row[key] = float(value)
    return rows


def run_id(row: dict) -> str:
    return f"{row['session']}/{row['run']}"


def row_commands(row: dict) -> tuple[Command, Command | None]:
    """A segment's command and the one before it (None for a run's first segment)."""
    k = int(row["k"])
    command = Command(k, row["vx"], row["vy"], row["omega"], row["dt"])
    previous = Command(k-1, row["prev_vx"], row["prev_vy"], row["prev_omega"], 0.0) if k else None
    return command, previous


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Inputs (FEATURE_NAMES) and targets (actual minus current model) of used segments."""
    used = [row for row in rows if row["use"]]
    x = np.array([features(*row_commands(row),
                           (row["nominal_dx"], row["nominal_dy"], row["nominal_dtheta"]))
                  for row in used], dtype=np.float32)
    y = np.array([[row["actual_dx"] - row["nominal_dx"], row["actual_dy"] - row["nominal_dy"],
                   row["actual_dtheta"] - row["nominal_dtheta"]] for row in used], dtype=np.float32)
    return x, y


def make_split(rows: list[dict]) -> dict[str, list[str]]:
    """Whole runs to train, validation and test, drawn separately in each session."""
    rng = random.Random(config.SPLIT_SEED)
    split = {"train": [], "valid": [], "test": []}
    sessions = sorted({row["session"] for row in rows})
    for session in sessions:
        runs = sorted({run_id(row) for row in rows if row["session"] == session and row["use"]})
        rng.shuffle(runs)
        if config.TEST_RUNS_PER_SESSION is None:
            test = runs[:round(config.TEST_SHARE * len(runs))]
        else:
            incomplete = {run_id(row) for row in rows if row["session"] == session and not row["use"]}
            test = [run for run in runs if run not in incomplete][
                :config.TEST_RUNS_PER_SESSION.get(session, 0)]
        rest = [run for run in runs if run not in test]
        n_valid = round(config.VALID_SHARE * len(runs))
        split["test"] += test
        split["valid"] += rest[:n_valid]
        split["train"] += rest[n_valid:]
    return {part: sorted(runs) for part, runs in split.items()}


def rows_in(rows: list[dict], runs: list[str]) -> list[dict]:
    wanted = set(runs)
    return [row for row in rows if run_id(row) in wanted]


def build() -> None:
    params = read_repo_model()
    rows, failures = [], []
    # Every session goes in segments.csv; EXCLUDE_SESSIONS is applied when loading it.
    for session_dir in sorted(p for p in config.RAW_ROOT.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            try:
                rows += run_segments(session_dir.name, run_dir, params)
            except (OSError, ValueError, KeyError) as error:
                failures.append(f"{session_dir.name}/{run_dir.name}: {error}")
    config.SEGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with config.SEGMENTS_CSV.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config.MODEL_PARAMS_JSON.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    split = make_split([row for row in rows if row["session"] not in config.EXCLUDE_SESSIONS])
    config.SPLIT_JSON.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    print(f"Current model from {config.REPO_ROOT} (commit {params['repo_commit']})")
    print(f"{'session':28s} {'runs':>5s} {'segments':>9s} {'used':>6s}  dropped")
    for session in sorted({row["session"] for row in rows}):
        mine = [row for row in rows if row["session"] == session]
        reasons = Counter("not tracked" if not row["tracked"] else
                          "not in model" if not row["covered"] else "other"
                          for row in mine if not row["use"])
        print(f"{session:28s} {len({row['run'] for row in mine}):5d} {len(mine):9d} "
              f"{sum(int(row['use']) for row in mine):6d}  "
              + ", ".join(f"{n} {reason}" for reason, n in sorted(reasons.items()))
              + ("  (excluded: config.EXCLUDE_SESSIONS)" if session in config.EXCLUDE_SESSIONS else ""))
    for part, runs in split.items():
        print(f"{part:5s}: {len(runs)} runs, "
              f"{sum(int(r['use']) for r in rows_in(rows, runs))} segments")
    for failure in failures:
        print("skipped", failure)
    print(f"Wrote {config.SEGMENTS_CSV}, {config.MODEL_PARAMS_JSON.name}, {config.SPLIT_JSON.name}")


if __name__ == "__main__":
    build()
