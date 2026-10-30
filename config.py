"""Settings for the Go2 n_net: paths, calibration per session, and training hyperparameters."""

import os
from pathlib import Path


# Paths ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
# The robot repo whose core.py (the current model) and config.py (its gains and
# offsets) the n_net builds on. On another computer, clone it and either edit
# this line or set the GO2_REPO environment variable.
REPO_ROOT = Path(os.environ.get("GO2_REPO",
                                Path.home() / "Desktop" / "holonomic_hardware_protocol"))
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
SEGMENTS_CSV = PROJECT_ROOT / "data" / "segments.csv"
MODEL_PARAMS_JSON = PROJECT_ROOT / "data" / "model_params.json"
SPLIT_JSON = PROJECT_ROOT / "data" / "split.json"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "n_net.pt"
RESULTS_ROOT = PROJECT_ROOT / "results"


# Mocap and calibration -----------------------------------------------------
# "legacy" (2026-09-25): ground plane is Motive x and -z, heading from the
# quaternion. "current": the x, y, theta_raw columns written by the collector.
MOCAP_FORMAT = {
    "2026-09-25": "legacy",
    "2026-10-09": "current",
    "2026-10-09_attempt2": "current",
    "2026-10-14": "current",
    "2026-10-21": "current",
    "2026-10-21_yaw_calibration": "current",
    "2026-10-27_random": "current",
    "2026-10-28_smooth": "current",
    "2026-10-30": "current",
}
# The yaw offset that session was analysed with (its phase0_results.csv, or the
# processed summary.txt for 2026-09-25).
YAW_OFFSET_RAD = {
    "2026-09-25": 2.338883,
    "2026-10-09": 2.40395144164075,
    "2026-10-09_attempt2": 2.40395144164075,
    "2026-10-14": 2.4086,
    "2026-10-21": 2.3810,
    "2026-10-21_yaw_calibration": 2.3810,
    "2026-10-27_random": 2.381,          # run.json of every run
    "2026-10-28_smooth": 2.381,          # run.json of every run
    "2026-10-30": 2.381,                 # same calibration (user, 2026-10-30)
}
# 2026-09-25 groups A-D and pilot_30 were recorded before the recalibration.
EARLY_0925_RUNS = ("group_A_", "group_B_", "group_C_", "group_D_", "group_E_pilot_30")
EARLY_0925_YAW_OFFSET_RAD = 2.409282
# Marker to body centre, robot frame (same rigid body in every session).
MARKER_TO_BODY_X_M = -0.07
MARKER_TO_BODY_Y_M = -0.01
# A segment is dropped if mocap loses the robot for longer than this inside it.
MAX_TRACKING_GAP_S = 0.20


def yaw_offset(session: str, run: str) -> float:
    if session == "2026-09-25" and run.startswith(EARLY_0925_RUNS):
        return EARLY_0925_YAW_OFFSET_RAD
    return YAW_OFFSET_RAD[session]


# Which segments to learn from ----------------------------------------------
# Only commands the current model covers are used (core.motion_kind is not
# None): this drops backward vx and vx+vy+omega together. True also drops
# commands outside the planner's limits in the repo's config.py.
ONLY_PLANNER_LIMITS = False
# Sessions left out entirely. 2026-10-14 ("1014 30phase 0 trial", 1 s segments)
# is out at the user's request (2026-10-28).
EXCLUDE_SESSIONS: tuple[str, ...] = ("2026-10-14",)
# "session/run". smooth_004: the robot stood still the whole run (mocap live,
# commands published, 0.00 m moved). 10-30 pilot_032-034: mocap heading ~90 deg
# off (rigid body redefined); pilot_050: dropped by the user in Robot_learning.
EXCLUDE_RUNS: tuple[str, ...] = ("2026-10-28_smooth/smooth_004",
                                 "2026-10-30/pilot_032", "2026-10-30/pilot_033",
                                 "2026-10-30/pilot_034", "2026-10-30/pilot_050")


# Split (whole runs, per session) -------------------------------------------
VALID_SHARE = 0.15
TEST_SHARE = 0.15
# Test runs drawn at random (only from runs whose every segment is usable) from
# the listed sessions; the other sessions only train and validate. None means
# TEST_SHARE of every session. 5 + 5 RRT pilots: the user's choice (2026-10-30).
TEST_RUNS_PER_SESSION: dict[str, int] | None = {"2026-10-21": 5, "2026-10-30": 5}
SPLIT_SEED = 20261025
# crossval.py: every run is tested once per repeat.
CV_FOLDS = 5
CV_REPEATS = 2
CV_SEED = 1


# Network and training (after the Hugging Face Fashion-MNIST MLP) -----------
HIDDEN_SIZES = (64, 32)
DROPOUT = 0.2
OPTIMIZER = "adam"         # "adam", or "sgd" as in the article
LEARNING_RATE = 1e-3       # the article's SGD used 0.005
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 400
EARLY_STOP_PATIENCE = 40   # epochs without a better validation loss
TRAIN_SEED = 0
DEVICE = "auto"            # "auto" picks cuda, then mps, then cpu
RIDGE_LAMBDA = 1.0         # linear baseline in evaluate.py
