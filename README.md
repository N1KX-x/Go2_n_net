# Go2 n_net

A small neural network that corrects the current Go2 motion model. For every
commanded segment, the current model (`core.nominal_step` with the gains,
offsets and turn lags in `holonomic_hardware_protocol/config.py`) predicts the
motion; the n_net learns what it gets wrong, from mocap. Prediction = current
model + n_net. Before training the n_net outputs zero, so it starts as exactly
the current model.

The network follows the Hugging Face Fashion-MNIST MLP (Linear, ReLU, Dropout
0.2, batches of 32, train/validate loop), changed for regression: MSE loss, 3
outputs (dx, dy, dtheta in the robot frame), inputs standardized, a separate
validation set split by whole runs, and early stopping.

## Run

```bash
python3 -m venv ~/.venvs/go2_n_net
~/.venvs/go2_n_net/bin/pip install -r requirements.txt
~/.venvs/go2_n_net/bin/python dataset.py    # raw runs -> data/segments.csv, split.json, model_params.json
~/.venvs/go2_n_net/bin/python train.py      # -> checkpoints/n_net.pt, results/loss_curve.png
~/.venvs/go2_n_net/bin/python evaluate.py   # -> results/evaluation.txt, test_runs.csv, test_rollouts.png
~/.venvs/go2_n_net/bin/python crossval.py   # 5 folds x 2 repeats by run -> results/crossval.txt (~5 min)
~/.venvs/go2_n_net/bin/python crossval.py --exclude --name crossval_all   # same, with every session
~/.venvs/go2_n_net/bin/python predict.py PLAN.csv --start X Y THETA
```

From other code: `from predict import CorrectedModel; CorrectedModel().rollout(commands, pose)`
returns the poses at every boundary, like `core.nominal_rollout`.

After the gains in the repo's `config.py` change, run `dataset.py` and
`train.py` again: the targets are "actual minus current model", so a network
trained on old gains does not fit new ones. `evaluate.py` and `predict.py`
warn when they differ.

## Files

| File | What it does |
|---|---|
| `config.py` | Repo path, mocap format and yaw offset per session, split shares, hyperparameters |
| `dataset.py` | Reads commands + mocap, cuts runs into segments, current-model motion vs measured motion, run split |
| `model.py` | The MLP (inputs -> 64 -> 32 -> 3), device choice, checkpoint loading |
| `train.py` | `Learner` with `train_epoch` / `validate_epoch` / `fit`, saves the best network |
| `evaluate.py` | Current model vs + linear correction vs + n_net, per segment and over whole runs |
| `crossval.py` | Same comparison over many splits by run, to see how stable the improvement is |
| `predict.py` | Current model + n_net for a plan; `CorrectedModel` for the planner |
| `data/raw/<session>/<run>/` | `commands.csv`, `mocap.csv`, `publish_log.csv` of every run; `data/manifest.csv` lists where each came from |

Inputs: command (vx, vy, omega, dt), previous command (zeros from standstill),
the current model's predicted motion, and the command kind (one-hot).

## Data

| Session | Runs | Segments used | Notes |
|---|---|---|---|
| 2026-09-25 | 35 | 61 of 88 | Older command format and mocap axes (x, -z); 2-10 s segments |
| 2026-10-09 | 18 | 150 of 303 | Many commands outside the model (backward vx, all three axes at once) |
| 2026-10-09_attempt2 | 2 | 8 of 12 | Two short runs |
| 2026-10-14 | 31 | excluded | 1 s segments; left out via `EXCLUDE_SESSIONS` in `config.py` |
| 2026-10-21 | 80 | 1200 | The RRT pilots the current gains were fitted on |
| 2026-10-21_yaw_calibration | 6 | 6 | 5 s straight walks |
| 2026-10-27_random | 50 | 1500 | `phase0_random`: 60 s runs, 2 s rows, planned with the raw model (no gains); the 3 `_discarded` runs are left out |
| 2026-10-28_smooth | 40 | 1170 | `phase0_smooth`: 60 s runs, 2 s rows with gradual ramps, raw-model planning; the 8 `_discarded` runs are left out, and smooth_004 (robot never moved) via `EXCLUDE_RUNS` |
| 2026-10-30 | 79 | 1125 | `Resuls/1030phase0`: RRT pilots planned with the raw model, 15 x 2 s; pilot_032-034 (mocap heading ~90 deg off) and pilot_050 left out via `EXCLUDE_RUNS` |

Test set: `TEST_RUNS_PER_SESSION` in `config.py` (now 5 random complete 2026-10-21
pilots + 5 random 2026-10-30 pilots); 15% of every session's other runs validate,
the rest train. Set it to `None` for `TEST_SHARE` of every session.

Only commands the current model covers are used (`core.motion_kind` is not
None). Set `ONLY_PLANNER_LIMITS = True` in `config.py` to also drop commands
outside the planner's limits.

## On another computer (e.g. a PC with an NVIDIA GPU)

Copy this folder, clone `holonomic_hardware_protocol`, and set `GO2_REPO` to
the clone (or edit `REPO_ROOT` in `config.py`). Install the CUDA build of
torch. `DEVICE = "auto"` picks CUDA, then Apple MPS, then CPU. The network is
small enough that a GPU gives no real speed-up: training takes under a minute
on a laptop.
