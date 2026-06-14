"""
tune_hyperparams.py
====================
Automated hyperparameter search for the Baseline Double DQN agent using Optuna.

Objective:
    Maximize the mean goal_reached rate over the LAST 20% of training episodes.
    This tells us whether the agent is still reliably solving the environment
    once exploration has tapered off -- i.e., whether it genuinely learned.

Usage:
    python src/tune_hyperparams.py

    Optional flags:
        --n-trials      Number of Optuna trials to run      (default: 50)
        --timesteps     Steps per trial                      (default: 30000)
        --env-id        Environment to tune on               (default: MiniGrid-Empty-5x5-v0)
        --output        Path to save the results CSV         (default: results/optuna_results.csv)
        --n-jobs        Parallel trials (1 = sequential)     (default: 1)
        --seed          Base seed for reproducibility        (default: 42)
        --study-name    Name of the Optuna study             (default: dqn_baseline_tune)

Requirements:
    pip install optuna
"""

import argparse
import csv
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable when run from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import optuna
from optuna.samplers import TPESampler

from dqn_common import train


# ============================================================================
# SEARCH SPACE DEFINITION
# ============================================================================

def build_args(trial, base_timesteps, env_id, seed):
    """
    Constructs a hyperparameter configuration for one Optuna trial.

    Broad ranges are intentional: Optuna TPE sampler learns quickly which
    regions of the space are promising and focuses its budget there.
    """
    args = types.SimpleNamespace()

    # ---- Environment ----
    args.env_id = env_id
    args.seed = seed
    args.exp_name = "dqn_baseline"
    args.total_timesteps = base_timesteps
    args.action_set = "task"

    # ---- Learning Rate ----
    # Broad range: 1e-5 (very cautious) to 1e-2 (very aggressive)
    # log=True makes Optuna sample evenly in log space so 1e-5 and 1e-2
    # are equally likely to be explored as 1e-4 and 1e-3.
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # ---- Batch Size ----
    # Larger batches = more stable gradient, smaller = faster per-step updates
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # ---- Replay Buffer Size ----
    args.buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 50000, 100000])

    # ---- Discount Factor ----
    args.gamma = trial.suggest_float("gamma", 0.90, 0.999)

    # ---- Target Network Update Frequency ----
    # How often (steps) target_net copies weights from q_net
    args.target_network_frequency = trial.suggest_int("target_network_frequency", 50, 1000)

    # ---- Exploration Schedule ----
    args.start_e = 1.0   # Always start fully random
    args.end_e = 0.0     # Decay to 0 (fully greedy at end) -- fixed per user request
    args.exploration_fraction = trial.suggest_float("exploration_fraction", 0.3, 0.9)

    # ---- Training Start ----
    args.learning_starts = trial.suggest_categorical("learning_starts", [500, 1000, 2000, 5000])

    # ---- Train Frequency ----
    args.train_frequency = trial.suggest_categorical("train_frequency", [1, 2, 4, 8])

    # ---- Network Architecture ----
    args.hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])

    # ---- Fixed / Logging Settings ----
    args.cuda = False
    args.save_model = False
    args.capture_video = False
    args.track = False
    args.wandb_project_name = "cleanRL"
    args.wandb_entity = None
    args.log_interval = 1000
    args.results_dir = "results/optuna_trials"

    # ---- Reward Shaping (disabled for baseline tuning) ----
    args.use_shaping = False
    args.stuck_penalty = -0.01

    return args


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def make_objective(base_timesteps, env_id, base_seed):
    """
    Returns the Optuna objective function with configuration baked in via closure.
    Optuna requires the signature to be f(trial) -> float.
    """
    def objective(trial):
        # Each trial gets a unique seed offset so results aren't
        # an artifact of one particular random sequence
        seed = base_seed + trial.number
        args = build_args(trial, base_timesteps, env_id, seed)

        # train() returns mean goal_reached rate over the last 20% of episodes
        goal_rate = train(args, use_shaping=False)
        return goal_rate

    return objective


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(study, output_path):
    """
    Saves the full trial history sorted by score (best first) to a CSV file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": trial.number,
            "goal_rate_last20pct": trial.value,
            "duration_seconds": (trial.datetime_complete - trial.datetime_start).total_seconds(),
        }
        row.update(trial.params)
        rows.append(row)

    rows.sort(key=lambda r: r["goal_rate_last20pct"], reverse=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {output}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Baseline Double DQN")
    parser.add_argument("--n-trials",   type=int,  default=50,
                        help="Number of Optuna trials to run (default: 50)")
    parser.add_argument("--timesteps",  type=int,  default=30000,
                        help="Training steps per trial (default: 30000)")
    parser.add_argument("--env-id",     type=str,  default="MiniGrid-Empty-5x5-v0",
                        help="Environment to tune on (default: MiniGrid-Empty-5x5-v0)")
    parser.add_argument("--output",     type=str,  default="results/optuna_results.csv",
                        help="Path to save results CSV (default: results/optuna_results.csv)")
    parser.add_argument("--n-jobs",     type=int,  default=1,
                        help="Parallel trials -- 1 = sequential (default: 1)")
    parser.add_argument("--seed",       type=int,  default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--study-name", type=str,  default="dqn_baseline_tune",
                        help="Optuna study name (default: dqn_baseline_tune)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Double DQN Baseline -- Hyperparameter Search")
    print("=" * 60)
    print(f"  Environment  : {args.env_id}")
    print(f"  Trials       : {args.n_trials}")
    print(f"  Steps/trial  : {args.timesteps}")
    print(f"  Objective    : Mean goal_reached rate (last 20% of episodes)")
    print(f"  Output       : {args.output}")
    print("=" * 60)

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
    )

    # Silence Optuna's per-trial verbose output -- the tqdm bar shows progress
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        make_objective(args.timesteps, args.env_id, args.seed),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Print summary
    best = study.best_trial
    print("\n" + "=" * 60)
    print("  Best Trial")
    print("=" * 60)
    print(f"  Trial #      : {best.number}")
    print(f"  Goal Rate    : {best.value:.4f}  (last 20% of episodes)")
    print("\n  Best Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<35} = {v}")

    save_results(study, args.output)
    print("\nDone! Use the best params above to update dqn_common.py defaults.")


if __name__ == "__main__":
    main()
