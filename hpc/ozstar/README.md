# OzSTAR Atari experiment launcher

Transfer or clone the entire repository onto `/fred` (not the login-node home
directory). The runtime-relevant files are:

- `src/dqn_atari_vqvae_masked.py` — Double-DQN baseline and masked agent (full 18-action space).
- `src/ppo_atari_vqvae_masked.py` — PPO baseline and masked agent (full 18-action space).
- `src/dqn_atari_baseline.py` — **Standard DQN baseline** (minimal action space, no VQ-VAE).
- `src/ppo_atari_baseline.py` — **Standard PPO baseline** (minimal action space, no VQ-VAE).
- `hpc/ozstar/` — setup, batch, and submission scripts.

The notebook and GitHub Actions workflow are not required for training on
OzSTAR. Keep the whole checkout anyway so the code stays versioned and the
plotting scripts are available after training.

## Available agents

| Agent name              | Script                          | Action space | VQ-VAE |
|-------------------------|---------------------------------|--------------|--------|
| `ddqn_baseline`         | `dqn_atari_vqvae_masked.py`    | Full (18)    | No     |
| `ddqn_vqvae_masked`     | `dqn_atari_vqvae_masked.py`    | Full (18)    | Yes    |
| `ppo_baseline`          | `ppo_atari_vqvae_masked.py`    | Full (18)    | No     |
| `ppo_vqvae_masked`      | `ppo_atari_vqvae_masked.py`    | Full (18)    | Yes    |
| `dqn_baseline_standard` | `dqn_atari_baseline.py`        | Minimal      | No     |
| `ppo_baseline_standard` | `ppo_atari_baseline.py`        | Minimal      | No     |

The **standard baselines** use the environment's default minimal action space
(typically 4–6 actions) and represent conventional Atari DQN/PPO benchmarks.
They share the same network architecture as the masked variants for fair comparison.

## Virtual environment

The scripts use the pre-existing virtual environment at:

```
/fred/oz347/rtandon/action_masking/virtual_env_for_action_masking_on_atari_env
```

The SLURM batch script activates this environment automatically—**no need to
create a new venv or install `requirements.txt`** for each job.

## Smoke test first

Always run one short job before launching the full matrix:

```bash
python hpc/ozstar/submit_atari_matrix.py \
  --project-dir "$PWD" \
  --envs ALE/DemonAttack-v5 \
  --agents dqn_baseline_standard \
  --seeds 1 \
  --timesteps 200000 \
  --time 04:00:00
```

Then inspect `hpc/ozstar/slurm_logs/` and the matching directory under
`results/`.

## Run only standard baselines

To submit just the two standard baselines across all games and 3 seeds:

```bash
python hpc/ozstar/submit_atari_matrix.py \
  --project-dir "$PWD" \
  --agents dqn_baseline_standard ppo_baseline_standard \
  --seeds 1 2 3 \
  --timesteps 10000000 \
  --time 2-00:00:00
```

## Full experiment

After the smoke test succeeds, submit the selected matrix. This example runs
all four games, **six agents**, and seeds 1–3 (72 independent GPU jobs):

```bash
python hpc/ozstar/submit_atari_matrix.py \
  --project-dir "$PWD" \
  --seeds 1 2 3 \
  --timesteps 10000000 \
  --time 2-00:00:00
```

Use `--dry-run` to print commands without submitting them. Submit from the
login host associated with the target cluster. OzSTAR's current guidance is
not to pin a partition; use `--partition milan-gpu --gres gpu:a100:1` only if
your account's established configuration requires it.

Monitor jobs with `squeue -u "$USER"`; inspect logs with `tail -f
hpc/ozstar/slurm_logs/<job-name>-<job-id>.out`.

## Important limitation

The current trainers write final run files but do not yet support checkpoint
resume. Request a time limit that comfortably fits one 10M-step run, or add
checkpoint/resume before launching a large production matrix. This is
especially important for PPO, whose present implementation uses one Atari
environment per GPU job.
