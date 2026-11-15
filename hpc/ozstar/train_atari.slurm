#!/bin/bash
# One environment × agent × seed per Slurm job.
# Override resources at submission time, e.g.:
# sbatch --time=2-00:00:00 --gres=gpu:a100:1 train_atari.slurm ...
#SBATCH --job-name=atari-train
#SBATCH --account=oz347
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:?Set PROJECT_DIR via --export when submitting.}
AGENT=${1:?Usage: train_atari.slurm <agent> <env-id> <seed> <timesteps>}
ENV_ID=${2:?Usage: train_atari.slurm <agent> <env-id> <seed> <timesteps>}
SEED=${3:?Usage: train_atari.slurm <agent> <env-id> <seed> <timesteps>}
TIMESTEPS=${4:?Usage: train_atari.slurm <agent> <env-id> <seed> <timesteps>}

# ── Resolve the directory that contains this script (and all trainers) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Activate the pre-existing virtual environment directly ──
VENV_DIR="/fred/oz347/rtandon/action_masking/virtual_env_for_action_masking_on_atari_env"
source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR" "$SCRIPT_DIR/slurm_logs"
cd "$SCRIPT_DIR"

echo "Host: $(hostname)"
echo "Script dir: $SCRIPT_DIR"
echo "Job: ${SLURM_JOB_ID}; agent=${AGENT}; env=${ENV_ID}; seed=${SEED}; steps=${TIMESTEPS}"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

case "$AGENT" in
  # ── VQ-VAE masked / full-action-space baselines ──
  ddqn_baseline|ddqn_vqvae_masked)
    srun python "$SCRIPT_DIR/dqn_atari_vqvae_masked.py" \
      --env-id "$ENV_ID" \
      --full-action-space \
      --agent "$AGENT" \
      --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --mask-penalty 1.0 \
      --vqvae-epochs 10 \
      --num-embeddings 256 \
      --embedding-dim 8 \
      --results-dir "$RESULTS_DIR"
    ;;
  ppo_baseline|ppo_vqvae_masked)
    srun python "$SCRIPT_DIR/ppo_atari_vqvae_masked.py" \
      --env-id "$ENV_ID" \
      --full-action-space \
      --agent "$AGENT" \
      --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --mask-penalty 1.0 \
      --vqvae-epochs 10 \
      --num-embeddings 256 \
      --embedding-dim 8 \
      --results-dir "$RESULTS_DIR"
    ;;
  # ── Standard baselines (minimal action space, no VQ-VAE) ──
  dqn_baseline_standard)
    srun python "$SCRIPT_DIR/dqn_atari_baseline.py" \
      --env-id "$ENV_ID" \
      --agent "$AGENT" \
      --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --results-dir "$RESULTS_DIR"
    ;;
  ppo_baseline_standard)
    srun python "$SCRIPT_DIR/ppo_atari_baseline.py" \
      --env-id "$ENV_ID" \
      --agent "$AGENT" \
      --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --results-dir "$RESULTS_DIR"
    ;;
  *)
    echo "Unknown agent: $AGENT" >&2
    exit 2
    ;;
esac
