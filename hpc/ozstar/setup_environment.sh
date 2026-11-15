#!/bin/bash
# Run this once from an OzSTAR login node after cloning the repository.
set -euo pipefail

PROJECT_DIR=${1:?Usage: bash hpc/ozstar/setup_environment.sh /fred/.../Capabilities_In_Reinforcement_Learning}
cd "$PROJECT_DIR"

module purge
ml gcccore/12.2.0
ml python/3.10.8-bare

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Installs Atari ROMs when AutoROM is available through gymnasium's
# accept-rom-license dependency.
python -m AutoROM --accept-license
python -c "import gymnasium, torch; print('Gymnasium:', gymnasium.__version__); print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
