#!/bin/bash
# Plantilla de job para la DGX. Ajusta partición, cuenta y módulos a lo que indique
# el servicio de la DGX de ICAI (puede que usen SLURM, Docker o acceso directo).
#SBATCH --job-name=layercollapse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # caché de modelos fuera del repo

CONFIG="${1:-configs/default.yaml}"
python scripts/extract_metrics.py --config "$CONFIG"
