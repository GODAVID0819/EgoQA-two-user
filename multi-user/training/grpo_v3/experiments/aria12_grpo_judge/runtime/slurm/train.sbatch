#!/bin/bash
#SBATCH --job-name=aria12-gated24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=750G
#SBATCH --time=09:00:00
set -euo pipefail
: "${ARIA_ROOT:?需要本次独立运行根}"
: "${ARIA_PYTHON:?需要已验证的Python}"
cd "$ARIA_ROOT"
exec "$ARIA_PYTHON" -u bootstrap.py --mode train --config configs/train.json
