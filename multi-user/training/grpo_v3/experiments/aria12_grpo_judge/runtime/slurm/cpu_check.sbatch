#!/bin/bash
#SBATCH --job-name=aria3-fec-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
: "${ARIA_ROOT:?需要本次独立运行根}"
: "${ARIA_PYTHON:?需要已验证的Python}"
cd "$ARIA_ROOT"
exec "$ARIA_PYTHON" -u bootstrap.py --mode cpu --config configs/train.json
