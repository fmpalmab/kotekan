#!/bin/bash
#SBATCH --job-name=charts_64ant_corr
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=slurm_charts_64ant_corr_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

exec "${REPO_ROOT}/trillium_charts_64ant_fengine_correlator.sbatch"
