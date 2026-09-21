#!/bin/bash
#SBATCH --job-name=charts_5min_realistic
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=slurm_charts_5min_realistic_%j.log

set -euo pipefail

# Under SLURM execution, $BASH_SOURCE[0] resolves to the spool directory (/var/spool/slurm/slurmd/job.../slurm_script).
# $SLURM_SUBMIT_DIR is provided by SLURM and contains the working directory where 'sbatch' was run.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f "${SUBMIT_DIR}/trillium_fengine_5min_realistic.sbatch" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../../trillium_fengine_5min_realistic.sbatch" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [ -d "/scratch/ferpb/kotekan" ] && [ -f "/scratch/ferpb/kotekan/trillium_fengine_5min_realistic.sbatch" ]; then
    REPO_ROOT="/scratch/ferpb/kotekan"
else
    # Fallback search
    CANDIDATE="$(find "${SUBMIT_DIR}" -maxdepth 3 -name "trillium_fengine_5min_realistic.sbatch" 2>/dev/null | head -n 1)"
    if [ -n "${CANDIDATE}" ]; then
        REPO_ROOT="$(dirname "${CANDIDATE}")"
    else
        echo "[ERROR] Cannot locate trillium_fengine_5min_realistic.sbatch from SUBMIT_DIR=${SUBMIT_DIR}"
        exit 1
    fi
fi

cd "${REPO_ROOT}"
echo "Executing: ${REPO_ROOT}/trillium_fengine_5min_realistic.sbatch"
exec bash "${REPO_ROOT}/trillium_fengine_5min_realistic.sbatch"
