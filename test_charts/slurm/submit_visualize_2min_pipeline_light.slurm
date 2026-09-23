#!/bin/bash
#SBATCH --job-name=charts_viz_2min_light
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm_charts_viz_2min_light_%j.log

set -euo pipefail

echo "======================================================================"
echo " CHARTS 2-Min Pipeline Light-Theme Visualization & 3D Lobes Generator"
echo " Hostname       : $(hostname)"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo " Scratch Dir    : ${SLURM_TMPDIR:-/tmp}"
echo "======================================================================"

# 1. Load Modules
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 2>/dev/null || module load gcc || true
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python || true
module load scipy-stack 2>/dev/null || true
module load hdf5 2>/dev/null || true
module load ffmpeg 2>/dev/null || true

# Setup virtual environment
VENV_DIR="${SLURM_TMPDIR:-/tmp}/kotekan_viz_pyenv_${SLURM_JOB_ID:-manual}"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}" || true
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
pip install --no-index matplotlib numpy h5py 2>/dev/null || pip install matplotlib numpy h5py 2>/dev/null || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

# 2. Locate kotekan_scratch simulation data
if [ -n "${SIM_DATA_DIR:-}" ]; then
    SRC_DIR="${SIM_DATA_DIR}"
elif [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH}/kotekan_scratch" ]; then
    LATEST_RUN=$(ls -td "${SCRATCH}/kotekan_scratch"/charts_2min_* 2>/dev/null | head -n 1 || true)
    SRC_DIR="${LATEST_RUN:-${SCRATCH}/kotekan_scratch}"
elif [ -d "${REPO_ROOT}/test_charts/data/2min_pipeline" ]; then
    SRC_DIR="${REPO_ROOT}/test_charts/data/2min_pipeline"
else
    SRC_DIR="${REPO_ROOT}/kotekan_scratch"
fi

WIN03_DIR="${SRC_DIR}/win03UTC"
if [ ! -d "${WIN03_DIR}" ]; then
    WIN03_DIR="${SRC_DIR}"
fi

FINAL_OUT_DIR="${OUTPUT_DIR:-${SRC_DIR}/plots_light}"
mkdir -p "${FINAL_OUT_DIR}"

# 3. Generate Visualizations
VIZ_SCRIPT="${REPO_ROOT}/test_charts/visualize_3d_lobes_and_pipeline_light.py"
if [ ! -f "${VIZ_SCRIPT}" ] && [ -f "${REPO_ROOT}/kotekan/test_charts/visualize_3d_lobes_and_pipeline_light.py" ]; then
    VIZ_SCRIPT="${REPO_ROOT}/kotekan/test_charts/visualize_3d_lobes_and_pipeline_light.py"
fi

python3 "${VIZ_SCRIPT}" \
    --window-dir "${WIN03_DIR}" \
    --out-dir "${FINAL_OUT_DIR}" \
    --frame-idx "${FRAME_IDX:-164}"

PERM_PLOTS="${REPO_ROOT}/test_charts/data/2min_pipeline/plots_light"
mkdir -p "${PERM_PLOTS}"
cp -v "${FINAL_OUT_DIR}"/*.png "${PERM_PLOTS}/" 2>/dev/null || true

echo "======================================================================"
echo " Successfully generated Light-Theme 3D Lobes & Spectrum Dashboard!"
echo " Saved to: ${FINAL_OUT_DIR}"
echo "======================================================================"
