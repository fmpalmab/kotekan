#!/bin/bash
#SBATCH --job-name=charts_64ant_fengine_corr
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=slurm_charts_64ant_corr_%j.log

set -euo pipefail

echo "======================================================================"
echo " CHARTS 64-Antenna Baseband & AstronCorrelator Production on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo " Output Base Dir: /project/def-vanderli/ferpb"
echo "======================================================================"

# ============================================================================
# 1. Load Trillium Environment Modules
# ============================================================================
echo "========================================="
echo " 1. Loading Trillium Modules"
echo "========================================="
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 2>/dev/null || module load gcc || true
module load cuda/12.6 2>/dev/null || module load cuda || true
module load cmake 2>/dev/null || true
module load boost 2>/dev/null || true
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python || true
module load scipy-stack 2>/dev/null || true
module load hdf5 2>/dev/null || true

echo "=== Loaded Compilers & Environment ==="
gcc --version | head -n 1
which nvcc && nvcc --version | tail -n 2
cmake --version | head -n 1
python3 --version | head -n 1
nvidia-smi

# Setup high-speed scratch virtual environment inheriting CVMFS modules (numpy, scipy, matplotlib, h5py)
VENV_DIR="${SLURM_TMPDIR:-/tmp}/kotekan_pyenv_${SLURM_JOB_ID:-manual}"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}" || true
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

# Prevent matplotlib font-cache lock issues
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

# Ensure required packages exist
pip install --no-index pyyaml jinja2 future h5py matplotlib numpy 2>/dev/null || pip install pyyaml jinja2 future h5py matplotlib numpy 2>/dev/null || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
echo "Working Directory: ${REPO_ROOT}"

# ============================================================================
# 2. Build Kotekan Framework & CUDA Stages
# ============================================================================
echo "========================================="
echo " 2. Building Kotekan Framework with CUDA"
echo "========================================="
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90;80" -DUSE_NUMA=ON -DWERROR=OFF -DCMAKE_SKIP_RPATH=TRUE
make -j${SLURM_CPUS_PER_TASK:-8} kotekan
cd ..

KOTEKAN_BIN="${REPO_ROOT}/build/kotekan/kotekan"
if [ ! -x "${KOTEKAN_BIN}" ]; then
    echo "[ERROR] Kotekan binary not found or not executable at ${KOTEKAN_BIN}"
    exit 1
fi
echo "Kotekan binary verified: ${KOTEKAN_BIN}"

# ============================================================================
# 3. Setup Relative Output Directory
# ============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-./dumps_charts_64ant}"
echo "========================================="
echo " 3. Setting Up Output Directory: ${OUTPUT_DIR}"
echo "========================================="
mkdir -p "${OUTPUT_DIR}/baseband"
mkdir -p "${OUTPUT_DIR}/correlator"
mkdir -p "${OUTPUT_DIR}/plots"
mkdir -p "${OUTPUT_DIR}/configs"

# ============================================================================
# 4. Execute End-to-End Simulation & AstronCorrelator Pipeline
# ============================================================================
SCENARIO="${SCENARIO:-all}"
CATEGORY="${CATEGORY:-all}"

echo " Running with Scenario: ${SCENARIO}, Category: ${CATEGORY}"

python3 test_charts/run_charts_fengine_correlator_pipeline.py \
    --output-dir "${OUTPUT_DIR}" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --scenario "${SCENARIO}" \
    --category "${CATEGORY}" \
    --num-frames 1 \
    --samples-per-frame 1536 \
    --antennas 64 \
    --num-freq 336

# ============================================================================
# 5. Generate Master PNG Dashboard (64 Antennas & All Correlation Matrices)
# ============================================================================
echo "========================================="
echo " Generating Master PNG Correlation Dashboard..."
echo "========================================="
python3 test_charts/visualize_charts_64ant_correlations.py \
    --dumps-dir "${OUTPUT_DIR}/correlator" \
    --output "${OUTPUT_DIR}/plots/charts_64ant_correlations_overview.png"

# ============================================================================
# 6. Inventory & Checksum Verification
# ============================================================================
echo "========================================="
echo " 6. Inventory of Generated Datasets in ${OUTPUT_DIR}"
echo "========================================="
echo "--- BASEBAND DUMPS (HDF5 & BIN) ---"
ls -lh "${OUTPUT_DIR}/baseband"

echo "--- CORRELATOR DUMPS (BIN) ---"
ls -lh "${OUTPUT_DIR}/correlator"

echo "--- DIAGNOSTIC PLOTS (PNG) ---"
ls -lh "${OUTPUT_DIR}/plots"

echo "======================================================================"
echo " CHARTS 64-ANTENNA SIMULATION & CORRELATION COMPLETED SUCCESSFULLY!"
echo " Local output directory: ${OUTPUT_DIR}"
echo " To transfer/move to permanent project directory when ready:"
echo "   rsync -av ${OUTPUT_DIR}/ /project/def-vanderli/ferpb/"
echo "======================================================================"
