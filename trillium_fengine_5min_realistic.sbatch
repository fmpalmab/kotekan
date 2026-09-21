#!/bin/bash
#SBATCH --job-name=charts_5min_realistic
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=slurm_charts_5min_realistic_%j.log

set -euo pipefail

echo "======================================================================"
echo " CHARTS 5-Minute Realistic Baseband & AstronCorrelator Production"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo " Scratch Dir    : ${SLURM_TMPDIR:-/tmp}"
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

# Resolve CUDA root and include directory for NVRTC and runtime compilers
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -n "${CUDA_ROOT:-}" ]; then
        export CUDA_HOME="${CUDA_ROOT}"
    elif [ -n "${EBROOTCUDA:-}" ]; then
        export CUDA_HOME="${EBROOTCUDA}"
    elif which nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(cd "$(dirname "$(which nvcc)")/.." && pwd)"
    fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_PATH="${CUDA_HOME}"
    export CUDA_ROOT="${CUDA_HOME}"
    export CPATH="${CUDA_HOME}/include:${CPATH:-}"
    export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
    echo " CUDA Toolkit Directory : ${CUDA_HOME}"
    echo " CUDA Include Directory : ${CUDA_HOME}/include"
fi

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
# 2. Build Kotekan Framework & CUDA Stages (if needed)
# ============================================================================
KOTEKAN_BIN="${REPO_ROOT}/build/kotekan/kotekan"
BUILD_KOTEKAN="${BUILD_KOTEKAN:-1}"

if [ "${BUILD_KOTEKAN}" = "1" ] || [ ! -x "${KOTEKAN_BIN}" ]; then
    echo "========================================="
    echo " 2. Building Kotekan Framework with CUDA"
    echo "========================================="
    mkdir -p build
    cd build
    cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90;80" -DUSE_NUMA=ON -DWERROR=OFF -DCMAKE_SKIP_RPATH=TRUE \
        ${CUDA_HOME:+-DCUDAToolkit_ROOT="${CUDA_HOME}"}
    make -j${SLURM_CPUS_PER_TASK:-8} kotekan
    cd ..
else
    echo "========================================="
    echo " 2. Kotekan Binary Found: Reusing Existing Build"
    echo "========================================="
fi

if [ ! -x "${KOTEKAN_BIN}" ]; then
    echo "[ERROR] Kotekan binary not found or not executable at ${KOTEKAN_BIN}"
    exit 1
fi
echo "Kotekan binary verified: ${KOTEKAN_BIN}"

# ============================================================================
# 3. Setup Simulation Parameters & Directories
# ============================================================================
# DURATION_S: default 300 s (5 min); can be set to 120 s (2 min) for lighter runs
DURATION_S="${DURATION_S:-300}"
BG_CADENCE_S="${BG_CADENCE_S:-2.0}"
EVENT_DENSE_S="${EVENT_DENSE_S:-1.0}"
ANTENNAS="${ANTENNAS:-64}"
NUM_FREQ="${NUM_FREQ:-336}"
NUM_EVENTS="${NUM_EVENTS:-3}"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

PERSISTENT_RFI_CHANNELS="${PERSISTENT_RFI_CHANNELS:-94,133,147}"
PERSISTENT_RFI_AMP="${PERSISTENT_RFI_AMP:-7.0}"

SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}/charts_windows_${SLURM_JOB_ID:-manual}"
PERM_OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/test_charts/data/5min_windows}"

WIN15_DIR="${SCRATCH_BASE}/win15UTC"
WIN03_DIR="${SCRATCH_BASE}/win03UTC"

mkdir -p "${WIN15_DIR}"
mkdir -p "${WIN03_DIR}"
mkdir -p "${PERM_OUTPUT_DIR}/win15UTC"
mkdir -p "${PERM_OUTPUT_DIR}/win03UTC"

echo "======================================================================"
echo " 3. Configuration:"
echo "    Duration           : ${DURATION_S} s ($(( DURATION_S / 60 )) minutes)"
echo "    Background Cadence : ${BG_CADENCE_S} s"
echo "    Event Dense Span   : ${EVENT_DENSE_S} s"
echo "    Antennas           : ${ANTENNAS}"
echo "    Frequency Channels : ${NUM_FREQ}"
echo "    Injected Events    : ${NUM_EVENTS} per window"
echo "    Persistent RFI     : channels [${PERSISTENT_RFI_CHANNELS}] (amp ${PERSISTENT_RFI_AMP} LSB)"
echo "    Workers            : ${WORKERS}"
echo "    Scratch Base       : ${SCRATCH_BASE}"
echo "    Persistent Storage : ${PERM_OUTPUT_DIR}"
echo "======================================================================"

# ============================================================================
# 4. Generate Window A: 15:00 UTC (Daytime, Sun UP at Carén)
# ============================================================================
echo "======================================================================"
echo " 4. Generating Window A: 15:00 UTC (Daytime, Sun UP)"
echo "======================================================================"
python3 test_charts/generate_realistic_5min_windows.py \
    --utc-hour 15 \
    --duration-s "${DURATION_S}" \
    --background-cadence-s "${BG_CADENCE_S}" \
    --event-dense-s "${EVENT_DENSE_S}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --num-events "${NUM_EVENTS}" \
    --persistent-rfi-channels "${PERSISTENT_RFI_CHANNELS}" \
    --persistent-rfi-amp "${PERSISTENT_RFI_AMP}" \
    --workers "${WORKERS}" \
    --out-dir "${WIN15_DIR}" \
    --file-name "win15UTC_${ANTENNAS}ant"

# ============================================================================
# 5. Generate Window B: 03:00 UTC (Nighttime, Sun BELOW Horizon)
# ============================================================================
echo "======================================================================"
echo " 5. Generating Window B: 03:00 UTC (Nighttime, Sun DOWN)"
echo "======================================================================"
python3 test_charts/generate_realistic_5min_windows.py \
    --utc-hour 3 \
    --duration-s "${DURATION_S}" \
    --background-cadence-s "${BG_CADENCE_S}" \
    --event-dense-s "${EVENT_DENSE_S}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --num-events "${NUM_EVENTS}" \
    --persistent-rfi-channels "${PERSISTENT_RFI_CHANNELS}" \
    --persistent-rfi-amp "${PERSISTENT_RFI_AMP}" \
    --workers "${WORKERS}" \
    --out-dir "${WIN03_DIR}" \
    --file-name "win03UTC_${ANTENNAS}ant"

# ============================================================================
# 6. Execute Kotekan Tensor Core Correlator for Window A (15:00 UTC)
# ============================================================================
echo "======================================================================"
echo " 6. Correlating Window A (15:00 UTC) via Kotekan AstronCorrelator"
echo "======================================================================"
python3 test_charts/run_5min_window_correlator.py \
    --window-dir "${WIN15_DIR}" \
    --file-name "win15UTC_${ANTENNAS}ant" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --inspect-frames "auto"

# ============================================================================
# 7. Execute Kotekan Tensor Core Correlator for Window B (03:00 UTC)
# ============================================================================
echo "======================================================================"
echo " 7. Correlating Window B (03:00 UTC) via Kotekan AstronCorrelator"
echo "======================================================================"
python3 test_charts/run_5min_window_correlator.py \
    --window-dir "${WIN03_DIR}" \
    --file-name "win03UTC_${ANTENNAS}ant" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --inspect-frames "auto"

# ============================================================================
# 8. Archive Lightweight Products (Metadata, Logs, Plots) to Persistent Storage
# ============================================================================
echo "======================================================================"
echo " 8. Archiving Diagnostic Products to ${PERM_OUTPUT_DIR}"
echo "======================================================================"
# Copy metadata, event tables, and diagnostic plots
cp -v "${WIN15_DIR}"/*_meta.h5 "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
cp -v "${WIN15_DIR}"/*_events.json "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
cp -rv "${WIN15_DIR}/plots" "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
cp -rv "${WIN15_DIR}/configs" "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true

cp -v "${WIN03_DIR}"/*_meta.h5 "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
cp -v "${WIN03_DIR}"/*_events.json "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
cp -rv "${WIN03_DIR}/plots" "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
cp -rv "${WIN03_DIR}/configs" "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true

# Optional: if user sets PERSIST_CORR_BIN=1, copy correlation binary dumps too
if [ "${PERSIST_CORR_BIN:-0}" = "1" ]; then
    echo " Copying correlation binary dumps to persistent storage..."
    cp -rv "${WIN15_DIR}/correlator" "${PERM_OUTPUT_DIR}/win15UTC/"
    cp -rv "${WIN03_DIR}/correlator" "${PERM_OUTPUT_DIR}/win03UTC/"
fi

# ============================================================================
# 9. Inventory & Verification Summary
# ============================================================================
echo "======================================================================"
echo " 9. INVENTORY & VERIFICATION SUMMARY"
echo "======================================================================"
echo "--- WINDOW A (15:00 UTC - DAYTIME WITH SUN) ---"
echo " Scratch dir size: $(du -sh "${WIN15_DIR}" | cut -f1)"
echo " Baseband frames : $(ls -1 "${WIN15_DIR}"/*.bin 2>/dev/null | wc -l) files"
echo " Correlator dumps: $(ls -1 "${WIN15_DIR}/correlator"/*.bin 2>/dev/null | wc -l) files"
echo " Diagnostic plots: $(ls -1 "${WIN15_DIR}/plots"/*.png 2>/dev/null | wc -l) files"
cat "${WIN15_DIR}"/*_events.json 2>/dev/null || true

echo ""
echo "--- WINDOW B (03:00 UTC - NIGHTTIME WITHOUT SUN) ---"
echo " Scratch dir size: $(du -sh "${WIN03_DIR}" | cut -f1)"
echo " Baseband frames : $(ls -1 "${WIN03_DIR}"/*.bin 2>/dev/null | wc -l) files"
echo " Correlator dumps: $(ls -1 "${WIN03_DIR}/correlator"/*.bin 2>/dev/null | wc -l) files"
echo " Diagnostic plots: $(ls -1 "${WIN03_DIR}/plots"/*.png 2>/dev/null | wc -l) files"
cat "${WIN03_DIR}"/*_events.json 2>/dev/null || true

echo "======================================================================"
echo " CHARTS 5-MINUTE REALISTIC PIPELINE COMPLETED SUCCESSFULLY!"
echo " Persistent diagnostic products: ${PERM_OUTPUT_DIR}"
echo " High-volume scratch datasets  : ${SCRATCH_BASE}"
echo "======================================================================"
