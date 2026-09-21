#!/bin/bash
#SBATCH --job-name=charts_2min_full_pipeline
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=slurm_charts_2min_full_pipeline_%j.log

set -euo pipefail

echo "======================================================================"
echo " CHARTS 2-Minute Full Pipeline: Baseband + Correlator + Beam Tracker"
echo " With 8 Random Events, CHARTS Thermal Noise, RFI, and Video Output"
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
module load ffmpeg 2>/dev/null || true

echo "=== Loaded Compilers & Environment ==="
gcc --version | head -n 1
which nvcc && nvcc --version | tail -n 2
cmake --version | head -n 1
python3 --version | head -n 1
nvidia-smi

# Resolve CUDA root
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
fi

# Setup scratch virtual environment
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

# Ensure required packages
pip install --no-index pyyaml jinja2 future h5py matplotlib numpy 2>/dev/null || pip install pyyaml jinja2 future h5py matplotlib numpy 2>/dev/null || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
echo "Working Directory: ${REPO_ROOT}"

# ============================================================================
# 2. Build Kotekan Framework & CUDA Stages
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
# 3. Simulation Parameters
# ============================================================================
# --- Timing & Decimation ---
DURATION_S="${DURATION_S:-120}"              # 2 minutes
BG_CADENCE_S="${BG_CADENCE_S:-2.0}"          # Background frame cadence
EVENT_DENSE_S="${EVENT_DENSE_S:-1.0}"        # Dense capture around events
EVENT_SPARSE_CADENCE_MS="${EVENT_SPARSE_CADENCE_MS:-100.0}" # Sparse capture cadence (ms)

# --- Array geometry ---
ANTENNAS="${ANTENNAS:-64}"
NUM_FREQ="${NUM_FREQ:-672}"
SAMPLES_PER_FRAME="${SAMPLES_PER_FRAME:-1536}"

# --- Events ---
NUM_EVENTS="${NUM_EVENTS:-8}"                # 8 random transient events

# --- Persistent RFI (frequency-selectable across full 300-501.6 MHz band) ---
# Specify by channel indices OR by frequency in MHz.
# If RFI_FREQS_MHZ is set, it overrides RFI_CHANNELS.
# Set to "none" to disable persistent RFI entirely.
# Channels 94 (328.2 MHz), 133 (339.9 MHz), 147 (344.1 MHz), 450 (435.0 MHz)
RFI_CHANNELS="${RFI_CHANNELS:-94,133,147,450}"   # Channel indices (300 + ch*0.3 MHz)
RFI_FREQS_MHZ="${RFI_FREQS_MHZ:-}"           # e.g. "328.2,339.9,344.1,435.0" (overrides channels)
RFI_AMP="${RFI_AMP:-7.0}"                    # Amplitude in LSB

# --- Beam tracker ---
MAX_BEAMS="${MAX_BEAMS:-8}"                  # 8 formed beams
INTEGRATION_SPECTRA="${INTEGRATION_SPECTRA:-320}"

# --- Video ---
VIDEO_DURATION_S="${VIDEO_DURATION_S:-${DURATION_S:-120}}"   # Full 2-minute window videos
VIDEO_FPS="${VIDEO_FPS:-10}"

# --- Workers ---
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

# --- Beam targets (RA/Dec with object names for 8 distinct celestial sources) ---
# Window A (15:00 UTC, LST ~5.575h, site lat -33.42°): 8 distinct celestial sources
BEAM_TARGETS_WIN15="${BEAM_TARGETS_WIN15:-Crab Pulsar:83.633,22.014;PSR J0437-4715:69.316,-47.252;Pictor A:79.957,-45.779;Orion M42:83.822,-5.391;Canopus:95.988,-52.696;Sirius:101.287,-16.716;PSR B0656+14:104.951,14.238;LMC J0537:84.440,-69.170}"

# Window B (03:00 UTC, LST ~17.5h, site lat -33.42°): 8 distinct celestial sources
BEAM_TARGETS_WIN03="${BEAM_TARGETS_WIN03:-Sgr A* (GC):266.417,-29.008;Centaurus A:201.365,-43.019;PSR B1642-03:251.272,-3.371;Sco X-1:244.979,-15.640;PSR B1749-28:268.064,-28.106;PSR B1818-04:275.318,-4.292;M87 Virgo A:187.706,12.391;PSR B1937+21:294.911,21.583}"

# --- Directories ---
# Determine scratch location:
# 1. Explicit user override ($SCRATCH_DIR)
# 2. Local SSD/NVMe $SLURM_TMPDIR (only if NOT on /dev/shm tmpfs, which consumes job RAM)
# 3. Lustre $SCRATCH (Trillium / Compute Canada high-speed persistent scratch, e.g. /scratch/ferpb)
# 4. Fallback to repo-local directory
if [ -n "${SCRATCH_DIR:-}" ]; then
    SCRATCH_BASE="${SCRATCH_DIR}/charts_2min_${SLURM_JOB_ID:-manual}"
elif [ -n "${SLURM_TMPDIR:-}" ] && [[ "${SLURM_TMPDIR}" != /dev/shm* ]]; then
    SCRATCH_BASE="${SLURM_TMPDIR}/charts_2min_${SLURM_JOB_ID:-manual}"
elif [ -n "${SCRATCH:-}" ]; then
    SCRATCH_BASE="${SCRATCH}/kotekan_scratch/charts_2min_${SLURM_JOB_ID:-manual}"
else
    SCRATCH_BASE="${REPO_ROOT}/scratch_charts_2min_${SLURM_JOB_ID:-manual}"
fi

PERM_OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/test_charts/data/2min_pipeline}"

WIN15_DIR="${SCRATCH_BASE}/win15UTC"
WIN03_DIR="${SCRATCH_BASE}/win03UTC"

mkdir -p "${WIN15_DIR}"
mkdir -p "${WIN03_DIR}"
mkdir -p "${PERM_OUTPUT_DIR}/win15UTC"
mkdir -p "${PERM_OUTPUT_DIR}/win03UTC"

# Resolve RFI channel specification
RFI_CHANNELS_ARG=""
RFI_FREQS_ARG=""
if [ -n "${RFI_FREQS_MHZ}" ] && [ "${RFI_FREQS_MHZ}" != "none" ]; then
    RFI_FREQS_ARG="--persistent-rfi-freqs ${RFI_FREQS_MHZ}"
    RFI_DESC="freqs [${RFI_FREQS_MHZ}] MHz"
elif [ "${RFI_CHANNELS}" != "none" ]; then
    RFI_CHANNELS_ARG="--persistent-rfi-channels ${RFI_CHANNELS}"
    RFI_DESC="channels [${RFI_CHANNELS}]"
else
    RFI_CHANNELS_ARG="--persistent-rfi-channels none"
    RFI_DESC="disabled"
fi

echo "======================================================================"
echo " 3. Configuration:"
echo "    Duration           : ${DURATION_S} s ($(( DURATION_S / 60 )) min)"
echo "    Background Cadence : ${BG_CADENCE_S} s"
echo "    Event Dense Span   : ${EVENT_DENSE_S} s"
echo "    Event Sparse Step  : ${EVENT_SPARSE_CADENCE_MS} ms"
echo "    Antennas           : ${ANTENNAS}"
echo "    Frequency Channels : ${NUM_FREQ}"
echo "    Samples/Frame      : ${SAMPLES_PER_FRAME}"
echo "    Injected Events    : ${NUM_EVENTS} per window"
echo "    Persistent RFI     : ${RFI_DESC} (amp ${RFI_AMP} LSB)"
echo "    Max Beams          : ${MAX_BEAMS}"
echo "    Window A Targets   : ${BEAM_TARGETS_WIN15}"
echo "    Window B Targets   : ${BEAM_TARGETS_WIN03}"
echo "    Video Duration     : ${VIDEO_DURATION_S} s @ ${VIDEO_FPS} fps (full window)"
echo "    Workers            : ${WORKERS}"
echo "    Scratch Base       : ${SCRATCH_BASE}"
echo "    Persistent Storage : ${PERM_OUTPUT_DIR}"
if [[ "${SCRATCH_BASE}" == /dev/shm* ]]; then
    echo "    [WARNING] Scratch Base is on /dev/shm (RAM tmpfs)! May trigger OOM if data exceeds RAM."
fi
echo "======================================================================"

# ============================================================================
# PART I: WINDOW A (15:00 UTC - Daytime, Sun UP at Carén)
# ============================================================================

echo "======================================================================"
echo " 4. Generating Window A: 15:00 UTC (Daytime, Sun UP)"
echo "======================================================================"
python3 test_charts/generate_realistic_5min_windows.py \
    --utc-hour 15 \
    --duration-s "${DURATION_S}" \
    --background-cadence-s "${BG_CADENCE_S}" \
    --event-dense-s "${EVENT_DENSE_S}" \
    --event-sparse-cadence-ms "${EVENT_SPARSE_CADENCE_MS}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --num-events "${NUM_EVENTS}" \
    ${RFI_CHANNELS_ARG} \
    ${RFI_FREQS_ARG} \
    --persistent-rfi-amp "${RFI_AMP}" \
    --workers "${WORKERS}" \
    --out-dir "${WIN15_DIR}" \
    --file-name "win15UTC_${ANTENNAS}ant"

echo "======================================================================"
echo " 5. Correlating Window A (15:00 UTC) via Kotekan AstronCorrelator"
echo "======================================================================"
python3 test_charts/run_5min_window_correlator.py \
    --window-dir "${WIN15_DIR}" \
    --file-name "win15UTC_${ANTENNAS}ant" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --inspect-frames "auto"

echo "======================================================================"
echo " 6. Beam Tracker: Window A (15:00 UTC) — ${MAX_BEAMS} Beams"
echo "======================================================================"
python3 test_charts/run_beam_tracker_replay.py \
    --window-dir "${WIN15_DIR}" \
    --file-name "win15UTC_${ANTENNAS}ant" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --max-beams "${MAX_BEAMS}" \
    --integration-spectra "${INTEGRATION_SPECTRA}" \
    --beam-targets "${BEAM_TARGETS_WIN15}" \
    --initial-lst-hours 5.575

echo "======================================================================"
echo " 7. Generating Videos for Window A (15:00 UTC)"
echo "======================================================================"
echo "--- 7a. Baseband Frequency Power Line Spectrum Video ---"
python3 test_charts/generate_pipeline_videos.py baseband \
    --window-dir "${WIN15_DIR}" \
    --base-name "win15UTC_${ANTENNAS}ant" \
    --output "${WIN15_DIR}/videos/baseband_spectrogram_win15UTC.mp4" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --duration-s "${VIDEO_DURATION_S}" \
    --fps "${VIDEO_FPS}" \
    2>&1 || echo "[WARNING] Baseband video generation failed for win15UTC"

echo "--- 7b. Correlator Dual-Channel Matrix Video ---"
python3 test_charts/generate_pipeline_videos.py correlator \
    --corr-dir "${WIN15_DIR}/correlator" \
    --corr-name "corr_win15UTC_${ANTENNAS}ant" \
    --output "${WIN15_DIR}/videos/correlator_matrix_win15UTC.mp4" \
    --num-elements "${ANTENNAS}" \
    --num-channels "${NUM_FREQ}" \
    --duration-s "${VIDEO_DURATION_S}" \
    --fps "${VIDEO_FPS}" \
    --freq-channels "auto" \
    2>&1 || echo "[WARNING] Correlator video generation failed for win15UTC"

echo "--- 7c. Beam Tracker Output Video (All Beams + Object Legend) ---"
python3 test_charts/generate_pipeline_videos.py tracker \
    --tracker-dir "${WIN15_DIR}/tracker" \
    --tracker-name "beams_win15UTC_${ANTENNAS}ant" \
    --output "${WIN15_DIR}/videos/beam_tracker_win15UTC.mp4" \
    --samples-per-data-set "${SAMPLES_PER_FRAME}" \
    --num-freq "${NUM_FREQ}" \
    --max-beams "${MAX_BEAMS}" \
    --duration-s "${VIDEO_DURATION_S}" \
    --fps "${VIDEO_FPS}" \
    --beam-targets "${BEAM_TARGETS_WIN15}" \
    2>&1 || echo "[WARNING] Tracker video generation failed for win15UTC"

echo "======================================================================"
echo " 8. Archiving Window A Products to ${PERM_OUTPUT_DIR}/win15UTC"
echo "======================================================================"
cp -v "${WIN15_DIR}"/*_meta.h5 "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
cp -v "${WIN15_DIR}"/*_events.json "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
cp -rv "${WIN15_DIR}/plots" "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
cp -rv "${WIN15_DIR}/configs" "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
cp -rv "${WIN15_DIR}/videos" "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true

if [ "${PERSIST_CORR_BIN:-0}" = "1" ]; then
    echo " Copying correlation binary dumps for win15UTC..."
    cp -rv "${WIN15_DIR}/correlator" "${PERM_OUTPUT_DIR}/win15UTC/"
fi
if [ "${PERSIST_TRACKER_BIN:-0}" = "1" ]; then
    echo " Copying beam tracker binary dumps for win15UTC..."
    cp -rv "${WIN15_DIR}/tracker" "${PERM_OUTPUT_DIR}/win15UTC/"
fi
if [ "${PERSIST_BASEBAND_BIN:-0}" = "1" ]; then
    echo " Copying baseband binary dumps for win15UTC..."
    cp -v "${WIN15_DIR}"/win15UTC_${ANTENNAS}ant_*.bin "${PERM_OUTPUT_DIR}/win15UTC/" 2>/dev/null || true
fi

# Optional cleanup of Window A raw baseband to save disk space
if [ "${CLEANUP_BASEBAND:-0}" = "1" ]; then
    echo " Cleaning up raw baseband frames for win15UTC to free scratch space..."
    rm -f "${WIN15_DIR}"/win15UTC_${ANTENNAS}ant_*.bin
fi

# ============================================================================
# PART II: WINDOW B (03:00 UTC - Nighttime, Sun BELOW Horizon)
# ============================================================================

echo "======================================================================"
echo " 9. Generating Window B: 03:00 UTC (Nighttime, Sun DOWN)"
echo "======================================================================"
python3 test_charts/generate_realistic_5min_windows.py \
    --utc-hour 3 \
    --duration-s "${DURATION_S}" \
    --background-cadence-s "${BG_CADENCE_S}" \
    --event-dense-s "${EVENT_DENSE_S}" \
    --event-sparse-cadence-ms "${EVENT_SPARSE_CADENCE_MS}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --num-events "${NUM_EVENTS}" \
    ${RFI_CHANNELS_ARG} \
    ${RFI_FREQS_ARG} \
    --persistent-rfi-amp "${RFI_AMP}" \
    --workers "${WORKERS}" \
    --out-dir "${WIN03_DIR}" \
    --file-name "win03UTC_${ANTENNAS}ant"

echo "======================================================================"
echo " 10. Correlating Window B (03:00 UTC) via Kotekan AstronCorrelator"
echo "======================================================================"
python3 test_charts/run_5min_window_correlator.py \
    --window-dir "${WIN03_DIR}" \
    --file-name "win03UTC_${ANTENNAS}ant" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --inspect-frames "auto"

echo "======================================================================"
echo " 11. Beam Tracker: Window B (03:00 UTC) — ${MAX_BEAMS} Beams"
echo "======================================================================"
python3 test_charts/run_beam_tracker_replay.py \
    --window-dir "${WIN03_DIR}" \
    --file-name "win03UTC_${ANTENNAS}ant" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --max-beams "${MAX_BEAMS}" \
    --integration-spectra "${INTEGRATION_SPECTRA}" \
    --beam-targets "${BEAM_TARGETS_WIN03}" \
    --initial-lst-hours 17.5

echo "======================================================================"
echo " 12. Generating Videos for Window B (03:00 UTC)"
echo "======================================================================"
echo "--- 12a. Baseband Frequency Power Line Spectrum Video ---"
python3 test_charts/generate_pipeline_videos.py baseband \
    --window-dir "${WIN03_DIR}" \
    --base-name "win03UTC_${ANTENNAS}ant" \
    --output "${WIN03_DIR}/videos/baseband_spectrogram_win03UTC.mp4" \
    --antennas "${ANTENNAS}" \
    --num-freq "${NUM_FREQ}" \
    --samples-per-frame "${SAMPLES_PER_FRAME}" \
    --duration-s "${VIDEO_DURATION_S}" \
    --fps "${VIDEO_FPS}" \
    2>&1 || echo "[WARNING] Baseband video generation failed for win03UTC"

echo "--- 12b. Correlator Dual-Channel Matrix Video ---"
python3 test_charts/generate_pipeline_videos.py correlator \
    --corr-dir "${WIN03_DIR}/correlator" \
    --corr-name "corr_win03UTC_${ANTENNAS}ant" \
    --output "${WIN03_DIR}/videos/correlator_matrix_win03UTC.mp4" \
    --num-elements "${ANTENNAS}" \
    --num-channels "${NUM_FREQ}" \
    --duration-s "${VIDEO_DURATION_S}" \
    --fps "${VIDEO_FPS}" \
    --freq-channels "auto" \
    2>&1 || echo "[WARNING] Correlator video generation failed for win03UTC"

echo "--- 12c. Beam Tracker Output Video (All Beams + Object Legend) ---"
python3 test_charts/generate_pipeline_videos.py tracker \
    --tracker-dir "${WIN03_DIR}/tracker" \
    --tracker-name "beams_win03UTC_${ANTENNAS}ant" \
    --output "${WIN03_DIR}/videos/beam_tracker_win03UTC.mp4" \
    --samples-per-data-set "${SAMPLES_PER_FRAME}" \
    --num-freq "${NUM_FREQ}" \
    --max-beams "${MAX_BEAMS}" \
    --duration-s "${VIDEO_DURATION_S}" \
    --fps "${VIDEO_FPS}" \
    --beam-targets "${BEAM_TARGETS_WIN03}" \
    2>&1 || echo "[WARNING] Tracker video generation failed for win03UTC"

echo "======================================================================"
echo " 13. Archiving Window B Products to ${PERM_OUTPUT_DIR}/win03UTC"
echo "======================================================================"
cp -v "${WIN03_DIR}"/*_meta.h5 "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
cp -v "${WIN03_DIR}"/*_events.json "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
cp -rv "${WIN03_DIR}/plots" "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
cp -rv "${WIN03_DIR}/configs" "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
cp -rv "${WIN03_DIR}/videos" "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true

if [ "${PERSIST_CORR_BIN:-0}" = "1" ]; then
    echo " Copying correlation binary dumps for win03UTC..."
    cp -rv "${WIN03_DIR}/correlator" "${PERM_OUTPUT_DIR}/win03UTC/"
fi
if [ "${PERSIST_TRACKER_BIN:-0}" = "1" ]; then
    echo " Copying beam tracker binary dumps for win03UTC..."
    cp -rv "${WIN03_DIR}/tracker" "${PERM_OUTPUT_DIR}/win03UTC/"
fi
if [ "${PERSIST_BASEBAND_BIN:-0}" = "1" ]; then
    echo " Copying baseband binary dumps for win03UTC..."
    cp -v "${WIN03_DIR}"/win03UTC_${ANTENNAS}ant_*.bin "${PERM_OUTPUT_DIR}/win03UTC/" 2>/dev/null || true
fi

# Optional cleanup of Window B raw baseband to save disk space
if [ "${CLEANUP_BASEBAND:-0}" = "1" ]; then
    echo " Cleaning up raw baseband frames for win03UTC to free scratch space..."
    rm -f "${WIN03_DIR}"/win03UTC_${ANTENNAS}ant_*.bin
fi

# ============================================================================
# 13. Inventory & Verification Summary
# ============================================================================
echo "======================================================================"
echo " 13. INVENTORY & VERIFICATION SUMMARY"
echo "======================================================================"

for WIN_LABEL in "win15UTC" "win03UTC"; do
    if [ "${WIN_LABEL}" = "win15UTC" ]; then
        WDIR="${WIN15_DIR}"
        WDESC="15:00 UTC - DAYTIME WITH SUN"
    else
        WDIR="${WIN03_DIR}"
        WDESC="03:00 UTC - NIGHTTIME WITHOUT SUN"
    fi

    echo ""
    echo "--- WINDOW: ${WDESC} ---"
    echo " Scratch dir size    : $(du -sh "${WDIR}" 2>/dev/null | cut -f1 || echo 'N/A')"
    echo " Baseband frames     : $(ls -1 "${WDIR}"/*.bin 2>/dev/null | wc -l) files"
    echo " Correlator dumps    : $(ls -1 "${WDIR}/correlator"/*.bin 2>/dev/null | wc -l) files"
    echo " Tracker dumps       : $(ls -1 "${WDIR}/tracker"/*.bin 2>/dev/null | wc -l) files"
    echo " Diagnostic plots    : $(ls -1 "${WDIR}/plots"/*.png 2>/dev/null | wc -l) files"
    echo " Videos              : $(ls -1 "${WDIR}/videos"/*.mp4 2>/dev/null | wc -l) files"
    echo ""
    echo " Event table:"
    cat "${WDIR}"/*_events.json 2>/dev/null || echo "  (no events file found)"
    echo ""
done

echo "======================================================================"
echo " CHARTS 2-MINUTE FULL PIPELINE COMPLETED SUCCESSFULLY!"
echo " Persistent products : ${PERM_OUTPUT_DIR}"
echo " Scratch datasets    : ${SCRATCH_BASE}"
echo "======================================================================"
