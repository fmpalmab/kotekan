#!/bin/bash
#SBATCH --job-name=charts_real_data_test
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=charts_real_data_%j.log

set -euo pipefail

echo "======================================================================"
echo " Starting CHARTS Beam Tracker Real Data Test on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Job ID         : ${SLURM_JOB_ID}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load Trillium Modules
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 2>/dev/null || module load gcc || true
module load cuda/13.3 2>/dev/null || module load cuda/12.6 2>/dev/null || module load cuda || true
module load cmake 2>/dev/null || true
module load python/3.10 2>/dev/null || module load python || true
module load scipy-stack 2>/dev/null || true
module load hdf5 2>/dev/null || true

# Verify Python packages
if ! python3 -c "import numpy, h5py, matplotlib" 2>/dev/null; then
    echo "Python scientific stack missing in module, installing locally..."
    pip install --user --no-cache-dir numpy h5py matplotlib || true
fi

# 2. Paths & Environment
REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/ferpb/kotekan}"
cd "${REPO_ROOT}"

DATA_DIR="/scratch/ferpb/260816T013722Z_CHARTS_hdf5"
SINGLE_CHUNK="${DATA_DIR}/bb_f0000_0084_chunk00000.h5"
VIRTUAL_DATASET="${DATA_DIR}/baseband_virtual.h5"
BENCHMARK_SCRIPT="${REPO_ROOT}/test_charts/benchmark_beam_tracker_hdf5.py"

echo "Working Dir     : ${REPO_ROOT}"
echo "Data Directory  : ${DATA_DIR}"
echo "Target Single   : ${SINGLE_CHUNK}"
echo "Target Virtual  : ${VIRTUAL_DATASET}"
echo "Benchmark Script: ${BENCHMARK_SCRIPT}"

# 3. Test on 1 Dump of Single Frequency Chunk (f0000 to f0084)
echo "======================================================================"
echo " 1. Testing & Benchmarking on 1 Dump of Single Chunk (${SINGLE_CHUNK})"
echo "======================================================================"
if [ -f "${SINGLE_CHUNK}" ]; then
    python3 "${BENCHMARK_SCRIPT}" "${SINGLE_CHUNK}" \
        --n-time 15360 \
        --out-plot "real_data_single_chunk_benchmark_${SLURM_JOB_ID}.png"
else
    echo "[WARNING] ${SINGLE_CHUNK} not found! Skipping single chunk test."
fi

# 4. Test on 1 Dump of Full Virtual Dataset (All Frequencies combined)
echo "======================================================================"
echo " 2. Testing & Benchmarking on 1 Dump of Virtual Dataset (${VIRTUAL_DATASET})"
echo "======================================================================"
if [ -f "${VIRTUAL_DATASET}" ]; then
    python3 "${BENCHMARK_SCRIPT}" "${VIRTUAL_DATASET}" \
        --n-time 15360 \
        --n-freq-max 336 \
        --out-plot "real_data_virtual_dataset_benchmark_${SLURM_JOB_ID}.png"
else
    echo "[WARNING] ${VIRTUAL_DATASET} not found! Skipping virtual dataset test."
fi

echo "======================================================================"
echo " CHARTS Real Data Beam Tracker Test Finished Successfully!"
echo " Results and Diagnostic Plots Saved in: $PWD"
echo "======================================================================"
