#!/bin/bash
#SBATCH --job-name=direct_vs_v5_tracker
#SBATCH --output=slurm_direct_vs_v5_%j.out
#SBATCH --error=slurm_direct_vs_v5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

set -euo pipefail

echo "======================================================================"
echo " Starting Head-to-Head Benchmark: Direct Beam Tracker vs Beam Tracker V5"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-Not set (default 0)}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load environment modules (Compute Canada / Trillium / standard HPC cluster)
if command -v module >/dev/null 2>&1; then
    module load cuda/12.6 2>/dev/null || module load cuda/13.3 2>/dev/null || module load cuda || true
    module load gcc/12.3 2>/dev/null || module load gcc || true
    module load cmake 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python || true
    module load scipy-stack 2>/dev/null || true
    module load hdf5 2>/dev/null || true
fi

# Print GPU architecture & specifications
nvidia-smi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

echo "Repository Root : ${REPO_ROOT}"
echo "Build Directory : ${BUILD_DIR}"
cd "${REPO_ROOT}"

# 2. Build Kotekan and Comparison Benchmark target
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo ">>> Configuring CMake with CUDA enabled..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;89;90;100;120" \
    -DUSE_NUMA=ON \
    -DWERROR=OFF

echo ">>> Compiling Head-to-Head Comparative Benchmark..."
make -j${SLURM_CPUS_PER_TASK:-8} benchmark_tracker_v5_vs_direct kotekan || true

# Fallback standalone compilation if needed
if [ ! -f "test_charts/benchmark_tracker_v5_vs_direct" ] && [ ! -f "${REPO_ROOT}/benchmark_tracker_v5_vs_direct" ]; then
    echo ">>> Building standalone benchmark_tracker_v5_vs_direct via NVCC fallback..."
    cd "${REPO_ROOT}"
    nvcc -std=c++17 -O3 --use_fast_math \
        -arch=native \
        -Xcompiler -fopenmp \
        -I lib/cuda -I lib/metadata -I lib/core -I lib/utils -I lib/stages -I lib/ \
        -I external/json -I external/fmt \
        test_charts/benchmark_tracker_v5_vs_direct.cpp \
        lib/cuda/cudaBeamTrackerV5.cu \
        lib/cuda/cudaDirectBeamTracker.cu \
        lib/core/kotekanLogging.cpp \
        lib/core/errors.c \
        -o benchmark_tracker_v5_vs_direct
    BENCHMARK_BIN="${REPO_ROOT}/benchmark_tracker_v5_vs_direct"
else
    BENCHMARK_BIN="${BUILD_DIR}/test_charts/benchmark_tracker_v5_vs_direct"
fi

cd "${REPO_ROOT}"

# 3. Run Head-to-Head Comparative Benchmark
echo "======================================================================"
echo " Running Comparative Benchmark Matrix (32, 64, 128, 256 Antennas x 1, 2, 4 Beams)"
echo "======================================================================"
"${BENCHMARK_BIN}"

# 4. Generate Analysis and Markdown Summary
if [ -f "tracker_benchmark_comparison.json" ]; then
    echo "======================================================================"
    echo " Generating Speedup Analysis & Markdown Summary"
    echo "======================================================================"
    python3 test_charts/compare_tracker_benchmarks.py tracker_benchmark_comparison.json \
        --md-out "tracker_benchmark_report_${SLURM_JOB_ID:-manual}.md"
fi

echo "======================================================================"
echo " Direct vs V5 Comparative Benchmark Completed Successfully!"
echo "======================================================================"
