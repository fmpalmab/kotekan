#!/bin/bash
#SBATCH --job-name=kotekan_charts_test
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=kotekan_test_%j.log

set -euo pipefail

echo "========================================="
echo " Loading Trillium Modules"
echo "========================================="
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 2>/dev/null || module load gcc || true
module load cuda/13.3 2>/dev/null || module load cuda/12.6 2>/dev/null || module load cuda || true
module load cmake 2>/dev/null || true
module load boost 2>/dev/null || true
module load python/3.10 2>/dev/null || module load python || true

echo "=== Loaded Compilers & Environment ==="
gcc --version | head -n 1
which nvcc && nvcc --version | tail -n 2
cmake --version | head -n 1
python3 --version | head -n 1
nvidia-smi

echo "========================================="
echo " 1. Building Kotekan Framework & Tests"
echo "========================================="
# Clean and build
rm -rf build
mkdir -p build
cd build

# Configure CMake with CUDA, NUMA, and Hopper sm_90
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90" -DUSE_NUMA=ON -DWERROR=OFF -DCMAKE_SKIP_RPATH=TRUE

# Compile Kotekan and test binaries using available CPU cores
make -j${SLURM_CPUS_PER_TASK}
cd ..

echo "========================================="
echo " 2. Building Standalone CUDA Benchmark & Test Suite"
echo "========================================="
# Compiles directly for the host GPU node with fast math, include directories, and OpenMP
nvcc -std=c++17 -O3 --use_fast_math \
    -arch=native \
    -Xptxas -O3 \
    -Xcompiler -fopenmp \
    -I lib/cuda \
    -I lib/metadata \
    -I lib/core \
    -I lib/utils \
    -I lib/stages \
    -I lib/dpdk \
    -I lib/ \
    -I external/json \
    -I external/fmt \
    -I external/gputils/src_lib \
    -I external/gputils/src_include \
    test_charts/test_beam_tracker_suite.cpp \
    lib/cuda/cudaBeamTrackerV5.cu \
    lib/core/kotekanLogging.cpp \
    lib/core/errors.c \
    -o test_suite_tracker

nvcc -std=c++17 -O3 --use_fast_math \
    -arch=native \
    -Xptxas -O3 \
    -I lib/cuda \
    -I lib/metadata \
    -I lib/core \
    -I lib/utils \
    -I lib/ \
    -I external/json \
    -I external/fmt \
    -I external/gputils/src_lib \
    -I external/gputils/src_include \
    test_charts/benchmark_kotekan_cuda_tracker_v5.cpp \
    lib/cuda/cudaBeamTrackerV5.cu \
    lib/core/kotekanLogging.cpp \
    lib/core/errors.c \
    -o benchmark_tracker

echo "========================================="
echo " 3. Running Comprehensive Test Suite (64, 128, 256 Antennas)"
echo "========================================="
./test_suite_tracker

echo "========================================="
echo " 4. Running Standalone Complex Voltage Benchmark"
echo "========================================="
./benchmark_tracker

echo "========================================="
echo " 5. Running Full Kotekan Pipeline Test (64 Antennas)"
echo "========================================="
mkdir -p /tmp/test_charts_tracker
rm -f /tmp/test_charts_tracker/*.bin

# Execute Kotekan (CVMFS compiler module handles runtime library paths)
LD_LIBRARY_PATH="$PWD/build/external/n2k:$LD_LIBRARY_PATH" ./build/kotekan/kotekan --config test_charts/config/64antennas_beam_tracker_test.yaml

echo "========================================="
echo " 6. Inspecting Generated Complex Voltage Output"
echo "========================================="
python3 test_charts/inspect_tracker_output.py

echo "========================================="
echo " ALL TRILLIUM BATCH TESTS COMPLETED SUCCESSFULLY"
echo "========================================="

