#!/bin/bash
#SBATCH --job-name=kotekan_charts_test
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=kotekan_test_%j.log

echo "========================================="
echo " Loading Trillium Modules"
echo "========================================="
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 2>/dev/null || module load gcc || true
module load cuda/12.6 2>/dev/null || module load cuda || true
module load cmake 2>/dev/null || true
module load boost 2>/dev/null || true

echo "=== Loaded Compilers ==="
gcc --version | head -n 1
which nvcc && nvcc --version | tail -n 2
cmake --version | head -n 1

echo "========================================="
echo " 1. Building Kotekan Framework"
echo "========================================="
# Clean old build
rm -rf build
mkdir -p build
cd build

# Configure CMake with CUDA, NUMA, and Hopper sm_90
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90" -DUSE_NUMA=ON -DWERROR=OFF -DCMAKE_SKIP_RPATH=TRUE

# Compile Kotekan using available CPU cores
make -j${SLURM_CPUS_PER_TASK}
cd ..

echo "========================================="
echo " 2. Building Standalone CUDA Benchmark"
echo "========================================="
# Compiles directly for the host GPU node with fast math, include directories, and logging symbols
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
echo " 3. Running Standalone Benchmark"
echo "========================================="
./benchmark_tracker

echo "========================================="
echo " 4. Running Full Kotekan Pipeline Test"
echo "========================================="
mkdir -p /tmp/test_charts_tracker

# Execute Kotekan (CVMFS compiler module handles runtime library paths)
# We add build/external/n2k to LD_LIBRARY_PATH because CMAKE_SKIP_RPATH removed it
LD_LIBRARY_PATH="$PWD/build/external/n2k:$LD_LIBRARY_PATH" ./build/kotekan/kotekan --config test_charts/config/64antennas_beam_tracker_test.yaml
