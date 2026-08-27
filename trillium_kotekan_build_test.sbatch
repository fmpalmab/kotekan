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
# Kotekan requires C++17 (GCC 9+ or 11+) and recent CUDA
# Adjust these module names based on what is available on Trillium (`module avail gcc cuda`)
module purge
module load gcc/11.3.0       # Or gcc/12, whatever is standard non-LTS for C++17
module load cuda/13.3        # Recent CUDA
module load cmake/3.24.2     # CMake > 3.8.2 required

echo "========================================="
echo " 1. Building Kotekan Framework"
echo "========================================="
# We build Kotekan out-of-source in a build directory
mkdir -p build
cd build

# Configure CMake with CUDA, H100/Blackwell architectures, and NUMA enabled
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90;120" -DUSE_NUMA=ON -DWERROR=OFF

# Compile Kotekan using available CPU cores
make -j${SLURM_CPUS_PER_TASK}
cd ..

echo "========================================="
echo " 2. Building Standalone CUDA Benchmark"
echo "========================================="
# The benchmark cpp includes kotekan headers, so we link it with the V5 cuda source
# Compiles dual-architecture fatbinary: Hopper (H100) + Blackwell (RTX 5090)
nvcc -std=c++17 -O3 --use_fast_math \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_120,code=compute_120 \
    -Xptxas -O3 \
    -I lib/cuda -I lib/core -I lib/ \
    test_charts/benchmark_kotekan_cuda_tracker_v5.cpp \
    lib/cuda/cudaBeamTrackerV5.cu \
    -o benchmark_tracker

echo "========================================="
echo " 3. Running Standalone Benchmark"
echo "========================================="
./benchmark_tracker

echo "========================================="
echo " 4. Running Full Kotekan Pipeline Test"
echo "========================================="
# Run the pipeline test yaml configuration
./build/kotekan/kotekan --config test_charts/config/64antennas_beam_tracker_test.yaml
