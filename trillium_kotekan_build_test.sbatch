#!/bin/bash
#SBATCH --job-name=kotekan_charts_test
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=kotekan_test_%j.log

echo "========================================="
echo " Loading Trillium Modules & Runtime Env"
echo "========================================="
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 2>/dev/null || module load gcc || true
module load cuda/12.6 2>/dev/null || module load cuda || true
module load cmake 2>/dev/null || true
module load boost 2>/dev/null || true

# Export CVMFS runtime dynamic library paths for GCC 12 and CUDA
export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64:${EBROOTGCCCORE:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/gcccore/12.3.0}/lib64:${EBROOTCUDA:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2}/lib64:${LD_LIBRARY_PATH:-}"

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

# Configure CMake with CUDA, NUMA, and RPATH enabled
cmake .. -DUSE_CUDA=ON -DUSE_NUMA=ON -DWERROR=OFF \
         -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
         -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE

# Compile Kotekan using available CPU cores
make -j${SLURM_CPUS_PER_TASK}
cd ..

echo "========================================="
echo " 2. Building Standalone CUDA Benchmark"
echo "========================================="
# Compiles directly for the host GPU node with fast math and all include directories
nvcc -std=c++17 -O3 --use_fast_math \
    -arch=native \
    -Xptxas -O3 \
    -I lib/cuda -I lib/metadata -I lib/core -I lib/utils -I lib/ \
    -I external/json -I external/fmt/include -I external/gputils/include \
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
# Create temporary output directory for intensity test file
mkdir -p /tmp/test_charts_tracker

# Run the pipeline test yaml configuration
./build/kotekan/kotekan --config test_charts/config/64antennas_beam_tracker_test.yaml
