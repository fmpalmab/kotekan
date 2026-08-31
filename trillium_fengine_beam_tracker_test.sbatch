#!/bin/bash
#SBATCH --job-name=kotekan_fengine_tracker
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=kotekan_fengine_tracker_%j.log

set -euo pipefail

echo "======================================================================"
echo " Starting CHARTS Radio Telescope F-Engine & Beam Tracker Test on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Job ID         : ${SLURM_JOB_ID}"
echo " Date           : $(date)"
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
module load julia/1.10 2>/dev/null || module load julia 2>/dev/null || true

echo "=== Loaded Compilers & Environment ==="
gcc --version | head -n 1
which nvcc && nvcc --version | tail -n 2
cmake --version | head -n 1
python3 --version | head -n 1
nvidia-smi

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"

# Install missing Python packages using CVMFS offline wheelhouse (--no-index)
if ! python3 -c "import yaml, jinja2" 2>/dev/null; then
    echo "Installing missing python runtime packages (pyyaml, jinja2) via offline CVMFS wheelhouse..."
    pip install --no-index --user pyyaml jinja2 || true
fi

# ============================================================================
# 2. Check Julia Environment
# ============================================================================
echo "========================================="
echo " 2. Checking Julia Environment"
echo "========================================="
HAS_JULIA=false
if command -v julia >/dev/null 2>&1; then
    echo "Julia binary found: $(which julia) ($(julia --version))"
    export JULIA_DEPOT_PATH="${SLURM_TMPDIR:-$HOME}/.julia:${JULIA_DEPOT_PATH:-$HOME/.julia}"
    if julia --project=test_charts/RadioTelescopeFEngine.jl -e 'using RadioTelescopeFEngine' 2>/dev/null; then
        echo "RadioTelescopeFEngine.jl packages are verified and available in Julia depot."
        HAS_JULIA=true
    else
        echo "[INFO] RadioTelescopeFEngine.jl packages not yet instantiated in depot."
        echo "       (To run pure Julia, run: 'julia --project=test_charts/RadioTelescopeFEngine.jl -e \"using Pkg; Pkg.instantiate()\"' once on the login node with internet access)."
    fi
else
    echo "[INFO] Julia binary not loaded in path. Using high-speed F-Engine Python simulation engine."
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
echo "Working Directory: ${REPO_ROOT}"

# ============================================================================
# 3. Build Kotekan Framework & Tests
# ============================================================================
echo "========================================="
echo " 3. Building Kotekan Framework"
echo "========================================="
rm -rf build
mkdir -p build
cd build

cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90;80" -DUSE_NUMA=ON -DWERROR=OFF -DCMAKE_SKIP_RPATH=TRUE
make -j${SLURM_CPUS_PER_TASK}
cd ..

# ============================================================================
# 4. Build Standalone CUDA Test Suite & Benchmark
# ============================================================================
echo "========================================="
echo " 4. Building Standalone CUDA Beam Tracker Suite"
echo "========================================="
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

# ============================================================================
# 5. Run CUDA Test Suite (64, 128, 256 Antennas)
# ============================================================================
echo "========================================="
echo " 5. Running CUDA Beam Tracker Test Suite (64 & 256 Antennas)"
echo "========================================="
./test_suite_tracker
./benchmark_tracker

# ============================================================================
# 6. Generate Synthetic F-Engine Datasets (64 & 256 Antennas)
# ============================================================================
echo "========================================="
echo " 6. Generating F-Engine Simulation Datasets"
echo "========================================="
mkdir -p test_charts/data

if [ "$HAS_JULIA" = true ]; then
    echo "--- Running 64-Antenna Simulation in Julia (RadioTelescopeFEngine.jl) ---"
    julia --project=test_charts/RadioTelescopeFEngine.jl test_charts/RadioTelescopeFEngine.jl/bin/run_charts_simulation.jl 64 test_charts/data/voltage_sim_64ant.h5

    echo "--- Running 256-Antenna Simulation in Julia (RadioTelescopeFEngine.jl) ---"
    julia --project=test_charts/RadioTelescopeFEngine.jl test_charts/RadioTelescopeFEngine.jl/bin/run_charts_simulation.jl 256 test_charts/data/voltage_sim_256ant.h5
else
    echo "--- Running 64-Antenna F-Engine Simulation (Python Engine) ---"
    python3 test_charts/generate_fengine_sim_data.py --antennas 64 --scenario zenith --output test_charts/data/voltage_sim_64ant.h5

    echo "--- Running 256-Antenna F-Engine Simulation (Python Engine) ---"
    python3 test_charts/generate_fengine_sim_data.py --antennas 256 --scenario zenith --output test_charts/data/voltage_sim_256ant.h5
fi

echo "--- Generating Dynamic Moving Celestial Target Scenarios ---"
python3 test_charts/generate_fengine_sim_data.py --antennas 64 --scenario moving --output test_charts/data/fengine_sim_64ant_moving.h5
python3 test_charts/generate_fengine_sim_data.py --antennas 256 --scenario moving --output test_charts/data/fengine_sim_256ant_moving.h5

# ============================================================================
# 7. Run Verification Bridge & Diagnostics
# ============================================================================
echo "========================================="
echo " 7. Verifying F-Engine Outputs with Kotekan Beam Tracker"
echo "========================================="
python3 test_charts/test_fengine_tracker_bridge.py test_charts/data/voltage_sim_64ant.h5 --plot-out "fengine_tracker_64ant_${SLURM_JOB_ID}.png"
python3 test_charts/test_fengine_tracker_bridge.py test_charts/data/voltage_sim_256ant.h5 --plot-out "fengine_tracker_256ant_${SLURM_JOB_ID}.png"

python3 test_charts/test_fengine_tracker_bridge.py test_charts/data/fengine_sim_64ant_moving.h5 --plot-out "fengine_tracker_sim_64ant_moving_${SLURM_JOB_ID}.png"
python3 test_charts/test_fengine_tracker_bridge.py test_charts/data/fengine_sim_256ant_moving.h5 --plot-out "fengine_tracker_sim_256ant_moving_${SLURM_JOB_ID}.png"

# ============================================================================
# 8. Run Full Kotekan Pipelines (64 & 256 Antennas)
# ============================================================================
echo "========================================="
echo " 8. Running Full Kotekan Pipelines (64 & 256 Antennas)"
echo "========================================="
mkdir -p /tmp/test_charts_tracker
rm -f /tmp/test_charts_tracker/*.bin

mkdir -p /tmp/test_charts_tracker_256
rm -f /tmp/test_charts_tracker_256/*.bin

echo "--- Executing 64-Antenna Pipeline ---"
LD_LIBRARY_PATH="$PWD/build/external/n2k:$LD_LIBRARY_PATH" ./build/kotekan/kotekan --config test_charts/config/64antennas_beam_tracker_test.yaml

echo "--- Executing 256-Antenna Pipeline ---"
LD_LIBRARY_PATH="$PWD/build/external/n2k:$LD_LIBRARY_PATH" ./build/kotekan/kotekan --config test_charts/config/256antennas_beam_tracker_test.yaml

# ============================================================================
# 9. Inspect Generated Complex Voltage Dumps
# ============================================================================
echo "========================================="
echo " 9. Inspecting Generated Complex Voltage Dumps"
echo "========================================="
python3 test_charts/inspect_all_tracker_outputs.py

echo "======================================================================"
echo " ALL 64 & 256 ANTENNA SIMULATIONS & BEAM TRACKER TESTS COMPLETED SUCCESSFULLY!"
echo " Diagnostic Plots Saved: fengine_tracker_*.png"
echo "======================================================================"
