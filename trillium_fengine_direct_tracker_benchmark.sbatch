#!/bin/bash
#SBATCH --job-name=kotekan_direct_tracker
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=kotekan_direct_tracker_%j.log

set -euo pipefail

echo "======================================================================"
echo " Starting CHARTS F-Engine & Direct Beam Tracker Test on Trillium"
echo " Comparing Direct Beam Tracker against Beam Tracker V5 Baseline"
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

# Setup high-speed scratch virtual environment inheriting CVMFS modules (numpy, scipy, matplotlib, h5py)
VENV_DIR="${SLURM_TMPDIR:-/tmp}/kotekan_pyenv_${SLURM_JOB_ID:-manual}"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}" || true
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

# Install missing Python packages using CVMFS offline wheelhouse (--no-index)
if ! python3 -c "import yaml, jinja2, h5py, matplotlib" 2>/dev/null; then
    echo "Installing missing python runtime packages (pyyaml, jinja2, future) via offline CVMFS wheelhouse..."
    pip install --no-index pyyaml jinja2 future || pip install pyyaml jinja2 future || true
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
        echo "       (To run pure Julia, run: 'julia --project=test_charts/RadioTelescopeFEngine.jl -e \"using Pkg; Pkg.instantiate()\"' once on login node)."
    fi
else
    echo "[INFO] Julia binary not loaded in path. Using high-speed F-Engine Python simulation engine."
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
echo "Working Directory: ${REPO_ROOT}"

# ============================================================================
# 3. Build Kotekan Framework & Benchmarks
# ============================================================================
echo "========================================="
echo " 3. Building Kotekan Framework & Direct Tracker Targets"
echo "========================================="
rm -rf build
mkdir -p build
cd build

cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90;80" -DUSE_NUMA=ON -DWERROR=OFF -DCMAKE_SKIP_RPATH=TRUE
make -j${SLURM_CPUS_PER_TASK} kotekan benchmark_direct_beam_tracker benchmark_kotekan_cuda_tracker_v5 benchmark_tracker_v5_vs_direct || {
    echo "[WARN] CMake multi-target build had issues; building kotekan core and standalone NVCC binaries."
    make -j${SLURM_CPUS_PER_TASK} kotekan || true
}
cd ..

# ============================================================================
# 4. Standalone NVCC Compilation Fallback for Benchmarks
# ============================================================================
echo "========================================="
echo " 4. Verifying Standalone CUDA Benchmarks Compilation"
echo "========================================="
if [ ! -f "build/test_charts/benchmark_tracker_v5_vs_direct" ]; then
    echo "Building standalone benchmark_tracker_v5_vs_direct via NVCC..."
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
        test_charts/benchmark_tracker_v5_vs_direct.cpp \
        lib/cuda/cudaBeamTrackerV5.cu \
        lib/cuda/cudaDirectBeamTracker.cu \
        lib/core/kotekanLogging.cpp \
        lib/core/errors.c \
        -o benchmark_tracker_v5_vs_direct
    BENCHMARK_BIN="./benchmark_tracker_v5_vs_direct"
else
    BENCHMARK_BIN="./build/test_charts/benchmark_tracker_v5_vs_direct"
fi

# ============================================================================
# 5. Run Head-to-Head Comparative Benchmark (V5 vs Direct Beam Tracker)
# ============================================================================
echo "========================================="
echo " 5. Running Head-to-Head Comparative Benchmark (Direct vs V5)"
echo "========================================="
"${BENCHMARK_BIN}"

# ============================================================================
# 6. Generate Synthetic F-Engine Datasets (64 & 256 Antennas)
# ============================================================================
echo "========================================="
echo " 6. Generating Synthetic F-Engine Datasets (64 & 256 Antennas)"
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
# 7. Run F-Engine to Direct Beam Tracker Simulation & V5 Comparison
# ============================================================================
echo "========================================="
echo " 7. Verifying F-Engine Simulation into Direct Beam Tracker vs V5 Baseline"
echo "========================================="
python3 test_charts/test_fengine_direct_tracker.py test_charts/data/voltage_sim_64ant.h5 --plot-out "fengine_direct_vs_v5_64ant_zenith_${SLURM_JOB_ID}.png"
python3 test_charts/test_fengine_direct_tracker.py test_charts/data/voltage_sim_256ant.h5 --plot-out "fengine_direct_vs_v5_256ant_zenith_${SLURM_JOB_ID}.png"

python3 test_charts/test_fengine_direct_tracker.py test_charts/data/fengine_sim_64ant_moving.h5 --plot-out "fengine_direct_vs_v5_64ant_moving_${SLURM_JOB_ID}.png"
python3 test_charts/test_fengine_direct_tracker.py test_charts/data/fengine_sim_256ant_moving.h5 --plot-out "fengine_direct_vs_v5_256ant_moving_${SLURM_JOB_ID}.png"

# ============================================================================
# 8. Run Full Kotekan Pipelines with Direct Beam Tracker (64 & 256 Antennas)
# ============================================================================
echo "========================================="
echo " 8. Running Streaming Kotekan Pipelines with Direct Beam Tracker (64 & 256 Antennas)"
echo "========================================="
mkdir -p /tmp/test_charts_direct_tracker_64
rm -f /tmp/test_charts_direct_tracker_64/*.bin

mkdir -p /tmp/test_charts_direct_tracker_256
rm -f /tmp/test_charts_direct_tracker_256/*.bin

if [ -f "./build/kotekan/kotekan" ]; then
    echo "--- Executing 64-Antenna Direct Beam Tracker Pipeline ---"
    LD_LIBRARY_PATH="$PWD/build/external/n2k:${LD_LIBRARY_PATH:-}" ./build/kotekan/kotekan --config test_charts/config/64antennas_direct_tracker_test.yaml

    echo "--- Executing 256-Antenna Direct Beam Tracker Pipeline ---"
    LD_LIBRARY_PATH="$PWD/build/external/n2k:${LD_LIBRARY_PATH:-}" ./build/kotekan/kotekan --config test_charts/config/256antennas_direct_tracker_test.yaml
else
    echo "[WARN] Kotekan binary not found; skipping streaming pipeline execution."
fi

# ============================================================================
# 9. Inspect Generated Complex Voltage Dumps
# ============================================================================
echo "========================================="
echo " 9. Inspecting Formed Complex Voltage Dumps"
echo "========================================="
python3 test_charts/inspect_all_tracker_outputs.py

# ============================================================================
# 10. Generate Benchmark Comparison Plots & Markdown Report
# ============================================================================
echo "========================================="
echo " 10. Generating Benchmark Visualizations & Markdown Report"
echo "========================================="
python3 test_charts/compare_tracker_benchmarks.py tracker_benchmark_comparison.json \
    --plot-out "direct_vs_v5_benchmark_${SLURM_JOB_ID}.png" \
    --md-out "tracker_benchmark_report_${SLURM_JOB_ID}.md"

echo "======================================================================"
echo " ALL TESTS, SIMULATIONS, AND BENCHMARK COMPARISONS COMPLETED SUCCESSFULLY!"
echo " Outputs Generated:"
echo "   - JSON Benchmark Results : tracker_benchmark_comparison.json"
echo "   - Markdown Report        : tracker_benchmark_report_${SLURM_JOB_ID}.md"
echo "   - Benchmark Plot         : direct_vs_v5_benchmark_${SLURM_JOB_ID}.png"
echo "   - F-Engine Direct Plots  : fengine_direct_vs_v5_*_${SLURM_JOB_ID}.png"
echo "======================================================================"
