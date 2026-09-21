#!/bin/bash
# ==============================================================================
# CHARTS Pipeline Video Generator Launcher
# Automatically loads scientific Python modules (matplotlib, scipy-stack, ffmpeg)
# on Trillium / Compute Canada clusters
# ==============================================================================
set -e

# Load modules if on a cluster environment
if command -v module >/dev/null 2>&1; then
    module load StdEnv/2023 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
    module load scipy-stack 2>/dev/null || true
    module load ffmpeg 2>/dev/null || true
    module load hdf5 2>/dev/null || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="python"
fi

exec "${PYTHON_EXEC}" "${SCRIPT_DIR}/generate_pipeline_videos.py" "$@"
