#!/usr/bin/env bash
# Capture a single NoClk RFSoC stream, then plot the recorded raw frames.
#
# Environment overrides:
#   KOTEKAN_BIN=/path/to/kotekan
#   DATA_DIR=/data/32_test
#   PLOT_FILE_COUNT=5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/test_charts/config/32antennas_raw_record_NoClk.yaml"
KOTEKAN_BIN="${KOTEKAN_BIN:-${REPO_ROOT}/build/kotekan/kotekan}"
PLOT_SCRIPT="${REPO_ROOT}/test_charts/plot_raw_intensity_64.py"
DATA_DIR="${DATA_DIR:-/data/32_test}"
PLOT_FILE_COUNT="${PLOT_FILE_COUNT:-5}"
PLOT_PREFIX="${DATA_DIR}/raw_intensity_32_no_clk"

if [[ ! -x "${KOTEKAN_BIN}" ]]; then
    echo "Kotekan binary is not executable: ${KOTEKAN_BIN}" >&2
    echo "Set KOTEKAN_BIN to an existing build; this script does not build Kotekan." >&2
    exit 1
fi

if [[ ! -r "${CONFIG}" || ! -r "${PLOT_SCRIPT}" ]]; then
    echo "Required config or plot script is missing." >&2
    exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    if [[ "${EUID}" -eq 0 ]]; then
        mkdir -p "${DATA_DIR}"
    else
        sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "${DATA_DIR}"
    fi
fi

if [[ ! -w "${DATA_DIR}" ]]; then
    echo "Data directory is not writable: ${DATA_DIR}" >&2
    exit 1
fi

shopt -s nullglob
existing_files=("${DATA_DIR}"/*_network_0_*.bin)
if (( ${#existing_files[@]} > 0 )); then
    echo "Refusing to overwrite existing raw frames in ${DATA_DIR}." >&2
    echo "Move or remove only the previous *_network_0_*.bin files, then rerun." >&2
    exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
    "${KOTEKAN_BIN}" -c "${CONFIG}"
else
    sudo -E "${KOTEKAN_BIN}" -c "${CONFIG}"
fi

captured_files=("${DATA_DIR}"/*_network_0_*.bin)
if (( ${#captured_files[@]} == 0 )); then
    echo "Capture ended without producing *_network_0_*.bin in ${DATA_DIR}." >&2
    exit 1
fi

python3 "${PLOT_SCRIPT}" \
    --data-dir "${DATA_DIR}" \
    --handler 0 \
    --channels 672 \
    --elements 32 \
    --spectra-per-frame 15360 \
    --file-count "${PLOT_FILE_COUNT}" \
    --file-position last \
    --antennas all \
    --exclude-zero-spectra \
    --spectra-grid \
    --output-prefix "${PLOT_PREFIX}"

echo "Plots written with prefix: ${PLOT_PREFIX}"
