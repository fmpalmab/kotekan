#!/usr/bin/env bash
# Capture finite NoClk RFSoC data, correlate it on the GPU, and plot selected inputs.
#
# Environment overrides:
#   KOTEKAN_BIN=/path/to/kotekan
#   PYTHON_BIN=/path/to/python
#   DATA_DIR=/data/32_corr_noclk
#   ANTENNAS=all        (or e.g. 0-7,16-23)
#   CORR_FILE_COUNT=5  (frames averaged for the plots; 0 means all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_TEMPLATE="${REPO_ROOT}/test_charts/config/32antennas_ingest_gpu_NoClk.yaml"
PLOT_SCRIPT="${REPO_ROOT}/test_charts/plot_correlations_gpu.py"
KOTEKAN_BIN="${KOTEKAN_BIN:-${REPO_ROOT}/build/kotekan/kotekan}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-/data/32_corr_noclk}"
RAW_DIR="${DATA_DIR}/raw"
PLOT_DIR="${DATA_DIR}/plots"
ANTENNAS="${ANTENNAS:-all}"
CORR_FILE_COUNT="${CORR_FILE_COUNT:-5}"

if [[ ! -x "${KOTEKAN_BIN}" ]]; then
    echo "Kotekan binary is not executable: ${KOTEKAN_BIN}" >&2
    echo "Set KOTEKAN_BIN to an existing build; this script does not build Kotekan." >&2
    exit 1
fi

if [[ ! -r "${CONFIG_TEMPLATE}" || ! -r "${PLOT_SCRIPT}" ]]; then
    echo "Required config template or plot script is missing." >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter is unavailable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    if [[ "${EUID}" -eq 0 ]]; then
        mkdir -p "${RAW_DIR}" "${PLOT_DIR}"
    else
        sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "${RAW_DIR}" "${PLOT_DIR}"
    fi
fi

if [[ ! -w "${DATA_DIR}" || ! -w "${RAW_DIR}" || ! -w "${PLOT_DIR}" ]]; then
    echo "Output directory is not writable: ${DATA_DIR}" >&2
    exit 1
fi

shopt -s nullglob
existing_files=("${RAW_DIR}"/correlation_0_*.bin)
if (( ${#existing_files[@]} )); then
    echo "Refusing to overwrite existing correlation frames in ${RAW_DIR}:" >&2
    printf '  %s\n' "${existing_files[@]}" >&2
    exit 1
fi

# Kotekan needs a concrete path in YAML. Substitute only the dedicated placeholder
# into a temporary config so DATA_DIR remains selectable without editing the template.
RUNTIME_CONFIG="$(mktemp)"
trap 'rm -f "${RUNTIME_CONFIG}"' EXIT
escaped_raw_dir="${RAW_DIR//\\/\\\\}"
escaped_raw_dir="${escaped_raw_dir//&/\\&}"
escaped_raw_dir="${escaped_raw_dir//|/\\|}"
sed "s|__CORR_RAW_DIR__|${escaped_raw_dir}|g" "${CONFIG_TEMPLATE}" > "${RUNTIME_CONFIG}"

cd "${REPO_ROOT}"
if [[ "${EUID}" -eq 0 ]]; then
    "${KOTEKAN_BIN}" -c "${RUNTIME_CONFIG}"
else
    sudo -E "${KOTEKAN_BIN}" -c "${RUNTIME_CONFIG}"
fi

correlation_files=("${RAW_DIR}"/correlation_0_*.bin)
if (( ! ${#correlation_files[@]} )); then
    echo "Kotekan completed but wrote no correlation files in ${RAW_DIR}." >&2
    exit 1
fi

"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
    --data-dir "${RAW_DIR}" \
    --output-dir "${PLOT_DIR}" \
    --file-count "${CORR_FILE_COUNT}" \
    --file-position last \
    --antennas "${ANTENNAS}" \
    --channels 672 \
    --elements 32 \
    --polarizations 2

echo "Correlation frames: ${RAW_DIR}"
echo "Plots and summary: ${PLOT_DIR}"
