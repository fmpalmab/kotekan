#!/usr/bin/env bash
# Capture a finite NoClk RFSoC raw stream for the 32-element CHARTS model and
# automatically generate intensity maps and one spectrum panel per antenna.
#
# Usage:
#   run_32antennas_raw_record_NoClk.sh SECONDS
#
# Environment overrides:
#   KOTEKAN_BIN=/path/to/kotekan
#   PYTHON_BIN=python3
#   DATA_DIR=/data/32_test/raw
#   TIME_BIN_SPECTRA=256
#   SCALE=db|linear
#   KOTEKAN_USE_SUDO=0      # run directly when the binary already has access

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_TEMPLATE="${REPO_ROOT}/charts/config/pipeline_32antennas_baseband_record.yaml"
PLOT_SCRIPT="${REPO_ROOT}/charts/plot_raw_baseband.py"

KOTEKAN_BIN="${KOTEKAN_BIN:-${REPO_ROOT}/build/kotekan/kotekan}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-/data/32_test/raw}"
CAPTURE_SECONDS="${1:-${CAPTURE_SECONDS:-}}"
TIME_BIN_SPECTRA="${TIME_BIN_SPECTRA:-256}"
SCALE="${SCALE:-db}"
KOTEKAN_USE_SUDO="${KOTEKAN_USE_SUDO:-1}"
SPECTRA_PER_SECOND=300000
SPECTRA_PER_FRAME=15360
PLOT_PREFIX="${DATA_DIR}/raw_intensity_32_no_clk"

usage() {
    echo "Usage: $0 SECONDS" >&2
    echo "Example: $0 2" >&2
}

if [[ -z "${CAPTURE_SECONDS}" ]]; then
    usage
    exit 2
fi
if [[ ! "${CAPTURE_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v seconds="${CAPTURE_SECONDS}" 'BEGIN { exit !(seconds > 0) }'; then
    echo "Capture duration must be a positive number of seconds: ${CAPTURE_SECONDS}" >&2
    exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter is unavailable: ${PYTHON_BIN}" >&2
    exit 1
fi

CAPTURE_FRAMES="$(awk -v seconds="${CAPTURE_SECONDS}" -v rate="${SPECTRA_PER_SECOND}" -v per_frame="${SPECTRA_PER_FRAME}" 'BEGIN { value = seconds * rate / per_frame; frames = int(value); if (value > frames) frames++; if (frames < 1) frames = 1; print frames }')"
EFFECTIVE_SECONDS="$(awk -v frames="${CAPTURE_FRAMES}" -v per_frame="${SPECTRA_PER_FRAME}" -v rate="${SPECTRA_PER_SECOND}" 'BEGIN { printf "%.6f", frames * per_frame / rate }')"

if [[ ! -x "${KOTEKAN_BIN}" ]]; then
    echo "Kotekan binary is not executable: ${KOTEKAN_BIN}" >&2
    echo "Set KOTEKAN_BIN to an existing build; this script does not build Kotekan." >&2
    exit 1
fi
if [[ ! -r "${CONFIG_TEMPLATE}" || ! -r "${PLOT_SCRIPT}" ]]; then
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

# Reuse the output directory, removing only files owned by this pipeline.
shopt -s nullglob
old_raw_files=("${DATA_DIR}"/*_network_0_*.bin)
old_plot_files=("${PLOT_PREFIX}"_grid.png "${PLOT_PREFIX}".npz)
if (( ${#old_raw_files[@]} )); then
    rm -f -- "${old_raw_files[@]}"
fi
for old_plot in "${old_plot_files[@]}"; do
    [[ -e "${old_plot}" ]] && rm -f -- "${old_plot}"
done

RUNTIME_CONFIG="$(mktemp "${TMPDIR:-/tmp}/charts32-raw-config.XXXXXX.yaml")"
trap 'rm -f "${RUNTIME_CONFIG}"' EXIT
escaped_raw_dir="${DATA_DIR//&/\&}"
escaped_raw_dir="${escaped_raw_dir//|/\|}"
sed \
    -e "s|__RAW_DIR__|${escaped_raw_dir}|g" \
    -e "s|^capture_n_frames:.*$|capture_n_frames: ${CAPTURE_FRAMES}|" \
    "${CONFIG_TEMPLATE}" > "${RUNTIME_CONFIG}"

echo "Capturing ${CAPTURE_FRAMES} frames (~${EFFECTIVE_SECONDS} s) into ${DATA_DIR}"
cd "${REPO_ROOT}"
if [[ "${EUID}" -eq 0 || "${KOTEKAN_USE_SUDO}" == "0" ]]; then
    "${KOTEKAN_BIN}" -c "${RUNTIME_CONFIG}"
else
    sudo -E "${KOTEKAN_BIN}" -c "${RUNTIME_CONFIG}"
fi

captured_files=("${DATA_DIR}"/*_network_0_*.bin)
if (( ! ${#captured_files[@]} )); then
    echo "Kotekan completed without producing *_network_0_*.bin in ${DATA_DIR}." >&2
    exit 1
fi

"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
    --data-dir "${DATA_DIR}" \
    --handler 0 \
    --channels 672 \
    --elements 32 \
    --spectra-per-frame "${SPECTRA_PER_FRAME}" \
    --file-count 0 \
    --file-position first \
    --antennas all \
    --exclude-zero-spectra \
    --time-bin-spectra "${TIME_BIN_SPECTRA}" \
    --scale "${SCALE}" \
    --output-prefix "${PLOT_PREFIX}"

echo "Raw frames: ${#captured_files[@]}"
echo "Plots written with prefix: ${PLOT_PREFIX}"
