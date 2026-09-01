#!/usr/bin/env python3
"""CHARTS Constants Compatibility Layer for Kotekan test_charts.

Loads official constants from the `charts_constants` package if installed or
present in the filesystem. If `charts_constants` is not available (e.g. initial
clone or standalone environment), gracefully falls back to embedded exact standard
CHARTS constants so that all test_charts scripts, bridges, and benchmarks run out of the box.
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("kotekan.charts.constants")

# ---------------------------------------------------------------------------
# Attempt to load from external `charts_constants` package or adjacent repository
# ---------------------------------------------------------------------------
_external_module = None
_source_description = "embedded_fallback"

# 1. Check if charts_constants is already in sys.modules or installed in environment
try:
    _external_module = importlib.import_module("charts_constants")
    _source_description = "installed_charts_constants"
except Exception:
    # 2. Check adjacent directories (e.g., ../../charts-constants, ../charts-constants)
    _test_charts_dir = Path(__file__).resolve().parent
    _kotekan_root = _test_charts_dir.parent
    _charts_root = _kotekan_root.parent
    _adjacent_paths = [
        _charts_root / "charts-constants",
        _kotekan_root / "charts-constants",
    ]
    for _path in _adjacent_paths:
        if _path.exists() and (_path / "charts_constants").is_dir():
            if str(_path) not in sys.path:
                sys.path.insert(0, str(_path))
            try:
                _external_module = importlib.import_module("charts_constants")
                _source_description = f"path_charts_constants ({_path})"
                break
            except Exception:
                pass


def _extract_val(val: Any) -> Any:
    """Extract raw float/int/array from Astropy Quantity or return as-is."""
    if hasattr(val, "value"):
        return val.value
    return val


def _has_external(attr_name: str) -> bool:
    return _external_module is not None and hasattr(_external_module, attr_name)


def _get_external(attr_name: str, default: Any) -> Any:
    if _has_external(attr_name):
        raw = getattr(_external_module, attr_name)
        return _extract_val(raw)
    return default


# ---------------------------------------------------------------------------
# Metadata Flags
# ---------------------------------------------------------------------------
USING_CHARTS_CONSTANTS_PACKAGE: bool = _external_module is not None
USING_FALLBACK_CONSTANTS: bool = not USING_CHARTS_CONSTANTS_PACKAGE
CONSTANTS_SOURCE: str = _source_description


# ---------------------------------------------------------------------------
# 1. Dispersion Measure & Physical Constants
# ---------------------------------------------------------------------------
K_DM: float = float(_get_external("K_DM", 4148.741601))
C_LIGHT: float = float(_get_external("C_LIGHT", 299_792_458.0))
SPEED_OF_LIGHT: float = C_LIGHT
SPEED_OF_LIGHT_M_PER_S: float = C_LIGHT


# ---------------------------------------------------------------------------
# 2. Instrumental & Sampling Constants
# ---------------------------------------------------------------------------
ADC_SAMPLING_FREQ_HZ: float = float(_get_external("ADC_SAMPLING_FREQ", 2457.6e6))
ADC_SAMPLING_FREQ_MHZ: float = float(_get_external("ADC_SAMPLING_FREQ_MHZ", 2457.6))
DEFAULT_SAMPLE_RATE: float = ADC_SAMPLING_FREQ_MHZ
FPGA_FREQ0_MHZ: float = ADC_SAMPLING_FREQ_MHZ

FPGA_NUM_SAMP_FFT: int = int(_get_external("FPGA_NUM_SAMP_FFT", 8192))
DEFAULT_NFFT: int = FPGA_NUM_SAMP_FFT
NFFT: int = FPGA_NUM_SAMP_FFT

CHARTS_CHANNEL_WIDTH_MHZ: float = float(_get_external("CHARTS_CHANNEL_WIDTH_MHZ", 0.3))
CHARTS_CHANNEL_WIDTH_HZ: float = CHARTS_CHANNEL_WIDTH_MHZ * 1e6
DEFAULT_CHANNEL_WIDTH_HZ: float = CHARTS_CHANNEL_WIDTH_HZ

FPGA_TIME_RESOLUTION_S: float = float(_get_external("FPGA_TIME_RESOLUTION_S", 8192 / 2457.6e6))
FPGA_TIME_RESOLUTION_MS: float = FPGA_TIME_RESOLUTION_S * 1e3
FPGA_TIME_RESOLUTION_US: float = FPGA_TIME_RESOLUTION_S * 1e6


# ---------------------------------------------------------------------------
# 3. Frequency Multiplexing & Band Constants
# ---------------------------------------------------------------------------
CHARTS_N_FREQ: int = int(_get_external("CHARTS_N_FREQ", 672))
NUM_CHANNELS: int = CHARTS_N_FREQ
ANTENNA_CHANNELS: int = CHARTS_N_FREQ
FULL_BAND_FREQUENCY_CHANNELS: int = CHARTS_N_FREQ

FREQUENCY_SHARD_COUNT: int = 2
LOCAL_FREQUENCY_CHANNELS: int = CHARTS_N_FREQ // FREQUENCY_SHARD_COUNT  # 336
DEFAULT_N_FREQ: int = LOCAL_FREQUENCY_CHANNELS

CHAIN_BANDS: List[Tuple[float, float]] = list(_get_external("CHAIN_BANDS", [
    (300.0, 501.6),     # Chain 0 (bins 1000:1672)
    (564.0, 765.6),     # Chain 1 (bins 1880:2552)
    (830.4, 1032.0),    # Chain 2 (bins 2768:3440)
    (1099.2, 1300.8),   # Chain 3 (bins 3664:4336)
    (1365.6, 1567.2),   # Chain 4 (bins 4552:5224)
    (1632.0, 1833.6),   # Chain 5 (bins 5440:6112)
    (1900.8, 2102.4),   # Chain 6 (bins 6336:7008)
    (2164.8, 2366.4),   # Chain 7 (bins 7216:7888)
]))

BAND_EDGES: List[float] = [edge for band in CHAIN_BANDS for edge in band]
DEFAULT_FREQUENCY_START_HZ: float = CHAIN_BANDS[0][0] * 1e6  # 300_000_000.0
DEFAULT_FREQUENCY_START_MHZ: float = CHAIN_BANDS[0][0]       # 300.0
BEAM_GRID_DESIGN_FREQUENCY_HZ: float = 400_000_000.0
DEFAULT_SPACING_M: float = 0.6


# ---------------------------------------------------------------------------
# 4. Site Location & Telescope Layout
# ---------------------------------------------------------------------------
PRESET_LOCATIONS: Dict[str, Tuple[float, float, float]] = {
    "Calan": (-33.39628, -70.536695, 860.0),
    "Caren": (-33.4211146, -70.8634710, 458.0),
}
if _has_external("PRESET_LOCATIONS"):
    PRESET_LOCATIONS = _get_external("PRESET_LOCATIONS", PRESET_LOCATIONS)

CHARTS_LATITUDE_DEG, CHARTS_LONGITUDE_DEG, CHARTS_ALTITUDE_M = PRESET_LOCATIONS["Caren"]
CHARTS_N_ANTENNAS: int = int(_get_external("CHARTS_N_ANTENNAS", 256))


# ---------------------------------------------------------------------------
# 5. CPT Dual-Band Constants
# ---------------------------------------------------------------------------
CPT_SAMPLE_RATE: float = float(_get_external("SAMPLE_RATE", 4915.2))
CPT_DELTA_TIME: float = float(_get_external("DELTA_TIME", (10.0 / 3.0) * 1e-6))
CPT_FREQ_0_BAND1: float = float(_get_external("FREQ_0_BAND1", 300.0))
CPT_FREQ_0_BAND2: float = float(_get_external("FREQ_0_BAND2", 1365.6))
CPT_DELTA_FREQ: float = float(_get_external("DELTA_FREQ", 0.3))
CPT_NUM_FREQ: int = int(_get_external("NUM_FREQ", 672))
CPT_NUM_BANDS: int = int(_get_external("NUM_BANDS", 2))
CPT_SPECTRA_PER_PACKET: int = int(_get_external("SPECTRA_PER_PACKET", 4))


# ---------------------------------------------------------------------------
# 6. Dynamic HDF5 Baseband Data Path Resolution
# ---------------------------------------------------------------------------
def get_default_charts_h5_path() -> Path:
    """Dynamically resolves path to baseband_virtual.h5 without hardcoded user directories."""
    import os
    env_path = os.environ.get("CHARTS_H5_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        _charts_root / "data" / "260816T013722Z_CHARTS_hdf5" / "baseband_virtual.h5",
        _kotekan_root / "data" / "260816T013722Z_CHARTS_hdf5" / "baseband_virtual.h5",
        Path.home() / "charts" / "data" / "260816T013722Z_CHARTS_hdf5" / "baseband_virtual.h5",
        Path.home() / "fpalma" / "charts" / "data" / "260816T013722Z_CHARTS_hdf5" / "baseband_virtual.h5",
        Path("/tmp/kotekan_continuous_tracker/input/window_replay_0000000.bin"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback to standard relative location
    return _charts_root / "data" / "260816T013722Z_CHARTS_hdf5" / "baseband_virtual.h5"

