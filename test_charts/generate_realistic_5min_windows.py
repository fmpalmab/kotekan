#!/usr/bin/env python3
"""
CHARTS Realistic 5-Minute Window Generator
=========================================
Generates physically realistic 5-minute baseband data windows for CHARTS:
  - Window A (15:00 UTC): Daytime window at Observatorio Carén (Sun up, high elevation,
    steady coherent solar emission elevating system temperature).
  - Window B (03:00 UTC): Nighttime window at Observatorio Carén (Sun below horizon,
    pure sky background + receiver thermal noise).
  - Injected Transients: Randomized Fast Radio Bursts (FRB, dispersed chirp),
    pulsar giant pulses / pulse trains (PSR J0437 or Vela-like), narrowband RFI,
    and fast-moving LEO satellite sweeps.
  - Streaming Decimation Strategy:
    Full 5-minute timeline is 58,593 frames (1.85 TB uncompressed).
    To maintain realistic data while keeping storage feasible on cluster scratch:
      * Periodic background frames (e.g. 1 frame every 2.0 s -> 150 frames)
      * Dense capture around transient events (1.0 s at full 5.12 ms cadence)
      * Sparse capture across dispersed sweeps
      * Output files are sequentially numbered (0000000.bin, 0000001.bin, ...)
        for seamless Kotekan rawFileRead streaming.
      * Full 10 Hz analytic lightcurve (3,000 points) and event truth table
        are saved to a companion HDF5 metadata file.
  - Parallelized with multiprocessing across CPU cores.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from charts_noise_model import AnalogChainParams, ChartsNoiseModel
from constants import (
    C_LIGHT,
    CHARTS_ALTITUDE_M,
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    FPGA_TIME_RESOLUTION_US,
    K_DM,
    LOCAL_FREQUENCY_CHANNELS,
)


# ---------------------------------------------------------------------------
# Array Geometry
# ---------------------------------------------------------------------------
def get_charts_antenna_positions(
    num_antennas: int = 64, spacing_m: float = DEFAULT_SPACING_M
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes physical (x, y) coordinates for uniform rectangular grid."""
    if num_antennas <= 64:
        cols = np.arange(num_antennas) & 7
        rows = np.arange(num_antennas) >> 3
    else:
        cols = np.arange(num_antennas) & 15
        rows = np.arange(num_antennas) >> 4
    pos_x = (cols * spacing_m).astype(np.float32)
    pos_y = (rows * spacing_m).astype(np.float32)
    return pos_x, pos_y


# ---------------------------------------------------------------------------
# Event Specifications
# ---------------------------------------------------------------------------
@dataclass
class SimulatedEvent:
    event_id: str
    event_type: str  # 'frb', 'pulsar', 'rfi_narrow', 'rfi_leo'
    t_start_s: float
    duration_s: float
    dm: float = 0.0
    nominal_amp: float = 3.0
    l0: float = 0.0
    m0: float = 0.0
    drift_dl: float = 0.0
    drift_dm: float = 0.0
    pulse_period_s: float = 0.0
    pulse_width_ms: float = 1.0
    channels: Optional[List[int]] = None
    is_persistent: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def schedule_random_events(
    duration_s: float = 300.0,
    num_events: int = 3,
    allowed_types: Optional[List[str]] = None,
    seed: int = 42,
    persistent_rfi_channels: Optional[List[int]] = None,
    persistent_rfi_amp: float = 7.0,
) -> List[SimulatedEvent]:
    """Generates a randomized schedule of transient events plus optional persistent site RFI."""
    rng = np.random.default_rng(seed)
    if allowed_types is None:
        allowed_types = ["frb", "pulsar", "rfi_narrow", "rfi_leo"]

    events: List[SimulatedEvent] = []

    # 1. Site Persistent Narrowband RFI (Always active for 100% of the window)
    if persistent_rfi_channels:
        az = float(rng.uniform(0.0, 2.0 * math.pi))
        l0 = float(0.96 * math.sin(az))
        m0 = float(0.96 * math.cos(az))
        chans = sorted(persistent_rfi_channels)
        freqs_str = ", ".join(f"{DEFAULT_FREQUENCY_START_MHZ + ch * CHARTS_CHANNEL_WIDTH_MHZ:.1f} MHz" for ch in chans)
        events.append(
            SimulatedEvent(
                event_id="site_persistent_rfi",
                event_type="rfi_narrow",
                t_start_s=0.0,
                duration_s=float(duration_s),
                nominal_amp=float(persistent_rfi_amp),
                l0=l0,
                m0=m0,
                channels=chans,
                is_persistent=True,
                description=f"Site continuous RFI on channels {chans} ({freqs_str}) (amp={persistent_rfi_amp:.1f} LSB, 100% duty cycle)",
            )
        )

    # 2. Transient Events (FRB, Pulsar, LEO sweep, etc.)
    # Candidate time slots with buffer margins
    margin_start = min(20.0, duration_s * 0.05)
    margin_end = max(margin_start + 0.01, duration_s - min(30.0, duration_s * 0.1))

    # Ensure events don't overlap excessively
    min_separation = max(0.01, (margin_end - margin_start) / max(1, num_events + 1))
    t_candidates = []
    for _ in range(500):
        if len(t_candidates) >= num_events:
            break
        t_prop = rng.uniform(margin_start, margin_end)
        if all(abs(t_prop - tc) >= min_separation for tc in t_candidates):
            t_candidates.append(t_prop)

    t_candidates.sort()

    for idx, t0 in enumerate(t_candidates):
        ev_type = rng.choice(allowed_types)
        ev_id = f"evt_{idx + 1:02d}_{ev_type}"

        if ev_type == "frb":
            dm = float(rng.uniform(100.0, 350.0))
            # Dispersed sweep duration across 300-400.8 MHz
            # Delta t = 4148.8 * DM * (300^-2 - 400.5^-2) ≈ 0.02027 * DM
            sweep_s = 4148.8 * dm * (1.0 / (300.0 ** 2) - 1.0 / (400.5 ** 2))
            width_ms = float(rng.uniform(1.2, 3.5))
            amp = float(rng.uniform(2.8, 4.5))
            # Random sky arrival within patch beam
            l0 = float(rng.normal(0.0, 0.08))
            m0 = float(rng.normal(0.0, 0.08))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="frb",
                    t_start_s=float(t0),
                    duration_s=float(sweep_s + 0.5),
                    dm=dm,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    pulse_width_ms=width_ms,
                    description=f"Synthetic FRB (DM={dm:.1f} pc/cm^3, width={width_ms:.2f} ms, sweep={sweep_s:.2f} s)",
                )
            )

        elif ev_type == "pulsar":
            # Either J0437-like (MSP) or Vela-like
            is_msp = rng.choice([True, False])
            if is_msp:
                period_s = 0.005757  # PSR J0437-4715 (5.75 ms)
                dm = 2.64
                width_ms = 0.18
                amp = float(rng.uniform(1.8, 2.8))
                duration = 15.0
                desc = f"PSR J0437-4715 MSP train (P={period_s*1e3:.2f} ms, DM={dm:.2f})"
            else:
                period_s = 0.08933  # Vela Pulsar (89.33 ms)
                dm = 67.99
                width_ms = 2.1
                amp = float(rng.uniform(2.2, 3.2))
                duration = 15.0
                desc = f"Vela Pulsar pulse train (P={period_s*1e3:.2f} ms, DM={dm:.2f})"

            l0 = float(rng.normal(0.0, 0.06))
            m0 = float(rng.normal(-0.15, 0.06))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="pulsar",
                    t_start_s=float(t0),
                    duration_s=duration,
                    dm=dm,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    pulse_period_s=period_s,
                    pulse_width_ms=width_ms,
                    description=desc,
                )
            )

        elif ev_type == "rfi_narrow":
            num_bad_chans = int(rng.integers(1, 4))
            chans = sorted(rng.choice(LOCAL_FREQUENCY_CHANNELS, size=num_bad_chans, replace=False).tolist())
            duration = float(rng.uniform(25.0, 50.0))
            amp = float(rng.uniform(3.5, 7.5))
            # Horizon direction (low elevation)
            az = rng.uniform(0.0, 2.0 * math.pi)
            l0 = float(0.96 * math.sin(az))
            m0 = float(0.96 * math.cos(az))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="rfi_narrow",
                    t_start_s=float(t0),
                    duration_s=duration,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    channels=chans,
                    description=f"Narrowband persistent RFI on channels {chans} (amp={amp:.1f} LSB, dur={duration:.1f} s)",
                )
            )

        elif ev_type == "rfi_leo":
            duration = float(rng.uniform(5.0, 8.0))
            amp = float(rng.uniform(11.0, 15.0))  # High amplitude, some ADC clipping
            drift_speed = float(rng.uniform(0.015, 0.035))  # rad / s
            drift_ang = rng.uniform(0.0, 2.0 * math.pi)
            dl = float(drift_speed * math.cos(drift_ang))
            dm = float(drift_speed * math.sin(drift_ang))
            l0 = float(rng.uniform(-0.1, 0.1))
            m0 = float(rng.uniform(-0.1, 0.1))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="rfi_leo",
                    t_start_s=float(t0),
                    duration_s=duration,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    drift_dl=dl,
                    drift_dm=dm,
                    description=f"Fast LEO Satellite RFI sweep (amp={amp:.1f} LSB, dur={duration:.1f} s, drift={drift_speed:.3f}/s)",
                )
            )

    return events


# ---------------------------------------------------------------------------
# Decimated Frame Selection Schedule
# ---------------------------------------------------------------------------
def build_frame_selection_schedule(
    total_frames: int,
    frame_duration_s: float,
    events: List[SimulatedEvent],
    background_cadence_s: float = 2.0,
    event_dense_s: float = 1.0,
    event_sparse_cadence_ms: float = 51.2,
) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Determines which frame indices to write to disk:
      - Periodic background frames
      - Dense coverage around transient onset
      - Sparse coverage during long sweeps / persistent RFI
    """
    selected_set = set()
    frame_events: Dict[int, List[str]] = {}

    # 1. Periodic background
    bg_step = max(1, int(round(background_cadence_s / frame_duration_s)))
    for k in range(0, total_frames, bg_step):
        selected_set.add(k)

    # 2. Event captures
    dense_frames_count = max(1, int(round(event_dense_s / frame_duration_s)))
    sparse_step = max(1, int(round((event_sparse_cadence_ms * 1e-3) / frame_duration_s)))

    for ev in events:
        if ev.is_persistent:
            # Persistent RFI is active in all frames; do not over-select frames for it
            continue

        t_start = ev.t_start_s
        t_end = t_start + ev.duration_s
        k_start = max(0, int(math.floor(t_start / frame_duration_s)))
        k_end = min(total_frames, int(math.ceil(t_end / frame_duration_s)))

        # Dense capture window (around start of event)
        k_dense_end = min(k_end, k_start + dense_frames_count)
        for k in range(k_start, k_dense_end):
            selected_set.add(k)
            frame_events.setdefault(k, []).append(ev.event_id)

        # Sparse capture window for the remaining sweep/duration
        for k in range(k_dense_end, k_end, sparse_step):
            selected_set.add(k)
            frame_events.setdefault(k, []).append(ev.event_id)

    # Attach persistent events to all selected frames
    persistent_evs = [ev.event_id for ev in events if ev.is_persistent]
    for k in selected_set:
        for p_id in persistent_evs:
            frame_events.setdefault(k, []).append(p_id)

    sorted_indices = sorted(selected_set)
    return sorted_indices, frame_events


# ---------------------------------------------------------------------------
# Worker Task: Chunked Frame Rendering & Direct File Writing
# ---------------------------------------------------------------------------
def render_and_write_frame(job: Dict[str, Any]) -> Tuple[int, int, float, float, float]:
    """
    Renders 4-bit complex baseband voltage for one frame and writes .bin directly.
    Returns: (out_idx, frame_idx, mean_power, max_power, clip_fraction)
    """
    out_idx = job["out_idx"]
    frame_idx = job["frame_idx"]
    out_file_path = job["out_file_path"]

    num_ant = job["num_antennas"]
    num_freq = job["num_freq"]
    samples_per_frame = job["samples_per_frame"]
    dt_s = job["dt_s"]
    t0_s = job["t_start_s"]
    freqs_hz = job["freqs_hz"]
    pos_x = job["pos_x"]
    pos_y = job["pos_y"]
    sigma_ant = job["sigma_ant"]
    bandpass = job["bandpass"]
    sun_info = job["sun_info"]
    active_events = job["active_events"]

    rng = np.random.default_rng(job["seed"])
    c_inv = np.float32(1.0 / C_LIGHT)
    two_pi = np.float32(2.0 * np.pi)

    # Chunked rendering to cap peak RAM (~256 samples per chunk)
    chunk_size = 256
    packed_frame = np.empty((samples_per_frame, num_freq, num_ant), dtype=np.uint8)

    total_power_sum = 0.0
    max_power_val = 0.0
    total_clipped = 0
    total_elements = samples_per_frame * num_freq * num_ant * 2

    # Pre-scale noise standard deviation: shape (1, num_freq, num_ant)
    noise_sigma = (sigma_ant[None, None, :] * bandpass[None, :, None]).astype(np.float32)

    f_top_hz = freqs_hz[-1]
    f_top_ghz = f_top_hz / 1e9

    for c0 in range(0, samples_per_frame, chunk_size):
        c1 = min(samples_per_frame, c0 + chunk_size)
        n_chunk = c1 - c0

        # Physical time within window
        t_indices = np.arange(c0, c1, dtype=np.float32)
        t_abs_s = (t0_s + t_indices * dt_s).astype(np.float32)  # shape (n_chunk,)

        # 1. Independent receiver & sky thermal noise
        v_real = rng.normal(0.0, 1.0, size=(n_chunk, num_freq, num_ant)).astype(np.float32) * noise_sigma
        v_imag = rng.normal(0.0, 1.0, size=(n_chunk, num_freq, num_ant)).astype(np.float32) * noise_sigma

        # 2. Sun Emission (Coherent Celestial Source)
        if sun_info is not None and sun_info["amp"] > 0.005:
            sun_amp = np.float32(sun_info["amp"])
            sun_l = np.float32(sun_info["l"])
            sun_m = np.float32(sun_info["m"])

            # Geometric delays: (num_ant,)
            sun_delays = (sun_l * pos_x + sun_m * pos_y) * c_inv
            # Geometric phase: (1, num_freq, num_ant)
            geom_phase = two_pi * (sun_delays[None, None, :] * freqs_hz[None, :, None])
            # Intrinsic phase evolution
            base_phase = two_pi * np.outer(t_abs_s * np.float32(0.005), freqs_hz * np.float32(1e-8))
            total_sun_phase = base_phase[:, :, None] - geom_phase

            v_real += sun_amp * np.cos(total_sun_phase)
            v_imag += sun_amp * np.sin(total_sun_phase)

        # 3. Active Transient Events
        for ev in active_events:
            ev_type = ev["event_type"]
            ev_amp = np.float32(ev["nominal_amp"])

            if ev_type == "frb":
                dm = ev["dm"]
                width_s = np.float32(ev["pulse_width_ms"] * 1e-3)
                t_arrival_top = ev["t_start_s"]
                # Frequency dispersion delay: Delta t = 4148.8 * DM * (f_mhz^-2 - f_top^-2)
                f_ghz = freqs_hz / 1e9
                dm_delays_s = ((K_DM * 1e-6) * dm * (1.0 / (f_ghz ** 2) - 1.0 / (f_top_ghz ** 2))).astype(np.float32)

                # Channel center time: (1, num_freq)
                t_chan_center = t_arrival_top + dm_delays_s[None, :]
                t_diff = t_abs_s[:, None] - t_chan_center  # (n_chunk, num_freq)
                envelope = np.exp(-0.5 * (t_diff / width_s) ** 2).astype(np.float32)

                # Wavefront phase
                l_ev = np.float32(ev["l0"])
                m_ev = np.float32(ev["m0"])
                geom_delays = (l_ev * pos_x + m_ev * pos_y) * c_inv
                geom_phases = two_pi * (geom_delays[None, None, :] * freqs_hz[None, :, None])
                base_phases = two_pi * np.outer(t_abs_s * np.float32(0.02), freqs_hz * np.float32(1e-8))
                tot_phase = base_phases[:, :, None] - geom_phases

                v_real += ev_amp * envelope[:, :, None] * np.cos(tot_phase)
                v_imag += ev_amp * envelope[:, :, None] * np.sin(tot_phase)

            elif ev_type == "pulsar":
                period_s = ev["pulse_period_s"]
                width_s = np.float32(ev["pulse_width_ms"] * 1e-3)
                dm = ev["dm"]
                f_ghz = freqs_hz / 1e9
                dm_delays_s = ((K_DM * 1e-6) * dm * (1.0 / (f_ghz ** 2) - 1.0 / (f_top_ghz ** 2))).astype(np.float32)

                # Determine which pulse numbers overlap this chunk
                t_min = t_abs_s[0] - dm_delays_s[-1] - 3.0 * width_s
                t_max = t_abs_s[-1] + 3.0 * width_s
                k_min = int(math.floor((t_min - ev["t_start_s"]) / period_s))
                k_max = int(math.ceil((t_max - ev["t_start_s"]) / period_s))

                l_ev = np.float32(ev["l0"])
                m_ev = np.float32(ev["m0"])
                geom_delays = (l_ev * pos_x + m_ev * pos_y) * c_inv
                geom_phases = two_pi * (geom_delays[None, None, :] * freqs_hz[None, :, None])
                base_phases = two_pi * np.outer(t_abs_s * np.float32(0.01), freqs_hz * np.float32(1e-8))
                tot_phase = base_phases[:, :, None] - geom_phases

                for p_idx in range(k_min, k_max + 1):
                    t_pulse_0 = ev["t_start_s"] + p_idx * period_s
                    t_chan = t_pulse_0 + dm_delays_s[None, :]
                    t_diff = t_abs_s[:, None] - t_chan
                    env = np.exp(-0.5 * (t_diff / width_s) ** 2).astype(np.float32)
                    v_real += ev_amp * env[:, :, None] * np.cos(tot_phase)
                    v_imag += ev_amp * env[:, :, None] * np.sin(tot_phase)

            elif ev_type == "rfi_narrow":
                chans = ev["channels"] or []
                l_ev = np.float32(ev["l0"])
                m_ev = np.float32(ev["m0"])
                geom_delays = (l_ev * pos_x + m_ev * pos_y) * c_inv
                geom_phases = two_pi * (geom_delays[None, None, :] * freqs_hz[None, :, None])
                base_phases = two_pi * np.outer(t_abs_s * np.float32(0.003), freqs_hz * np.float32(1e-8))
                tot_phase = base_phases[:, :, None] - geom_phases

                for ch in chans:
                    v_real[:, ch, :] += ev_amp * np.cos(tot_phase[:, ch, :])
                    v_imag[:, ch, :] += ev_amp * np.sin(tot_phase[:, ch, :])

            elif ev_type == "rfi_leo":
                t_rel = t_abs_s - ev["t_start_s"]
                l_t = (np.float32(ev["l0"]) + np.float32(ev["drift_dl"]) * t_rel).astype(np.float32)
                m_t = (np.float32(ev["m0"]) + np.float32(ev["drift_dm"]) * t_rel).astype(np.float32)

                delays = (np.outer(l_t, pos_x) + np.outer(m_t, pos_y)) * c_inv
                geom_phases = two_pi * (delays[:, None, :] * freqs_hz[None, :, None])
                base_phases = two_pi * np.outer(t_abs_s * np.float32(0.05), freqs_hz * np.float32(1e-8))
                tot_phase = base_phases[:, :, None] - geom_phases

                v_real += ev_amp * np.cos(tot_phase)
                v_imag += ev_amp * np.sin(tot_phase)

        # Compute power diagnostics before quantization
        pwr_chunk = v_real ** 2 + v_imag ** 2
        total_power_sum += float(np.sum(pwr_chunk))
        max_power_val = max(max_power_val, float(np.max(pwr_chunk)))

        # Count clipped samples before clipping
        total_clipped += int(np.sum(np.abs(v_real) > 7.5)) + int(np.sum(np.abs(v_imag) > 7.5))

        # Quantize to 4-bit integer [-7, +7]
        r_quant = np.clip(np.round(v_real), -7, 7).astype(np.int8)
        i_quant = np.clip(np.round(v_imag), -7, 7).astype(np.int8)

        r_nibble = (r_quant & 0x0F).astype(np.uint8)
        i_nibble = ((i_quant & 0x0F) << 4).astype(np.uint8)

        packed_frame[c0:c1] = r_nibble | i_nibble

    # Write directly to .bin file with uint32(0) metadata size header
    with open(out_file_path, "wb") as f:
        np.uint32(0).tofile(f)
        packed_frame.tofile(f)

    mean_power = total_power_sum / (samples_per_frame * num_freq * num_ant)
    clip_frac = total_clipped / total_elements

    return out_idx, frame_idx, mean_power, max_power_val, clip_frac


# ---------------------------------------------------------------------------
# Analytic Full-Timeline Lightcurve (10 Hz Cadence)
# ---------------------------------------------------------------------------
def compute_analytic_lightcurve(
    duration_s: float,
    noise_power_base: float,
    sun_power_series: np.ndarray,
    events: List[SimulatedEvent],
    freqs_hz: np.ndarray,
    cadence_hz: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a 10 Hz analytic power lightcurve for the entire 300 s window.
    P(t) = P_noise + P_sun(t) + sum_events P_ev(t).
    """
    num_points = int(round(duration_s * cadence_hz))
    t_points = np.linspace(0.0, duration_s, num_points, endpoint=False)
    p_total = np.full(num_points, noise_power_base, dtype=np.float64)

    # Add Sun power
    p_total += sun_power_series

    f_top_hz = freqs_hz[-1]
    f_top_ghz = f_top_hz / 1e9
    f_ghz = freqs_hz / 1e9

    for ev in events:
        amp_sq = ev.nominal_amp ** 2
        t0 = ev.t_start_s
        dur = ev.duration_s

        if ev.event_type == "frb":
            dm = ev.dm
            width_s = ev.pulse_width_ms * 1e-3
            dm_delays_s = (K_DM * 1e-6) * dm * (1.0 / (f_ghz ** 2) - 1.0 / (f_top_ghz ** 2))
            # Average envelope^2 across frequency channels at each lightcurve time
            for idx, t in enumerate(t_points):
                if t0 - 0.5 <= t <= t0 + dur + 0.5:
                    t_chan = t0 + dm_delays_s
                    env_sq = np.exp(-((t - t_chan) / width_s) ** 2)
                    p_total[idx] += amp_sq * float(np.mean(env_sq))

        elif ev.event_type == "pulsar":
            period_s = ev.pulse_period_s
            width_s = ev.pulse_width_ms * 1e-3
            duty_cycle = min(1.0, width_s / period_s)
            mask = (t_points >= t0) & (t_points <= t0 + dur)
            p_total[mask] += amp_sq * duty_cycle

        elif ev.event_type == "rfi_narrow":
            frac_chans = len(ev.channels or [1]) / len(freqs_hz)
            if ev.is_persistent:
                p_total += amp_sq * frac_chans
            else:
                mask = (t_points >= t0) & (t_points <= t0 + dur)
                p_total[mask] += amp_sq * frac_chans

        elif ev.event_type == "rfi_leo":
            mask = (t_points >= t0) & (t_points <= t0 + dur)
            p_total[mask] += amp_sq

    return t_points, p_total


# ---------------------------------------------------------------------------
# Main Generation Controller
# ---------------------------------------------------------------------------
def generate_5min_window(
    utc_hour: int = 15,
    date_str: str = "2026-03-20",
    duration_s: float = 300.0,
    antennas: int = 64,
    num_freq: int = 336,
    samples_per_frame: int = 1536,
    background_cadence_s: float = 2.0,
    event_dense_s: float = 1.0,
    event_sparse_cadence_ms: float = 51.2,
    num_events: int = 3,
    event_types: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    workers: int = 16,
    seed: Optional[int] = None,
    sun_activity: str = "quiet",
    persistent_rfi_channels: Optional[List[int]] = None,
    persistent_rfi_amp: float = 7.0,
):
    """Generates the full 5-minute window dataset and metadata."""
    if seed is None:
        seed = 1000 * utc_hour + 42

    dt_start = datetime.datetime.strptime(f"{date_str} {utc_hour:02d}:00:00", "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )

    out_base_name = file_name or f"win{utc_hour:02d}UTC_{antennas}ant"
    target_dir = (out_dir or Path(f"./dumps_5min_window_{utc_hour:02d}UTC")).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dt_s = FPGA_TIME_RESOLUTION_US * 1e-6
    frame_duration_s = samples_per_frame * dt_s
    total_window_frames = int(math.floor(duration_s / frame_duration_s))

    freqs_hz = (DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ) * 1e6
    pos_x, pos_y = get_charts_antenna_positions(num_antennas=antennas)

    # Initialize noise model
    noise_model = ChartsNoiseModel()
    sigma_ant_base = noise_model.generate_antenna_gain_dispersion(num_antennas=antennas, seed=seed)
    bandpass = noise_model.generate_bandpass_shape(freqs_hz)

    # Determine Sun status across the window
    # Sample Sun position at start, middle, and end
    dt_mid = dt_start + datetime.timedelta(seconds=duration_s * 0.5)
    sun_state = noise_model.system_temperature(400.0, utc_dt=dt_mid, include_sun=True, sun_activity=sun_activity)
    sun_el = float(sun_state["sun_elevation_deg"])
    sun_amp = float(noise_model.temp_to_adc_sigma(sun_state["t_sun_pb"]))
    sun_l = float(sun_state["sun_l"])
    sun_m = float(sun_state["sun_m"])

    sun_is_up = sun_el > 0.0
    sun_info = (
        {"l": sun_l, "m": sun_m, "amp": sun_amp, "el_deg": sun_el, "t_pb_k": float(sun_state["t_sun_pb"])}
        if sun_is_up
        else None
    )

    # Base thermal noise ADC standard deviation
    t_noise_incoherent = float(sun_state["t_noise_incoherent"])
    sigma_noise_base = float(noise_model.temp_to_adc_sigma(t_noise_incoherent))
    sigma_ant = sigma_ant_base * sigma_noise_base

    # Schedule transient events + persistent site RFI
    events = schedule_random_events(
        duration_s=duration_s,
        num_events=num_events,
        allowed_types=event_types,
        seed=seed + 77,
        persistent_rfi_channels=persistent_rfi_channels,
        persistent_rfi_amp=persistent_rfi_amp,
    )

    # Select frames to write
    written_indices, frame_events_map = build_frame_selection_schedule(
        total_frames=total_window_frames,
        frame_duration_s=frame_duration_s,
        events=events,
        background_cadence_s=background_cadence_s,
        event_dense_s=event_dense_s,
        event_sparse_cadence_ms=event_sparse_cadence_ms,
    )

    num_written = len(written_indices)

    print("=" * 78)
    print(f" CHARTS 5-MINUTE REALISTIC WINDOW GENERATION ({utc_hour:02d}:00 UTC)")
    print("=" * 78)
    print(f" Target Directory       : {target_dir}")
    print(f" Output File Base Name  : {out_base_name}_%07d.bin")
    print(f" Window Start UTC       : {dt_start.isoformat()}")
    print(f" Duration               : {duration_s:.1f} s ({total_window_frames:,} total physical frames)")
    print(f" Decimated Frames to Write: {num_written:,} frames (~{num_written * 31.5 / 1024:.2f} GB)")
    print(f" Background Cadence     : Every {background_cadence_s:.1f} s")
    print(f" Solar Elevation        : {sun_el:+.2f} deg ({'SUN UP - COHERENT EMISSION' if sun_is_up else 'SUN BELOW HORIZON'})")
    if sun_is_up:
        print(f" Sun Antenna Temp (PB)  : {sun_state['t_sun_pb']:.2f} K (amp = {sun_amp:.3f} LSB, l={sun_l:+.3f}, m={sun_m:+.3f})")
    print(f" Base Incoherent Noise  : T_noise = {t_noise_incoherent:.2f} K -> sigma = {sigma_noise_base:.3f} LSB")
    print(f" Injected Events Count  : {len(events)}")
    for ev in events:
        print(f"   * [{ev.event_id}] t={ev.t_start_s:.1f}s dur={ev.duration_s:.1f}s: {ev.description}")
    print(f" Multiprocessing Workers: {workers}")
    print("=" * 78)

    # Prepare job payloads
    jobs = []
    for out_idx, frame_idx in enumerate(written_indices):
        t_start_s = frame_idx * frame_duration_s
        out_bin_file = target_dir / f"{out_base_name}_{out_idx:07d}.bin"

        # Determine events active in this frame
        active_evs = []
        ev_ids = frame_events_map.get(frame_idx, [])
        for ev in events:
            if ev.event_id in ev_ids:
                active_evs.append(ev.to_dict())

        job = {
            "out_idx": out_idx,
            "frame_idx": frame_idx,
            "out_file_path": str(out_bin_file),
            "num_antennas": antennas,
            "num_freq": num_freq,
            "samples_per_frame": samples_per_frame,
            "dt_s": dt_s,
            "t_start_s": t_start_s,
            "freqs_hz": freqs_hz,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "sigma_ant": sigma_ant,
            "bandpass": bandpass,
            "sun_info": sun_info,
            "active_events": active_evs,
            "seed": seed + frame_idx,
        }
        jobs.append(job)

    # Execute frame generation via multiprocessing pool
    t0_gen = time.perf_counter()
    pool_workers = min(workers, mp.cpu_count() or 4)

    print(f"\n>>> Dispatching {num_written} frames across {pool_workers} workers...")
    stats_list = []
    if pool_workers > 1:
        with mp.Pool(processes=pool_workers, maxtasksperchild=100) as pool:
            for res in pool.imap_unordered(render_and_write_frame, jobs, chunksize=1):
                stats_list.append(res)
                if len(stats_list) % max(1, num_written // 10) == 0 or len(stats_list) == num_written:
                    pct = (len(stats_list) / num_written) * 100.0
                    print(f"    Progress: {len(stats_list):4d}/{num_written} frames ({pct:.1f}%)")
    else:
        for job in jobs:
            res = render_and_write_frame(job)
            stats_list.append(res)

    t_gen_s = time.perf_counter() - t0_gen
    print(f">>> Completed frame rendering in {t_gen_s:.2f} s ({t_gen_s / max(1, num_written):.3f} s/frame)")

    # Sort stats by out_idx
    stats_list.sort(key=lambda x: x[0])

    # -----------------------------------------------------------------------
    # Save Metadata & 10 Hz Analytic Lightcurve
    # -----------------------------------------------------------------------
    print("\n>>> Computing 10 Hz full-timeline analytic lightcurve...")
    noise_pwr_base = float(np.mean(2.0 * (sigma_ant ** 2)) * np.mean(bandpass ** 2))
    sun_pwr_series = np.full(int(duration_s * 10.0), sun_amp ** 2 if sun_is_up else 0.0)

    lc_t, lc_power = compute_analytic_lightcurve(
        duration_s=duration_s,
        noise_power_base=noise_pwr_base,
        sun_power_series=sun_pwr_series,
        events=events,
        freqs_hz=freqs_hz,
        cadence_hz=10.0,
    )

    meta_h5_path = target_dir / f"{out_base_name}_meta.h5"
    events_json_path = target_dir / f"{out_base_name}_events.json"

    # Save event list JSON
    with open(events_json_path, "w") as f:
        json.dump([ev.to_dict() for ev in events], f, indent=2)

    # Save HDF5 metadata
    written_frames_arr = np.array(written_indices, dtype=np.int32)
    written_out_arr = np.arange(num_written, dtype=np.int32)
    written_times_s = written_frames_arr * frame_duration_s
    written_powers = np.array([st[2] for st in stats_list], dtype=np.float32)
    written_clip = np.array([st[4] for st in stats_list], dtype=np.float32)

    with h5py.File(meta_h5_path, "w") as hf:
        hf.attrs["window_name"] = out_base_name
        hf.attrs["utc_hour"] = utc_hour
        hf.attrs["date"] = date_str
        hf.attrs["start_time_iso"] = dt_start.isoformat()
        hf.attrs["duration_s"] = duration_s
        hf.attrs["num_antennas"] = antennas
        hf.attrs["num_freq"] = num_freq
        hf.attrs["samples_per_frame"] = samples_per_frame
        hf.attrs["frame_duration_ms"] = frame_duration_s * 1000.0
        hf.attrs["total_physical_frames"] = total_window_frames
        hf.attrs["written_frames_count"] = num_written
        hf.attrs["sun_is_up"] = bool(sun_is_up)
        hf.attrs["sun_elevation_deg"] = sun_el
        hf.attrs["sun_antenna_temp_k"] = float(sun_state["t_sun_pb"])
        hf.attrs["t_noise_incoherent_k"] = t_noise_incoherent
        hf.attrs["sigma_noise_base"] = sigma_noise_base
        hf.attrs["site_lat_deg"] = CHARTS_LATITUDE_DEG
        hf.attrs["site_lon_deg"] = CHARTS_LONGITUDE_DEG
        hf.attrs["site_alt_m"] = CHARTS_ALTITUDE_M

        hf.create_dataset("lightcurve_t_s", data=lc_t, dtype=np.float32)
        hf.create_dataset("lightcurve_power", data=lc_power, dtype=np.float32)
        hf.create_dataset("written_out_index", data=written_out_arr, dtype=np.int32)
        hf.create_dataset("written_window_frame", data=written_frames_arr, dtype=np.int32)
        hf.create_dataset("written_time_s", data=written_times_s, dtype=np.float32)
        hf.create_dataset("written_mean_power", data=written_powers, dtype=np.float32)
        hf.create_dataset("written_clip_fraction", data=written_clip, dtype=np.float32)

    print(f"Saved metadata HDF5 : {meta_h5_path}")
    print(f"Saved events JSON   : {events_json_path}")
    print(f"\nFRAMES_WRITTEN={num_written}")
    print("=" * 78)

    return target_dir, out_base_name, num_written


def main():
    parser = argparse.ArgumentParser(description="CHARTS Realistic 5-Minute Window Generator")
    parser.add_argument("--utc-hour", type=int, default=15, choices=[15, 3], help="UTC hour (15 = Day, 3 = Night)")
    parser.add_argument("--date", type=str, default="2026-03-20", help="Observation date YYYY-MM-DD")
    parser.add_argument("--duration-s", type=float, default=300.0, help="Window duration in seconds (default: 300)")
    parser.add_argument("--antennas", type=int, default=64, help="Number of antennas (default: 64)")
    parser.add_argument("--num-freq", type=int, default=336, help="Frequency channels (default: 336)")
    parser.add_argument("--samples-per-frame", type=int, default=1536, help="Samples per frame (default: 1536)")
    parser.add_argument("--background-cadence-s", type=float, default=2.0, help="Background frame cadence (s)")
    parser.add_argument("--event-dense-s", type=float, default=1.0, help="Dense capture duration around events (s)")
    parser.add_argument("--event-sparse-cadence-ms", type=float, default=51.2, help="Sparse capture cadence (ms)")
    parser.add_argument("--num-events", type=int, default=3, help="Number of transient events (default: 3)")
    parser.add_argument("--events", type=str, default=None, help="Comma-separated event types (frb,pulsar,rfi_narrow,rfi_leo)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--file-name", type=str, default=None, help="Base file name")
    parser.add_argument("--workers", type=int, default=16, help="Multiprocessing workers")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--sun-activity", type=str, default="quiet", choices=["quiet", "moderate", "active"])
    parser.add_argument(
        "--persistent-rfi-channels",
        type=str,
        default="94,133,147",
        help="Comma-separated channel indices for continuous site RFI (default: '94,133,147' -> 328.2, 339.9, 344.1 MHz; set 'none' to disable)",
    )
    parser.add_argument(
        "--persistent-rfi-freqs",
        type=str,
        default=None,
        help="Comma-separated frequencies in MHz for continuous site RFI (e.g. '328.2,339.9,344.1')",
    )
    parser.add_argument(
        "--persistent-rfi-amp",
        type=float,
        default=7.0,
        help="Amplitude of persistent site RFI in LSB (default: 7.0 LSB)",
    )
    args = parser.parse_args()

    ev_types = [s.strip() for s in args.events.split(",")] if args.events else None

    # Parse persistent RFI channels
    persistent_chans = None
    if args.persistent_rfi_freqs:
        freq_list = [float(x.strip()) for x in args.persistent_rfi_freqs.split(",") if x.strip()]
        persistent_chans = [
            int(round((f - DEFAULT_FREQUENCY_START_MHZ) / CHARTS_CHANNEL_WIDTH_MHZ))
            for f in freq_list
        ]
    elif args.persistent_rfi_channels and args.persistent_rfi_channels.lower() != "none":
        persistent_chans = [
            int(x.strip()) for x in args.persistent_rfi_channels.split(",") if x.strip().isdigit()
        ]

    generate_5min_window(
        utc_hour=args.utc_hour,
        date_str=args.date,
        duration_s=args.duration_s,
        antennas=args.antennas,
        num_freq=args.num_freq,
        samples_per_frame=args.samples_per_frame,
        background_cadence_s=args.background_cadence_s,
        event_dense_s=args.event_dense_s,
        event_sparse_cadence_ms=args.event_sparse_cadence_ms,
        num_events=args.num_events,
        event_types=ev_types,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        file_name=args.file_name,
        workers=args.workers,
        seed=args.seed,
        sun_activity=args.sun_activity,
        persistent_rfi_channels=persistent_chans,
        persistent_rfi_amp=args.persistent_rfi_amp,
    )


if __name__ == "__main__":
    main()
