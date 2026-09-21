#!/usr/bin/env python3
"""
CHARTS Pipeline Video Generator
================================
Generates 30-second MP4 videos from CHARTS pipeline outputs:
  1. Baseband 8x8 spectrogram video  — frequency power per antenna over time
  2. Correlator matrix video         — 64x64 visibility matrix evolution
  3. Beam tracker output video       — formed beam power over time/frequency

Uses matplotlib animation with FFMpegWriter to produce .mp4 files.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    LOCAL_FREQUENCY_CHANNELS,
)

# 4-bit two's complement LUT
INT4_LUT = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Baseband frame loading
# ---------------------------------------------------------------------------
def load_baseband_frame(
    bin_path: Path,
    num_antennas: int = 64,
    num_freq: int = 336,
    samples_per_frame: int = 1536,
) -> np.ndarray:
    """
    Loads a raw int4x2 baseband frame and returns complex voltage array.
    Returns shape: (samples_per_frame, num_freq, num_antennas) complex128
    """
    raw = np.fromfile(bin_path, dtype=np.uint8)
    # Skip 4-byte metadata header + metadata payload
    if raw.size < 4:
        raise ValueError(f"File too small: {bin_path}")
    meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload = raw[4 + meta_size:]

    expected_bytes = samples_per_frame * num_freq * num_antennas
    if payload.size < expected_bytes:
        raise ValueError(
            f"Payload {payload.size} < expected {expected_bytes} in {bin_path}"
        )

    # Each byte packs two 4-bit values: high nibble = real, low nibble = imag
    packed = payload[:expected_bytes]
    real_idx = (packed >> 4) & 0x0F
    imag_idx = packed & 0x0F
    voltages = INT4_LUT[real_idx] + 1j * INT4_LUT[imag_idx]
    return voltages.reshape(samples_per_frame, num_freq, num_antennas)


def compute_baseband_spectrogram(
    voltages: np.ndarray,
    num_antennas: int = 64,
) -> np.ndarray:
    """
    Computes per-antenna frequency power (averaged over time).
    Input: (samples, freq, antennas) complex
    Returns: (num_antennas, num_freq) power in dB
    """
    power = np.mean(np.abs(voltages) ** 2, axis=0)  # (freq, antennas)
    power_db = 10.0 * np.log10(power.T + 1e-12)  # (antennas, freq)
    return power_db


# ---------------------------------------------------------------------------
# Correlator frame loading
# ---------------------------------------------------------------------------
def load_correlator_frame(
    bin_path: Path,
    num_elements: int = 64,
    num_channels: int = 336,
    polarizations: int = 2,
) -> np.ndarray:
    """
    Loads a cudaCorrelatorAstron binary dump.
    Returns: (num_channels, num_elements, num_elements) complex128
    """
    from inspect_correlator_dump import load_astron_correlator_dump
    return load_astron_correlator_dump(
        bin_path, num_elements=num_elements, num_channels=num_channels,
        polarizations=polarizations,
    )


# ---------------------------------------------------------------------------
# Beam tracker frame loading
# ---------------------------------------------------------------------------
def load_beam_tracker_frame(
    bin_path: Path,
    samples_per_data_set: int = 1536,
    num_freq: int = 336,
    max_beams: int = 8,
) -> np.ndarray:
    """
    Loads a beam tracker output frame (complex float32 formed beams).
    Layout: [time][freq][beam] of float2 (real, imag)
    Returns: (samples_per_data_set, num_freq, max_beams) complex128
    """
    raw = np.fromfile(bin_path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File too small: {bin_path}")
    meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload = raw[4 + meta_size:]

    expected_floats = samples_per_data_set * num_freq * max_beams * 2
    expected_bytes = expected_floats * 4
    if payload.size < expected_bytes:
        raise ValueError(
            f"Payload {payload.size} < expected {expected_bytes} in {bin_path}"
        )

    floats = np.frombuffer(payload[:expected_bytes], dtype="<f4")
    # Reshape to (time, freq, beams, 2) then combine to complex
    shaped = floats.reshape(samples_per_data_set, num_freq, max_beams, 2)
    return shaped[..., 0].astype(np.float64) + 1j * shaped[..., 1].astype(np.float64)


# ---------------------------------------------------------------------------
# Video 1: Baseband 8x8 Spectrogram
# ---------------------------------------------------------------------------
def generate_baseband_spectrogram_video(
    window_dir: Path,
    base_name: str,
    output_path: Path,
    num_antennas: int = 64,
    num_freq: int = 336,
    samples_per_frame: int = 1536,
    duration_s: float = 30.0,
    fps: int = 10,
    max_frames: Optional[int] = None,
):
    """
    Generates a video showing the 8x8 antenna grid, each cell showing
    the frequency power spectrum for that antenna, evolving over frames.
    """
    bin_files = sorted(window_dir.glob(f"{base_name}_*.bin"))
    if not bin_files:
        print(f"[WARNING] No baseband files found matching {base_name}_*.bin in {window_dir}")
        return

    total_available = len(bin_files)
    if max_frames:
        bin_files = bin_files[:max_frames]

    # Subsample to fit duration
    total_video_frames = int(duration_s * fps)
    if len(bin_files) > total_video_frames:
        step = len(bin_files) / total_video_frames
        indices = [int(i * step) for i in range(total_video_frames)]
        bin_files = [bin_files[i] for i in indices]
    else:
        total_video_frames = len(bin_files)

    print(f"  Baseband video: {total_video_frames} frames from {total_available} available")

    # Compute global color scale from first frame
    first_volts = load_baseband_frame(bin_files[0], num_antennas, num_freq, samples_per_frame)
    first_spec = compute_baseband_spectrogram(first_volts, num_antennas)
    vmin = np.percentile(first_spec, 2)
    vmax = np.percentile(first_spec, 98)

    freq_mhz = DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ

    # Setup figure: 8x8 grid
    fig, axes = plt.subplots(8, 8, figsize=(20, 12), facecolor="#0a0a1a")
    fig.suptitle("CHARTS Baseband Frequency Power — 8×8 Antenna Grid", fontsize=16, color="white", y=0.98)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.94, bottom=0.06, left=0.04, right=0.96)

    ims = []
    for ant in range(num_antennas):
        row = 7 - (ant >> 3)  # Row 7 at top
        col = ant & 7
        ax = axes[row, col]
        ax.set_facecolor("#111122")
        im = ax.imshow(
            first_spec[ant].reshape(-1, 1).T,
            aspect="auto",
            origin="lower",
            extent=[0, 1, freq_mhz[0], freq_mhz[-1]],
            vmin=vmin, vmax=vmax,
            cmap="inferno",
        )
        ims.append(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.92, f"A{ant}", transform=ax.transAxes, fontsize=5,
                color="white", ha="center", va="top", alpha=0.7)

    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12, color="cyan")

    def update(frame_idx):
        volts = load_baseband_frame(bin_files[frame_idx], num_antennas, num_freq, samples_per_frame)
        spec = compute_baseband_spectrogram(volts, num_antennas)
        for ant in range(num_antennas):
            ims[ant].set_data(spec[ant].reshape(-1, 1).T)
        t_s = frame_idx * (duration_s / max(1, total_video_frames))
        time_text.set_text(f"Frame {frame_idx}/{total_video_frames}  |  t ≈ {t_s:.1f} s")
        return ims + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=total_video_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Video 2: Correlator Matrix
# ---------------------------------------------------------------------------
def generate_correlator_video(
    corr_dir: Path,
    corr_name: str,
    output_path: Path,
    num_elements: int = 64,
    num_channels: int = 336,
    duration_s: float = 30.0,
    fps: int = 10,
    freq_channel_idx: Optional[int] = None,
    max_frames: Optional[int] = None,
):
    """
    Generates a video showing the 64x64 visibility matrix magnitude
    at a selected frequency channel, evolving over frames.
    """
    bin_files = sorted(corr_dir.glob(f"{corr_name}_*.bin"))
    if not bin_files:
        print(f"[WARNING] No correlator files found matching {corr_name}_*.bin in {corr_dir}")
        return

    total_available = len(bin_files)
    if max_frames:
        bin_files = bin_files[:max_frames]

    total_video_frames = int(duration_s * fps)
    if len(bin_files) > total_video_frames:
        step = len(bin_files) / total_video_frames
        indices = [int(i * step) for i in range(total_video_frames)]
        bin_files = [bin_files[i] for i in indices]
    else:
        total_video_frames = len(bin_files)

    if freq_channel_idx is None:
        freq_channel_idx = num_channels // 2

    freq_mhz = DEFAULT_FREQUENCY_START_MHZ + freq_channel_idx * CHARTS_CHANNEL_WIDTH_MHZ

    print(f"  Correlator video: {total_video_frames} frames from {total_available} available, channel {freq_channel_idx} ({freq_mhz:.1f} MHz)")

    # Load first frame for color scale
    first_vis = load_correlator_frame(bin_files[0], num_elements, num_channels)
    first_mag = np.abs(first_vis[freq_channel_idx])
    vmin = 0.0
    vmax = np.percentile(first_mag, 99)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0a0a1a")
    fig.suptitle(
        f"CHARTS Correlator Visibility Matrix — Channel {freq_channel_idx} ({freq_mhz:.1f} MHz)",
        fontsize=14, color="white",
    )

    # Magnitude
    ax_mag = axes[0]
    ax_mag.set_facecolor("#111122")
    im_mag = ax_mag.imshow(first_mag, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax_mag.set_title("|V_ij| (Magnitude)", color="white", fontsize=12)
    ax_mag.set_xlabel("Antenna j", color="white")
    ax_mag.set_ylabel("Antenna i", color="white")
    ax_mag.tick_params(colors="white")
    plt.colorbar(im_mag, ax=ax_mag, shrink=0.8)

    # Phase
    ax_ph = axes[1]
    ax_ph.set_facecolor("#111122")
    first_phase = np.angle(first_vis[freq_channel_idx])
    im_ph = ax_ph.imshow(first_phase, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax_ph.set_title("arg(V_ij) (Phase)", color="white", fontsize=12)
    ax_ph.set_xlabel("Antenna j", color="white")
    ax_ph.set_ylabel("Antenna i", color="white")
    ax_ph.tick_params(colors="white")
    plt.colorbar(im_ph, ax=ax_ph, shrink=0.8)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12, color="cyan")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    def update(frame_idx):
        vis = load_correlator_frame(bin_files[frame_idx], num_elements, num_channels)
        mag = np.abs(vis[freq_channel_idx])
        phase = np.angle(vis[freq_channel_idx])
        im_mag.set_data(mag)
        im_ph.set_data(phase)
        t_s = frame_idx * (duration_s / max(1, total_video_frames))
        time_text.set_text(f"Frame {frame_idx}/{total_video_frames}  |  t ≈ {t_s:.1f} s")
        return [im_mag, im_ph, time_text]

    anim = animation.FuncAnimation(fig, update, frames=total_video_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Video 3: Beam Tracker Output
# ---------------------------------------------------------------------------
def generate_beam_tracker_video(
    tracker_dir: Path,
    tracker_name: str,
    output_path: Path,
    samples_per_data_set: int = 1536,
    num_freq: int = 336,
    max_beams: int = 8,
    duration_s: float = 30.0,
    fps: int = 10,
    max_frames: Optional[int] = None,
):
    """
    Generates a video showing beam tracker formed-beam power.
    Top panel: beam power vs frequency for all beams (waterfall-style).
    Bottom panel: integrated beam power lightcurves.
    """
    bin_files = sorted(tracker_dir.glob(f"{tracker_name}_*.bin"))
    if not bin_files:
        print(f"[WARNING] No tracker files found matching {tracker_name}_*.bin in {tracker_dir}")
        return

    total_available = len(bin_files)
    if max_frames:
        bin_files = bin_files[:max_frames]

    total_video_frames = int(duration_s * fps)
    if len(bin_files) > total_video_frames:
        step = len(bin_files) / total_video_frames
        indices = [int(i * step) for i in range(total_video_frames)]
        bin_files = [bin_files[i] for i in indices]
    else:
        total_video_frames = len(bin_files)

    print(f"  Tracker video: {total_video_frames} frames from {total_available} available, {max_beams} beams")

    freq_mhz = DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ

    # Beam colors
    beam_colors = plt.cm.tab10(np.linspace(0, 1, max_beams))

    # Load first frame for setup
    first_beams = load_beam_tracker_frame(bin_files[0], samples_per_data_set, num_freq, max_beams)
    first_power = np.mean(np.abs(first_beams) ** 2, axis=0)  # (freq, beams)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="#0a0a1a",
                              gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle("CHARTS Beam Tracker Output — Formed Beam Power", fontsize=14, color="white")

    # Top: Power vs Frequency per beam
    ax_spec = axes[0]
    ax_spec.set_facecolor("#111122")
    lines = []
    for b in range(max_beams):
        line, = ax_spec.plot(
            freq_mhz, 10 * np.log10(first_power[:, b] + 1e-12),
            color=beam_colors[b], linewidth=1.0, label=f"Beam {b}", alpha=0.85,
        )
        lines.append(line)
    ax_spec.set_xlabel("Frequency (MHz)", color="white")
    ax_spec.set_ylabel("Power (dB)", color="white")
    ax_spec.legend(loc="upper right", fontsize=8, ncol=4, facecolor="#1a1a2e", labelcolor="white")
    ax_spec.tick_params(colors="white")
    ax_spec.grid(True, alpha=0.2, color="gray")

    # Bottom: Integrated power bar chart
    ax_bar = axes[1]
    ax_bar.set_facecolor("#111122")
    integrated = np.sum(first_power, axis=0)  # (beams,)
    bars = ax_bar.bar(
        range(max_beams), 10 * np.log10(integrated + 1e-12),
        color=beam_colors, edgecolor="white", linewidth=0.5,
    )
    ax_bar.set_xlabel("Beam Index", color="white")
    ax_bar.set_ylabel("Integrated Power (dB)", color="white")
    ax_bar.set_xticks(range(max_beams))
    ax_bar.set_xticklabels([f"B{b}" for b in range(max_beams)], color="white")
    ax_bar.tick_params(colors="white")
    ax_bar.grid(True, alpha=0.2, color="gray", axis="y")

    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12, color="cyan")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    def update(frame_idx):
        beams = load_beam_tracker_frame(bin_files[frame_idx], samples_per_data_set, num_freq, max_beams)
        power = np.mean(np.abs(beams) ** 2, axis=0)  # (freq, beams)
        for b in range(max_beams):
            lines[b].set_ydata(10 * np.log10(power[:, b] + 1e-12))
        integrated = np.sum(power, axis=0)
        for b, bar in enumerate(bars):
            bar.set_height(10 * np.log10(integrated[b] + 1e-12))
        t_s = frame_idx * (duration_s / max(1, total_video_frames))
        time_text.set_text(f"Frame {frame_idx}/{total_video_frames}  |  t ≈ {t_s:.1f} s")
        return lines + list(bars) + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=total_video_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CHARTS Pipeline Video Generator")
    subparsers = parser.add_subparsers(dest="command", help="Video type to generate")

    # --- baseband ---
    p_bb = subparsers.add_parser("baseband", help="Baseband 8x8 spectrogram video")
    p_bb.add_argument("--window-dir", type=str, required=True)
    p_bb.add_argument("--base-name", type=str, required=True)
    p_bb.add_argument("--output", type=str, required=True)
    p_bb.add_argument("--antennas", type=int, default=64)
    p_bb.add_argument("--num-freq", type=int, default=336)
    p_bb.add_argument("--samples-per-frame", type=int, default=1536)
    p_bb.add_argument("--duration-s", type=float, default=30.0)
    p_bb.add_argument("--fps", type=int, default=10)
    p_bb.add_argument("--max-frames", type=int, default=None)

    # --- correlator ---
    p_corr = subparsers.add_parser("correlator", help="Correlator matrix video")
    p_corr.add_argument("--corr-dir", type=str, required=True)
    p_corr.add_argument("--corr-name", type=str, required=True)
    p_corr.add_argument("--output", type=str, required=True)
    p_corr.add_argument("--num-elements", type=int, default=64)
    p_corr.add_argument("--num-channels", type=int, default=336)
    p_corr.add_argument("--duration-s", type=float, default=30.0)
    p_corr.add_argument("--fps", type=int, default=10)
    p_corr.add_argument("--freq-channel", type=int, default=None)
    p_corr.add_argument("--max-frames", type=int, default=None)

    # --- tracker ---
    p_trk = subparsers.add_parser("tracker", help="Beam tracker output video")
    p_trk.add_argument("--tracker-dir", type=str, required=True)
    p_trk.add_argument("--tracker-name", type=str, required=True)
    p_trk.add_argument("--output", type=str, required=True)
    p_trk.add_argument("--samples-per-data-set", type=int, default=1536)
    p_trk.add_argument("--num-freq", type=int, default=336)
    p_trk.add_argument("--max-beams", type=int, default=8)
    p_trk.add_argument("--duration-s", type=float, default=30.0)
    p_trk.add_argument("--fps", type=int, default=10)
    p_trk.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    if args.command == "baseband":
        generate_baseband_spectrogram_video(
            window_dir=Path(args.window_dir),
            base_name=args.base_name,
            output_path=Path(args.output),
            num_antennas=args.antennas,
            num_freq=args.num_freq,
            samples_per_frame=args.samples_per_frame,
            duration_s=args.duration_s,
            fps=args.fps,
            max_frames=args.max_frames,
        )
    elif args.command == "correlator":
        generate_correlator_video(
            corr_dir=Path(args.corr_dir),
            corr_name=args.corr_name,
            output_path=Path(args.output),
            num_elements=args.num_elements,
            num_channels=args.num_channels,
            duration_s=args.duration_s,
            fps=args.fps,
            freq_channel_idx=args.freq_channel,
            max_frames=args.max_frames,
        )
    elif args.command == "tracker":
        generate_beam_tracker_video(
            tracker_dir=Path(args.tracker_dir),
            tracker_name=args.tracker_name,
            output_path=Path(args.output),
            samples_per_data_set=args.samples_per_data_set,
            num_freq=args.num_freq,
            max_beams=args.max_beams,
            duration_s=args.duration_s,
            fps=args.fps,
            max_frames=args.max_frames,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
