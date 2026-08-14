#!/usr/bin/env python3
"""Compare integrated CPU and GPU visibility phases.

The selected frames are coherently integrated (complex visibilities are summed)
before plotting. Exactly 32 images are written: one phase plot per antenna.
Each plot contains that antenna's 31 baselines, with CPU and GPU overlaid.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Kotekan CHARTS parameters
NUM_ELEMENTS = 32
NUM_LOCAL_FREQ = 672
SAMPLES_PER_DATA_SET = 15360

# Kernel parameters
NR_POLARIZATIONS = 2
NR_RECEIVERS = NUM_ELEMENTS // NR_POLARIZATIONS
NUM_BASELINES = NR_RECEIVERS * (NR_RECEIVERS + 1) // 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", type=Path, default=Path("/data/corr/network"))
    parser.add_argument("--corr-dir", type=Path, default=Path("/data/corr/host_corr2"))
    parser.add_argument(
        "--num-files", type=int, default=1,
        help="number of matching frames to integrate (0 means all)",
    )
    parser.add_argument(
        "--start-index", type=int, default=None,
        help="first frame index to process (default: lowest common index)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("corr_cpu_gpu_plots"),
        help="directory in which plots are written",
    )
    parser.add_argument(
        "--dpi", type=int, default=100,
        help="resolution of saved PNGs (default: 100)",
    )
    parser.add_argument("--show", action="store_true", help="also display plots interactively")
    return parser.parse_args()


def unpack(u8):
    real = (u8 >> 4).astype(np.int8)
    imag = (u8 & 0x0F).astype(np.int8)
    real[real >= 8] -= 16
    imag[imag >= 8] -= 16
    return real.astype(np.int16) + 1j * imag.astype(np.int16)


def baseline_index(recv_y, recv_x):
    return recv_y * (recv_y + 1) // 2 + recv_x


def elem_to_recv_pol(element):
    return element // 2, element % 2


def build_cpu_corr_in_gpu_layout(voltage_u8):
    """Return correlations in GPU layout [frequency, baseline, polY, polX]."""
    x = unpack(voltage_u8)  # [time, frequency, element]
    x = np.ascontiguousarray(np.transpose(x, (1, 2, 0)))  # [frequency, element, time]
    x = x.reshape(NUM_LOCAL_FREQ, NR_RECEIVERS, NR_POLARIZATIONS, SAMPLES_PER_DATA_SET)

    cpu_corr = np.zeros(
        (NUM_LOCAL_FREQ, NUM_BASELINES, NR_POLARIZATIONS, NR_POLARIZATIONS),
        dtype=np.complex128,
    )
    for recv_y in range(NR_RECEIVERS):
        y = x[:, recv_y, :, :]
        for recv_x in range(recv_y + 1):
            x_receiver = x[:, recv_x, :, :]
            visibility = np.einsum("fpt,fqt->fpq", y, np.conj(x_receiver), optimize=True)
            cpu_corr[:, baseline_index(recv_y, recv_x), :, :] = visibility
    return cpu_corr


def extract_pair_from_gpu_layout(corr_gpu_layout, element_a, element_b):
    recv_a, pol_a = elem_to_recv_pol(element_a)
    recv_b, pol_b = elem_to_recv_pol(element_b)
    if recv_b <= recv_a:
        bidx = baseline_index(recv_a, recv_b)
        return corr_gpu_layout[:, bidx, pol_a, pol_b]
    bidx = baseline_index(recv_b, recv_a)
    return np.conj(corr_gpu_layout[:, bidx, pol_b, pol_a])


def load_rawfile_payload_bytes(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"{path} is too small to contain rawFileWrite metadata")
    metadata_size = int(np.frombuffer(raw[:4].tobytes(), dtype=np.uint32)[0])
    offset = 4 + metadata_size
    if offset > raw.size:
        raise ValueError(f"Invalid metadata size in {path}: {metadata_size}")
    return raw[offset:]


def load_network_payload_from_raw(path):
    payload = load_rawfile_payload_bytes(path)
    expected = SAMPLES_PER_DATA_SET * NUM_LOCAL_FREQ * NUM_ELEMENTS
    if payload.size != expected:
        raise ValueError(
            f"{path}: expected {expected} network bytes, got {payload.size}. "
            "Check SAMPLES_PER_DATA_SET and the YAML used to capture the frame."
        )
    return payload.reshape(SAMPLES_PER_DATA_SET, NUM_LOCAL_FREQ, NUM_ELEMENTS)


def load_corr_payload_from_raw(path):
    payload = load_rawfile_payload_bytes(path)
    expected_bytes = (
        NUM_LOCAL_FREQ * NUM_BASELINES * NR_POLARIZATIONS * NR_POLARIZATIONS * 2 * 4
    )
    if payload.size != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} correlation bytes, got {payload.size}")
    raw_i32 = np.frombuffer(payload.tobytes(), dtype=np.int32)
    corr_i32 = raw_i32.reshape(
        NUM_LOCAL_FREQ, NUM_BASELINES, NR_POLARIZATIONS, NR_POLARIZATIONS, 2
    )
    return corr_i32[..., 0].astype(np.int64) + 1j * corr_i32[..., 1].astype(np.int64)


def frame_index(path):
    match = re.search(r"_(\d+)\.bin$", path.name)
    if match is None:
        raise ValueError(f"Cannot determine frame index from filename: {path}")
    return int(match.group(1))


def matching_frames(network_dir, corr_dir, start_index):
    network = {frame_index(path): path for path in network_dir.glob("*.bin")}
    corr = {frame_index(path): path for path in corr_dir.glob("*.bin")}
    common = sorted(network.keys() & corr.keys())
    if start_index is not None:
        common = [index for index in common if index >= start_index]
    if not common:
        raise FileNotFoundError(f"No matching frame indices in {network_dir} and {corr_dir}")
    return [(index, network[index], corr[index]) for index in common]


def plot_integrated_phase(cpu_corr, gpu_corr, frame_ids, output_dir, show=False, dpi=100):
    """Plot integrated visibility phase in 31 separate baseline subplots."""
    frequencies = np.linspace(300, 501.6, NUM_LOCAL_FREQ, endpoint=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_range = f"{frame_ids[0]:07d}-{frame_ids[-1]:07d}"
    num_frames = len(frame_ids)

    for antenna in range(NUM_ELEMENTS):
        pairs = [(antenna, other) for other in range(NUM_ELEMENTS) if other != antenna]
        # One image per antenna, with one independent axis for every baseline.
        # Keeping baselines in separate axes makes phase structure visible.
        n_cols = 4
        n_rows = int(np.ceil(len(pairs) / n_cols))
        fig_phase, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(18, 3.1 * n_rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()

        for axis, (a, b) in zip(axes, pairs):
            cpu_pair = extract_pair_from_gpu_layout(cpu_corr, a, b)
            gpu_pair = extract_pair_from_gpu_layout(gpu_corr, a, b)
            axis.scatter(
                frequencies, np.angle(cpu_pair), s=2, alpha=0.75,
                label="CPU", rasterized=True,
            )
            axis.scatter(
                frequencies, np.angle(gpu_pair), s=2, alpha=0.75,
                label="GPU", marker="x", rasterized=True,
            )
            axis.set_title(f"Baseline {a}–{b}", fontsize=9)
            axis.set_ylim(-np.pi, np.pi)
            axis.grid(True, alpha=0.35)
            axis.tick_params(labelsize=7)

        for axis in axes[: len(pairs)]:
            axis.set_xlabel("Frequency [MHz]", fontsize=8)
            axis.set_ylabel("Phase [rad]", fontsize=8)
        for axis in axes[len(pairs) :]:
            axis.axis("off")

        fig_phase.suptitle(
            f"Integrated visibility phase: antenna {antenna} "
            f"({num_frames} frame(s), {frame_range})"
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig_phase.legend(handles, labels, loc="upper right")
        fig_phase.tight_layout(rect=(0, 0, 1, 0.97))
        fig_phase.savefig(output_dir / f"antenna_{antenna:02d}_phase.png", dpi=dpi)

        if show:
            plt.show()
        plt.close(fig_phase)


def main():
    args = parse_args()
    if args.num_files < 0:
        raise ValueError("--num-files must be non-negative (0 means all frames)")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    frames = matching_frames(args.network_dir, args.corr_dir, args.start_index)
    if args.num_files:
        frames = frames[: args.num_files]
    print(f"Integrating {len(frames)} frame(s) into {args.output_dir}")

    cpu_integrated = None
    gpu_integrated = None
    frame_ids = []
    for index, network_path, corr_path in frames:
        print(f"Frame {index}: {network_path.name} / {corr_path.name}")
        voltage_frame = load_network_payload_from_raw(network_path)
        cpu_corr = build_cpu_corr_in_gpu_layout(voltage_frame)
        gpu_corr = load_corr_payload_from_raw(corr_path)
        print(f"  CPU/GPU correlation shape: {cpu_corr.shape} / {gpu_corr.shape}")
        cpu_integrated = cpu_corr if cpu_integrated is None else cpu_integrated + cpu_corr
        gpu_integrated = gpu_corr if gpu_integrated is None else gpu_integrated + gpu_corr
        frame_ids.append(index)

    plot_integrated_phase(
        cpu_integrated, gpu_integrated, frame_ids, args.output_dir, args.show, args.dpi
    )
    print(f"Wrote {NUM_ELEMENTS} phase plots to {args.output_dir}")


if __name__ == "__main__":
    main()
