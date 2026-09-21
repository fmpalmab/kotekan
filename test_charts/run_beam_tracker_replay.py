#!/usr/bin/env python3
"""
CHARTS Beam Tracker Replay Runner
==================================
Generates a Kotekan YAML config for replaying baseband frames through
cudaBeamTrackerCommand (8-beam formation) and executes the pipeline.

The tracker reads raw int4x2 baseband frames (same format as the correlator)
and outputs complex float32 formed-beam voltages [time][freq][beam].
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from constants import (
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_SPACING_M,
)


def create_beam_tracker_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    baseband_name: str,
    tracker_dir: Path,
    tracker_name: str,
    num_frames: int,
    samples_per_data_set: int = 1536,
    num_elements: int = 64,
    num_local_freq: int = 336,
    max_beams: int = 8,
    integration_spectra: int = 320,
    spacing_m: float = DEFAULT_SPACING_M,
    site_lat_deg: float = CHARTS_LATITUDE_DEG,
    site_lon_deg: float = CHARTS_LONGITUDE_DEG,
    source_ra_deg: float = 83.633,
    source_dec_deg: float = 22.014,
    initial_lst_hours: float = 5.575,
    buffer_depth: int = 4,
):
    """Writes Kotekan YAML for rawFileRead -> cudaBeamTrackerCommand -> rawFileWrite."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    sizeof_complex_float = 8

    content = f"""######################################################################
# CHARTS Beam Tracker Replay Pipeline
# Source: {baseband_dir / baseband_name}_%07d.bin ({num_frames} frames)
# Beams: {max_beams} | Integration: {integration_spectra} spectra
######################################################################
type: config
log_level: info

cpu_affinity: [0, 1, 2, 3]

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples_per_data_set}
num_data_sets: 1
sizeof_complex_float: {sizeof_complex_float}
buffer_depth: {buffer_depth}

main_pool:
  kotekan_metadata_pool: chordMetadata
  num_metadata_objects: 50

# Input baseband buffer [time][freq][antenna] (int4x2_t)
network_capture_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * num_elements
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

# Output formed beams buffer [time][freq][beam] of float2
host_formed_beams_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * {max_beams} * sizeof_complex_float
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

# Stage 1: Read simulated baseband frames from disk
replay_reader:
  kotekan_stage: rawFileRead
  base_dir: {baseband_dir.as_posix()}
  file_name: {baseband_name}
  file_ext: bin
  buf: network_capture_buf
  prefix_hostname: false
  end_interrupt: true

# Stage 2: GPU Beam Tracker
gpu:
  profiling: false
  kernel_path: "lib/cuda/kernels"
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaBeamTrackerCommand
      gpu_mem_voltage: voltage
      gpu_mem_formed_beams: formed_beams
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      integration_spectra: {integration_spectra}
      spacing_m: {spacing_m}
      max_beams: {max_beams}
      initial_active_beams: {max_beams}
      site_lat_deg: {site_lat_deg}
      site_lon_deg: {site_lon_deg}
      source_ra_deg: {source_ra_deg}
      source_dec_deg: {source_dec_deg}
      initial_lst_hours: {initial_lst_hours}
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: formed_beams
      out_buf: host_formed_beams
  gpu_0:
    kotekan_stage: cudaProcess
    gpu_id: 0
    buffer_depth: 3
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_formed_beams: host_formed_beams_buffer

# Stage 3: Write formed beam voltages to disk
write_tracker_output:
  kotekan_stage: rawFileWrite
  in_buf: host_formed_beams_buffer
  base_dir: {tracker_dir.as_posix()}
  file_name: {tracker_name}
  file_ext: bin
  num_frames_per_file: 1
  exit_after_n_files: {num_frames}
  prefix_hostname: false
"""
    with open(yaml_path, "w") as f:
        f.write(content)


def run_beam_tracker(
    window_dir: Path,
    file_name: Optional[str] = None,
    num_frames: Optional[int] = None,
    kotekan_bin: Optional[Path] = None,
    tracker_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
    num_antennas: int = 64,
    num_freq: int = 336,
    samples_per_frame: int = 1536,
    max_beams: int = 8,
    integration_spectra: int = 320,
    source_ra_deg: float = 83.633,
    source_dec_deg: float = 22.014,
    initial_lst_hours: float = 5.575,
):
    """Executes Kotekan beam tracker replay on baseband window frames."""
    window_dir = window_dir.resolve()

    # Auto-detect base name and frame count
    if file_name is None:
        meta_files = list(window_dir.glob("*_meta.h5"))
        if meta_files:
            base_name = meta_files[0].stem.replace("_meta", "")
        else:
            bin_files = sorted(window_dir.glob("*.bin"))
            if not bin_files:
                raise FileNotFoundError(f"No .bin or _meta.h5 files in {window_dir}")
            base_name = "_".join(bin_files[0].stem.split("_")[:-1])
    else:
        base_name = file_name

    if num_frames is None:
        meta_file = window_dir / f"{base_name}_meta.h5"
        if meta_file.exists():
            import h5py
            with h5py.File(meta_file, "r") as hf:
                num_frames = int(hf.attrs.get("written_frames_count", 0))
                if num_frames == 0 and "written_out_index" in hf:
                    num_frames = len(hf["written_out_index"])
        else:
            num_frames = len(list(window_dir.glob(f"{base_name}_*.bin")))

    if num_frames <= 0:
        raise ValueError(f"No frames found in {window_dir}")

    kotekan_executable = (kotekan_bin or (_kotekan_root / "build" / "kotekan" / "kotekan")).resolve()
    if not kotekan_executable.exists():
        raise FileNotFoundError(f"Kotekan binary not found: {kotekan_executable}")

    t_dir = (tracker_dir or (window_dir / "tracker")).resolve()
    cfg_dir = (configs_dir or (window_dir / "configs")).resolve()
    t_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    tracker_name = f"beams_{base_name}"
    yaml_path = cfg_dir / f"config_tracker_{base_name}.yaml"

    print("=" * 78)
    print(" CHARTS BEAM TRACKER REPLAY RUNNER")
    print("=" * 78)
    print(f" Window Directory     : {window_dir}")
    print(f" Base Stream Name     : {base_name}_%07d.bin")
    print(f" Total Frames         : {num_frames}")
    print(f" Max Beams            : {max_beams}")
    print(f" Kotekan Binary       : {kotekan_executable}")
    print(f" Tracker Output Dir   : {t_dir}")
    print(f" YAML Configuration   : {yaml_path}")
    print("=" * 78)

    # 1. Generate YAML
    print("\n[1/2] Generating Beam Tracker YAML Config...")
    create_beam_tracker_yaml(
        yaml_path=yaml_path,
        baseband_dir=window_dir,
        baseband_name=base_name,
        tracker_dir=t_dir,
        tracker_name=tracker_name,
        num_frames=num_frames,
        samples_per_data_set=samples_per_frame,
        num_elements=num_antennas,
        num_local_freq=num_freq,
        max_beams=max_beams,
        integration_spectra=integration_spectra,
        source_ra_deg=source_ra_deg,
        source_dec_deg=source_dec_deg,
        initial_lst_hours=initial_lst_hours,
    )
    print(f"    Saved: {yaml_path.name}")

    # 2. Execute Kotekan
    print(f"\n[2/2] Executing Kotekan Beam Tracker over {num_frames} frames...")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{_kotekan_root / 'build' / 'external' / 'n2k'}:{env.get('LD_LIBRARY_PATH', '')}"

    if "CUDA_HOME" not in env:
        for var in ["CUDA_ROOT", "EBROOTCUDA", "CUDA_PATH"]:
            if var in env:
                env["CUDA_HOME"] = env[var]
                break
        if "CUDA_HOME" not in env:
            import shutil
            nvcc_bin = shutil.which("nvcc")
            if nvcc_bin:
                env["CUDA_HOME"] = str(Path(nvcc_bin).resolve().parent.parent)

    if "CUDA_HOME" in env:
        env["CUDA_PATH"] = env["CUDA_HOME"]
        env["CPATH"] = f"{env['CUDA_HOME']}/include:{env.get('CPATH', '')}"

    cmd = [str(kotekan_executable), "--config", str(yaml_path)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(_kotekan_root), env=env, capture_output=True, text=True)
    t1 = time.perf_counter()

    if res.returncode != 0:
        print(f"[ERROR] Kotekan beam tracker failed with exit code {res.returncode}!")
        print("STDERR:")
        print(res.stderr[-2000:] if len(res.stderr) > 2000 else res.stderr)
        print("STDOUT:")
        print(res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout)
        sys.exit(1)

    print(f"    Completed in {(t1 - t0):.2f} s ({(t1 - t0) / max(1, num_frames):.3f} s/frame)")

    # Verify output
    output_files = sorted(t_dir.glob(f"{tracker_name}_*.bin"))
    print(f"\n    Output files: {len(output_files)} in {t_dir}")

    print("\n" + "=" * 78)
    print(f" BEAM TRACKER PIPELINE FOR [{base_name}] COMPLETED SUCCESSFULLY!")
    print(f" Output Files: {t_dir}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="CHARTS Beam Tracker Replay Runner")
    parser.add_argument("--window-dir", type=str, required=True)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--kotekan-bin", type=str, default=None)
    parser.add_argument("--tracker-dir", type=str, default=None)
    parser.add_argument("--configs-dir", type=str, default=None)
    parser.add_argument("--antennas", type=int, default=64)
    parser.add_argument("--num-freq", type=int, default=336)
    parser.add_argument("--samples-per-frame", type=int, default=1536)
    parser.add_argument("--max-beams", type=int, default=8)
    parser.add_argument("--integration-spectra", type=int, default=320)
    parser.add_argument("--source-ra-deg", type=float, default=83.633)
    parser.add_argument("--source-dec-deg", type=float, default=22.014)
    parser.add_argument("--initial-lst-hours", type=float, default=5.575)
    args = parser.parse_args()

    run_beam_tracker(
        window_dir=Path(args.window_dir),
        file_name=args.file_name,
        num_frames=args.num_frames,
        kotekan_bin=Path(args.kotekan_bin) if args.kotekan_bin else None,
        tracker_dir=Path(args.tracker_dir) if args.tracker_dir else None,
        configs_dir=Path(args.configs_dir) if args.configs_dir else None,
        num_antennas=args.antennas,
        num_freq=args.num_freq,
        samples_per_frame=args.samples_per_frame,
        max_beams=args.max_beams,
        integration_spectra=args.integration_spectra,
        source_ra_deg=args.source_ra_deg,
        source_dec_deg=args.source_dec_deg,
        initial_lst_hours=args.initial_lst_hours,
    )


if __name__ == "__main__":
    main()
