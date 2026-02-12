import datetime
import numpy as np
import h5py
import os
import argparse


# RFSoC / Instrument Parameters
DELTA_TIME = 10 / 3  # microseconds
FREQ_0 = 300.0       # MHz
DELTA_FREQ = 0.3     # MHz
NUM_FREQ = 672
NUM_ELEMENTS = 32

# Kotekan Raw Parameters
METADATA_SIZE = 96
SPECTRUM_SIZE = 21504
SPECTRA_PER_FRAME = 15000
FRAME_SIZE = METADATA_SIZE + SPECTRA_PER_FRAME * SPECTRUM_SIZE


def convert_to_hdf5(raw_file, output_path):

    filesize = os.path.getsize(raw_file)
    num_frames = filesize // FRAME_SIZE
    num_spectra_total = num_frames * SPECTRA_PER_FRAME

    print(f"Reading {raw_file}")
    print(f"Frames: {num_frames}")
    print(f"Total spectra: {num_spectra_total}")

    with open(raw_file, "rb") as fraw, h5py.File(output_path, "w") as fh5:


        meta = fraw.read(METADATA_SIZE)
        meta = np.frombuffer(meta, dtype=np.uint8)

        time0_fpga = int(meta[32:40].view(np.uint64)[0])
        frame_fpga_seq = int(meta[64:72].view(np.int64)[0])

        start_time_us = (time0_fpga * 3 + frame_fpga_seq * 10) // 3
        start_time_dt = datetime.datetime.fromtimestamp(
            start_time_us / 1_000_000, tz=datetime.timezone.utc
        )

        print(f"Time0 FPGA: {time0_fpga} ({start_time_dt.isoformat()})")

        fh5.attrs["time0_fpga"] = int(time0_fpga)
        fh5.attrs["start_time_utc_us"] = int(start_time_us)
        fh5.attrs["delta_time_s"] = DELTA_TIME / 1e6
        fh5.attrs["freq_0_MHz"] = FREQ_0
        fh5.attrs["delta_freq_MHz"] = DELTA_FREQ
        fh5.attrs["num_antennas"] = NUM_ELEMENTS
        fh5.attrs["num_freq"] = NUM_FREQ
        fh5.attrs["instrument"] = "32-antenna CHARTS"
        fh5.attrs["data_format"] = "complex_4bit_packed"
        fh5.attrs["real_nibble"] = "high"
        fh5.attrs["imag_nibble"] = "low"

        dset = fh5.create_dataset(
            "baseband",
            shape=(NUM_ELEMENTS, NUM_FREQ, num_spectra_total),
            dtype=np.uint8,
            compression=None,
        )

        dset.attrs["axes"] = ["antenna", "frequency", "time"]
        dset.attrs["units"] = "packed_complex_4bit"


        t = 0  # global time index

        for _ in range(SPECTRA_PER_FRAME):
            raw = fraw.read(SPECTRUM_SIZE)

            spec = np.frombuffer(raw, dtype=np.uint8)
            spec = spec.reshape(NUM_FREQ, NUM_ELEMENTS)
            spec = spec[:, ::-1]
            spec = spec.T  # (ant, freq)

            dset[:, :, t] = spec
            t += 1

        for frame in range(1, num_frames):
            print(f"Frame {frame}/{num_frames}")

            _ = fraw.read(METADATA_SIZE)

            for _ in range(SPECTRA_PER_FRAME):
                raw = fraw.read(SPECTRUM_SIZE)

                spec = np.frombuffer(raw, dtype=np.uint8)
                spec = spec.reshape(NUM_FREQ, NUM_ELEMENTS)
                spec = spec[:, ::-1]
                spec = spec.T

                dset[:, :, t] = spec
                t += 1

    print("Done.")
    print(f"Final shape: {dset.shape}")
    print(f"File size: {os.path.getsize(output_path)/1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Kotekan raw baseband to HDF5 (streaming, no RAM duplication)"
    )
    parser.add_argument("raw_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    convert_to_hdf5(args.raw_file, args.output_file)
