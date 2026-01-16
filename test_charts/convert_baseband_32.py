import numpy as np
import h5py
import os
import argparse

# ==============================
# RFSoC / Instrument Parameters
# ==============================
DELTA_TIME = 3.33e-6  # s
FREQ_0 = 300.0        # MHz
DELTA_FREQ = 0.3      # MHz
NUM_FREQ = 672
NUM_ELEMENTS = 32

# ==============================
# Kotekan Raw Parameters
# ==============================
METADATA_SIZE = 96
SPECTRUM_SIZE = 21504
SPECTRA_PER_FRAME = 15000
FRAME_SIZE = METADATA_SIZE + SPECTRA_PER_FRAME * SPECTRUM_SIZE


def read_raw_file(filename):

    filesize = os.path.getsize(filename)
    num_frames = filesize // FRAME_SIZE

    print(f"Reading {filename}")
    print(f"Frames: {num_frames}")

    spectra = []

    with open(filename, "rb") as f:
        for frame in range(num_frames):
            f.read(METADATA_SIZE)

            for _ in range(SPECTRA_PER_FRAME):
                raw = f.read(SPECTRUM_SIZE)

                spec = np.frombuffer(raw, dtype=np.uint8)
                spec = spec.reshape(NUM_FREQ, NUM_ELEMENTS)

                # Orden correcto de antenas
                spec = spec[:, ::-1]

                # (freq, ant) → (ant, freq)
                spec = spec.T

                spectra.append(spec)

    # (time, ant, freq)
    spectra = np.stack(spectra, axis=0)
    spectra = np.transpose(spectra, (1,2,0))  # (ant, freq, time)
    print(f"Final shape: {spectra.shape}")
    return spectra


def convert_to_hdf5(raw_file, output_path):

    data = read_raw_file(raw_file)
    if data is None:
        print("Conversion aborted.")
        return
    
    print(data.dtype)
    print(f"Data shape before saving: {data.shape}")
    print(data.nbytes)

    with h5py.File(output_path, "w") as f:
        print(f"Writing {output_path}")

        # ======================
        # Global attributes
        # ======================
        f.attrs["time0_ctime"] = os.path.getctime(raw_file)
        f.attrs["delta_time_s"] = DELTA_TIME
        f.attrs["freq_0_MHz"] = FREQ_0
        f.attrs["delta_freq_MHz"] = DELTA_FREQ
        f.attrs["num_antennas"] = NUM_ELEMENTS
        f.attrs["num_freq"] = NUM_FREQ
        f.attrs["instrument"] = "CPT_RFSoC"
        f.attrs["data_format"] = "complex_4bit_packed"
        f.attrs["real_nibble"] = "high"
        f.attrs["imag_nibble"] = "low"

        # ======================
        # Baseband dataset
        # ======================
        dset = f.create_dataset(
            "baseband",
            data=data,
            dtype=np.uint8,
            compression=None,
        )

        dset.attrs["axes"] = ["time", "antenna", "frequency"]
        dset.attrs["units"] = "packed_complex_4bit"

    print("Done.")
    print(f"Final shape: {data.shape}")
    print(f"File size: {os.path.getsize(output_path)/1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Kotekan raw baseband to HDF5")
    parser.add_argument("raw_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    convert_to_hdf5(args.raw_file, args.output_file)
