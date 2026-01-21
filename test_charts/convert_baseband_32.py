import datetime
import numpy as np
import h5py
import os
import argparse

# ==============================
# RFSoC / Instrument Parameters
# ==============================
DELTA_TIME = 10/3 # microseconds
FREQ_0 = 300.0        # MHz
DELTA_FREQ = 0.3      # MHz
NUM_FREQ = 672
NUM_ELEMENTS = 32

# ==============================
# Kotekan Raw Parameters
# ==============================
METADATA_SIZE = 96 # Baseband frame metadata size in bytes
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

            if frame == 0:
                print("Reading frame 0 metadata...")    
                meta = f.read(METADATA_SIZE)
                meta = np.frombuffer(meta, dtype=np.uint8)
                time0_fpga = int(meta[32:40].view(np.uint64)[0])
                frame_fpga_seq = int(meta[64:72].view(np.int64)[0])  
                start_time_us = (time0_fpga*3 + frame_fpga_seq * 10) //3 # to avoid float precision issues
                start_time_dt = datetime.datetime.fromtimestamp(start_time_us / 1_000_000, tz=datetime.timezone.utc)
                print(f"Time0 FPGA: {time0_fpga} ({start_time_dt.isoformat()})")


            else:  
                meta = f.read(METADATA_SIZE)

            for _ in range(SPECTRA_PER_FRAME):
                raw = f.read(SPECTRUM_SIZE)
                spec = np.frombuffer(raw, dtype=np.uint8)
                spec = spec.reshape(NUM_FREQ, NUM_ELEMENTS)

                #Orden correcto de antenas
                spec = spec[:, ::-1]

                # (freq, ant) → (ant, freq)
                spec = spec.T
                spectra.append(spec)

    # (time, ant, freq)
    spectra = np.stack(spectra, axis=0)
    spectra = np.transpose(spectra, (1,2,0))  # (ant, freq, time)
    print(f"Final shape: {spectra.shape}")

    return spectra, time0_fpga, frame_fpga_seq, start_time_us


def convert_to_hdf5(raw_file, output_path):

    data, time0_fpga, frame_fpga_seq, start_time_us = read_raw_file(raw_file)
    if data is None:
        print("Conversion aborted.")
        return
    
    print(f"Data shape before saving: {data.shape}")

    with h5py.File(output_path, "w") as f:
        print(f"Writing {output_path}")

        # Global attributes
        f.attrs["time0_fpga"] = int(time0_fpga)
        f.attrs["start_time_utc_us"] = int(start_time_us)
        f.attrs["delta_time_s"] = DELTA_TIME / 1e6
        f.attrs["freq_0_MHz"] = FREQ_0
        f.attrs["delta_freq_MHz"] = DELTA_FREQ
        f.attrs["num_antennas"] = NUM_ELEMENTS
        f.attrs["num_freq"] = NUM_FREQ
        f.attrs["instrument"] = "32-antenna CHARTS"
        f.attrs["data_format"] = "complex_4bit_packed"
        f.attrs["real_nibble"] = "high"
        f.attrs["imag_nibble"] = "low"

        # Baseband dataset
        dset = f.create_dataset(
            "baseband",
            data=data,
            dtype=np.uint8,
            compression=None,
        )
        dset.attrs["axes"] = ["antenna", "frequency", "time"]
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
