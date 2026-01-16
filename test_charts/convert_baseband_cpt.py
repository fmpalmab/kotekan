import numpy as np
import h5py
import os
import time
import argparse


# RFSoC Parameters
SAMPLE_RATE = 4915.2 # MHz
NYQUIST_ZONE = 1
FFT_LEN = 8192
DELTA_TIME = 3.33e-6 # seconds
FREQ_0 = 300.0  # MHz
DELTA_FREQ = 0.3 # MHz
NUM_FREQ = 672


# Kotekan and Raw File Parameters
METADATA_SIZE = 96
PACKETS_PER_FRAME = 20480
SPECTRUM_SIZE = 21504
NUM_POLS = 2 #CPT 2
NUM_ELEMENTS = NUM_POLS * NUM_FREQ  # NEEDS TO BE REVIWED
FRAME_PAYLOAD_SIZE = PACKETS_PER_FRAME * PACKET_PAYLOAD_SIZE
spectrums_per_frame =   
TOTAL_FRAME_SIZE = METADATA_SIZE + FRAME_PAYLOAD_SIZE

def read_raw_file(filename):
    filesize = os.path.getsize(filename)
    num_frames = filesize // TOTAL_FRAME_SIZE
    
    packets_per_frame = PACKETS_PER_FRAME 
    
    print(f"Reading {filename}...")
    print(f"Size: {filesize} bytes")
    print(f"Frames: {num_frames}")
    print(f"Packets per frame: {packets_per_frame}")
    print(f"Samples per frame: {SAMPLES_PER_FRAME}")

    all_spectrum = []
    
    try:
        with open(filename, "rb") as f:
            for _ in range(num_frames):
                metadata = f.read(METADATA_SIZE)
              
                for _ in range(spectrums_per_frame):
                    spectrum_data = f.read(SPECTRUM_SIZE)
                    byte_array = np.frombuffer(spectrum_data, dtype=np.uint8)
                    byte_array = byte_array.reshape(NUM_FREQ, NUM_ELEMENTS)
                    all_packets.append(byte_array)


        return samples_all

    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_index_map():

    # Index map para polarizaciones
    dt_pol = np.dtype([("chan_id", "<u2"), ("pol", "S8")])
    pol_map = np.zeros(NUM_POLS, dtype=dt_pol)
    pol_map[0] = (0, b"X")
    pol_map[1] = (1, b"Y")
    
    # Index map para frecuencias
    dt_freq = np.dtype([("chan_id", "<u2"), ("freq_id", "<u2")])
    freq_map = np.zeros(NUM_FREQ, dtype=dt_freq)
    for i in range(NUM_FREQ):
        freq_map[i] = (i, i)
    
    return pol_map, freq_map

def convert_to_hdf5(raw_file, output_path):
    
    data = read_raw_file(raw_file)
    if data is None:
        print("Conversion Aborted.")
        return

    with h5py.File(output_path, "w") as f:
        print(f"Writing HDF5 to {output_path}...")

        # A. Global Attributes
        f.attrs["time0_ctime"] = os.path.getctime(raw_file)
        f.attrs["delta_time [s]"] = DELTA_TIME  # seconds
        f.attrs["freq_0 [MHz]"] = FREQ_0
        f.attrs["delta_f [MHz]"] = DELTA_FREQ  # MHz
        f.attrs["num_pols"] = NUM_POLS
        f.attrs["num_freq"] = NUM_FREQ
        f.attrs["samples_per_data_set"] = SAMPLES_PER_FRAME
        f.attrs["instrument_name"] = "CPT_RFSoC"
        f.attrs["packet_format"] = "4bit_complex"
        f.attrs["real_nibble"] = "high"
        f.attrs["imag_nibble"] = "low"
        
        # B. Index Maps (separados por dimensión)
        grp_im = f.create_group("index_map")
        pol_map, freq_map = create_index_map()
        grp_im.create_dataset("pol", data=pol_map)
        grp_im.create_dataset("freq", data=freq_map)
        
        # C. Baseband Data con shape (time, pol, freq)
        dset = f.create_dataset("baseband", data=data, dtype=np.uint8)
        dset.attrs["axis"] = ["time", "pol", "freq"] 
    
    print(f"Success. Output: {output_path}")
    print(f"Shape: {data.shape}, Size: {os.path.getsize(output_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Kotekan .data raw dump to Baseband HDF5")
    parser.add_argument("raw_file", help="Input raw file path")
    parser.add_argument("output_file", help="Output HDF5 file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.raw_file):
        print(f"Input file {args.raw_file} does not exist.")
    else:
        convert_to_hdf5(args.raw_file, args.output_file)

