import datetime
import numpy as np
import h5py
import os
import argparse

# RFSoC / Instrument Parameters
DELTA_TIME = 10/3 # microseconds
FREQ_0 = 300.0        # MHz
DELTA_FREQ = 0.3      # MHz
NUM_FREQ = 672
NUM_ELEMENTS = 32

# Kotekan Raw Parameters
METADATA_SIZE = 96 # Baseband frame metadata size in bytes
SPECTRUM_SIZE = 21504
SPECTRA_PER_FRAME = 15000
FRAME_SIZE = METADATA_SIZE + SPECTRA_PER_FRAME * SPECTRUM_SIZE


# File output parameters: objective is 500 MB files
FREQ_CHUNK = 672//8 
MAX_SAMPLES_CHUNK = 186000 # aprox 0.56 sec 
OUTDIR = "/hdd/32_test/

buffers = {}
time_counters = {}
file_indices = {}
first_sample_index = {}

for f0 in range(0, NUM_FREQ, FREQ_CHUNK):
    buffers[f0] = []
    time_counters[f0] = 0
    file_indices[f0] = 0
    first_sample_index[f0] = None

def stream_raw_file(filename, process_spectrum):

    filesize = os.path.getsize(filename)
    num_frames = filesize // FRAME_SIZE
    print(f"Reading {filename}")
    print(f"Frames: {num_frames}")      
    global_sample_index = 0
    start_time_us_global = None

    with open(filename, "rb") as f:
        for frame in range(num_frames):

            meta = f.read(METADATA_SIZE)
            meta = np.frombuffer(meta, dtype=np.uint8)

            if frame == 0:
                time0_fpga = int(meta[32:40].view(np.uint64)[0])
                frame_fpga_seq = int(meta[64:72].view(np.int64)[0])
        
                start_time_us_global = (time0_fpga * 3 + frame_fpga_seq * 10) // 3

                print(f"Start time global (us): {start_time_us_global}")

            for _ in range(SPECTRA_PER_FRAME):

                raw = f.read(SPECTRUM_SIZE)
                if not raw:
                    print("End of file reached unexpectedly")
                    return

                spec = np.frombuffer(raw, dtype=np.uint8)
                spec = spec.reshape(NUM_FREQ, NUM_ELEMENTS)
                spec = spec[:, ::-1].T  # (ant, freq)

                process_spectrum(
                    spec,
                    global_sample_index,
                    start_time_us_global)

                global_sample_index += 1

def process_spectrum(spec, global_sample_index, start_time_us_global):

    for f0 in range(0, NUM_FREQ, FREQ_CHUNK):
        f1 = min(f0 + FREQ_CHUNK, NUM_FREQ)
        freq_slice = spec[:, f0:f1]

        if time_counters[f0] == 0:
            first_sample_index[f0] = global_sample_index

        buffers[f0].append(freq_slice)
        time_counters[f0] += 1
    

        if time_counters[f0] >= MAX_SAMPLES_CHUNK:

            first_sample = first_sample_index[f0]
            start_time_file_us = (3*start_time_us_global +first_sample *10) //3  # avoid float precision issues
            

            write_hdf5(
                buffers[f0],
                f0,
                f1,
                start_time_file_us,
                file_indices[f0]
            )

            buffers[f0] = []
            time_counters[f0] = 0
            file_indices[f0] += 1
            first_sample_index[f0] = None


def write_hdf5(buffer, f0, f1, start_time_us, idx):

    data = np.stack(buffer, axis=-1)  # (ant, freq, time)

    fname = f"bb_f{f0:04d}_{f1:04d}_chunk{idx:05d}.h5"


    if os.path.exists(OUTDIR) == False:
        os.makedirs(OUTDIR)

    with h5py.File(os.path.join(OUTDIR, fname), "w") as f:

        f.attrs["start_time_utc_us"] = start_time_us
        f.attrs["delta_time_us"] = DELTA_TIME
        f.attrs["freq_start_MHz"] = FREQ_0 + f0 * DELTA_FREQ
        f.attrs["delta_freq_MHz"] = DELTA_FREQ

        dset = f.create_dataset("baseband", data=data, dtype=np.uint8)
        dset.attrs["axes"] = ["antenna", "frequency", "time"]


def main():

    global buffers, time_counters, file_indices
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_file", help="Input Kotekan raw baseband file")
    args = ap.parse_args()

    stream_raw_file(args.raw_file, process_spectrum)    

    for f0 in buffers:
        if time_counters[f0] > 0:
            write_hdf5(
                buffers[f0],
                f0,
                min(f0 + FREQ_CHUNK, NUM_FREQ),
                None,
                file_indices[f0]
            )

if __name__ == "__main__":
    main()