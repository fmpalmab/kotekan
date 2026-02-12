import datetime
import numpy as np
import h5py
import os
import argparse
import glob

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

buffers = {}
time_counters = {}
file_indices = {}
first_sample_index = {}

for f0 in range(0, NUM_FREQ, FREQ_CHUNK):
    buffers[f0] = []
    time_counters[f0] = 0
    file_indices[f0] = 0
    first_sample_index[f0] = None
    
    

def stream_raw_file(filename, process_spectrum, outdir_base):
    
    global OUTDIR

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

                folder_time = datetime.datetime.fromtimestamp(
                    start_time_us_global / 1_000_000, tz=datetime.timezone.utc
                ).strftime("%y%m%dT%H%M%SZ")
                OUTDIR = os.path.join(outdir_base, f"{folder_time}_CHARTS_hdf5")
                os.makedirs(OUTDIR, exist_ok=True)

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
                first_sample,
                start_time_file_us,
                file_indices[f0]
                
            )

            buffers[f0] = []
            time_counters[f0] = 0
            file_indices[f0] += 1
            first_sample_index[f0] = None


def write_hdf5(buffer, f0, f1, first_sample_idx, start_time_us, idx):

    data = np.stack(buffer, axis=-1)  # (ant, freq, time)

    fname = f"bb_f{f0:04d}_{f1:04d}_chunk{idx:05d}.h5"


    if os.path.exists(OUTDIR) == False:
        os.makedirs(OUTDIR)

    with h5py.File(os.path.join(OUTDIR, fname), "w") as f:

        
        f.attrs["freq_start_idx"] = f0
        f.attrs["freq_end_idx"] = f1
        f.attrs["first_sample_idx"] = first_sample_idx
        f.attrs["time_len"] = data.shape[2]
        
        if start_time_us is not None:
            f.attrs["start_time_utc_us"] = start_time_us
        f.attrs["delta_time_us"] = DELTA_TIME
        f.attrs["freq_start_MHz"] = FREQ_0 + f0 * DELTA_FREQ
        f.attrs["delta_freq_MHz"] = DELTA_FREQ

        dset = f.create_dataset("baseband", data=data, dtype=np.uint8)
        dset.attrs["axes"] = ["antenna", "frequency", "time"]
        
        
def create_vds(outdir, vds_name="baseband_virtual.h5"):


    files = sorted(glob.glob(os.path.join(outdir, "bb_f*_chunk*.h5")))
    if len(files) == 0:
        raise RuntimeError("No chunked HDF5 files found to build VDS")


    time_max = 0
    for fname in files:
        with h5py.File(fname, "r") as f:
            t0 = int(f.attrs["first_sample_idx"])
            tlen = int(f.attrs["time_len"])
            time_max = max(time_max, t0 + tlen)

    print(f"[VDS] Total time samples: {time_max}")

    layout = h5py.VirtualLayout(
        shape=(NUM_ELEMENTS, NUM_FREQ, time_max),
        dtype=np.uint8
    )

    for fname in files:
        with h5py.File(fname, "r") as f:
            dset = f["baseband"]

            f0 = int(f.attrs["freq_start_idx"])
            f1 = int(f.attrs["freq_end_idx"])
            t0 = int(f.attrs["first_sample_idx"])
            t1 = t0 + int(f.attrs["time_len"])

            vsource = h5py.VirtualSource(
                fname,
                "baseband",
                shape=dset.shape
            )

            layout[:, f0:f1, t0:t1] = vsource

    vds_path = os.path.join(outdir, vds_name)
    with h5py.File(vds_path, "w") as f:
        dset = f.create_virtual_dataset("baseband", layout)
        dset.attrs["axes"] = ["antenna", "frequency", "time"]
        dset.attrs["description"] = "Virtual baseband dataset (CHARTS)"

    print(f"[VDS] Created: {vds_path}")


def main():

    global buffers, time_counters, file_indices, first_sample_index
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_file", help="Input Kotekan raw baseband file")
    ap.add_argument("--outdir-base", help="Output base directory", default="/hdd")
    args = ap.parse_args()

    stream_raw_file(args.raw_file, process_spectrum, args.outdir_base)

    for f0 in buffers:
        if time_counters[f0] > 0:
            
            first_sample = first_sample_index[f0]
            
            start_time_file_us = None
            print("Writing final chunk for freq range", f0, "to", min(f0 + FREQ_CHUNK, NUM_FREQ))
            write_hdf5(
                buffers[f0],
                f0,
                min(f0 + FREQ_CHUNK, NUM_FREQ),
                first_sample,
                start_time_file_us,
                file_indices[f0]
            )
    create_vds(OUTDIR)

if __name__ == "__main__":
    main()