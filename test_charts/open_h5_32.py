#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


# =====================
# Utils
# =====================
def unpack_4bit_complex(u8: np.ndarray) -> np.ndarray:
    real = (u8 >> 4).astype(np.int8)
    imag = (u8 & 0x0F).astype(np.int8)
    real[real >= 8] -= 16
    imag[imag >= 8] -= 16
    return real.astype(np.float32) + 1j * imag.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", help="Input HDF5 baseband file")
    ap.add_argument("--ant-x", type=int, default=0, help="Antenna X")
    ap.add_argument("--ant-y", type=int, default=8, help="Antenna Y")
    ap.add_argument("--n-time", type=int, default=0, help="Use first N time samples (0 = all)")
    args = ap.parse_args()


    with h5py.File(args.h5, "r") as f:
        dset = f["baseband"]

        if args.n_time > 0:
            packed = dset[:, :, :args.n_time]
        else:
            packed = dset[:, :, :]


        # Header info
        print("File attributes:")
        for k in f.attrs.keys():
            print(f"  {k}: {f.attrs[k]}")

            start_time_us = f.attrs.get("start_time_utc_us", 0)
            delta_time_s = f.attrs.get("delta_time_s", 0)
            freq_0_MHz = f.attrs.get("freq_0_MHz", 0)
            delta_freq_MHz = f.attrs.get("delta_freq_MHz", 0)
    

        
    # packed shape: (time, antenna, freq)
    print("Packed shape:", packed.shape)

    x_u8 = packed[args.ant_x,: , :]   
    y_u8 = packed[args.ant_y,: , :]   #freq,time
    valid_mask = np.any(x_u8 != 0, axis=0)
    print(valid_mask)
    print(np.where(valid_mask))
    #x_u8 = x_u8[:, valid_mask]
    #y_u8 = y_u8[:, valid_mask]

    print("Valid samples:", np.sum(valid_mask))
    print(x_u8.shape, y_u8.shape)

     #Unpack to complex

    x = unpack_4bit_complex(x_u8)
    y = unpack_4bit_complex(y_u8)

    #Plot spectrum ant x on time

    # time_axis = np.arange(x.shape[1]) * delta_time_s + start_time_us * 1e-6
    # freq_axis = np.arange(x.shape[0]) * delta_freq_MHz + freq_0_MHz

    # plt.figure(figsize=(10,5))
    # plt.imshow(np.abs(x), aspect="auto", cmap="viridis", extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]], origin="lower")
    # plt.show()

    mean_x = np.mean(np.abs(x)**2, axis=1)
    mean_y = np.mean(np.abs(y)**2, axis=1)
    print("Mean power Ant X:", np.mean(mean_x))
    print("Mean power Ant Y:", np.mean(mean_y))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(mean_x, label=f"Ant {args.ant_x}")
    ax.plot(mean_y, label=f"Ant {args.ant_y}")
    ax.set_xlabel("Frequency channel")
    ax.set_ylabel("Mean Power")
    ax.set_title("Mean Power Spectrum")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    Cxy = np.mean(x * np.conj(y), axis=1)   # average over time
    phase = np.angle(Cxy)
    print(phase.shape)

    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(phase.size), phase, s=5)
    plt.ylim(-2*np.pi,2*np.pi)
    plt.xlabel("Frequency channel")
    plt.ylabel("Visibility phase [rad]")
    plt.title(f"Visibility Phase – Ant {args.ant_x} × Ant {args.ant_y}")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    # valid_mask = np.any(packed != 0, axis=2)
    # print("Valid samples:", np.sum(valid_mask))


    # chains = unpack_4bit_complex(packed)
    

    # n_ant = chains.shape[0]
    # n_freq = chains.shape[1]
    
    # print(f"Chains decoded shape: {chains.shape} (Antennas, Freqs, Time)")

    # fig, axes = plt.subplots(n_ant, n_ant, figsize=(20, 20))
    # frequencies = np.linspace(300, 501.6, n_freq, endpoint=False)  # Frequency axis




    # print("Calculating correlations and plotting...")

    # for i in range(n_ant):
    #     for j in range(n_ant):
    #         # Calculate correlation: Average( Antenna_i * conj(Antenna_j) ) over time axis
    #         # chains[i] shape is (freq, time)
    #         # Result shape is (freq,)
    #         corr_spectrum = np.mean(chains[i] * np.conj(chains[j]), axis=1)
    #         phase = np.angle(corr_spectrum)


    #         ax = axes[i, j]
    #         ax.scatter(frequencies, phase, s=1)
    #         ax.set_ylim(-np.pi, np.pi)  

    #         # Formatting axes to reduce clutter
    #         if i == n_ant - 1:
    #             # Bottom row labels
    #             if j == 0: ax.set_xlabel("Freq [MHz]", fontsize=7)
    #         else:
    #             ax.set_xticks([])

    #         if j == 0:
    #             # Left column labels
    #             ax.set_ylabel(f"Ant {i}", fontsize=7)
    #         else:
    #             ax.set_yticks([])
            
    #         # Top row titles
    #         if i == 0:
    #             ax.set_title(f"Ant {j}", fontsize=7)

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()