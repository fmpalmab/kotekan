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
    ap.add_argument("--ant-x", type=int, default=3, help="Antenna X")
    ap.add_argument("--ant-y", type=int, default=4, help="Antenna Y")
    ap.add_argument("--n-time", type=int, default=0, help="Use first N time samples (0 = all)")
    args = ap.parse_args()


    with h5py.File(args.h5, "r") as f:
        dset = f["baseband"]

        if args.n_time > 0:
            packed = dset[:, :, :args.n_time]
        else:
            packed = dset[:, :, :]

    # packed shape: (time, antenna, freq)
    print("Packed shape:", packed.shape)

    x_u8 = packed[args.ant_x,: , :]   
    y_u8 = packed[args.ant_y,: , :]   #freq,time
    valid_mask = np.any(x_u8 != 0, axis=0)

    #x_u8 = x_u8[:, valid_mask]
    #y_u8 = y_u8[:, valid_mask]

    print("Valid samples:", np.sum(valid_mask))
    print(x_u8.shape, y_u8.shape)

     # Unpack to complex

    x = unpack_4bit_complex(x_u8)
    y = unpack_4bit_complex(y_u8)

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


if __name__ == "__main__":
    main()
