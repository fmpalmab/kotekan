#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


NUM_FREQ = 672
NUM_ANT = 32


def unpack_4bit_complex(u8):
    """
    u8: np.uint8 array
    returns: np.complex64 array
    """
    real = (u8 >> 4).astype(np.int8)
    imag = (u8 & 0x0F).astype(np.int8)

    # sign extension 4-bit
    real[real >= 8] -= 16
    imag[imag >= 8] -= 16

    return real.astype(np.float32) + 1j * imag.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Plot mean power spectrum from baseband npy")
    p.add_argument("input", help="Input .npy with shape (time, ant, freq)")
    p.add_argument("--ant", type=int, default=0, help="Antenna index (default 0)")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print("File not found:", args.input, file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r") as f:
        data = f["baseband"]
    assert data.ndim == 3, "Expected data with shape (time, ant, freq)"

    # -----------------------------
    # Flip endianness per 8 samples
    # -----------------------------
    flat = data.reshape(-1)
    flat = flat.reshape(-1, 8)[:, ::-1].reshape(-1)
    data = flat.reshape(data.shape)

    # -----------------------------
    # Unpack 4+4 bits
    # -----------------------------
    data_c = unpack_4bit_complex(data)

    # -----------------------------
    # Compute mean spectrum
    # -----------------------------
    ant = args.ant
    spec = np.mean(np.abs(data_c[:, ant, :])**2, axis=0)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(spec)
    plt.title(f"Mean Power Spectrum – Antenna {ant}")
    plt.xlabel("Frequency bin")
    plt.ylabel("Mean |V|²")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"mean_spectrum_ant{ant}.png", dpi=150)
    plt.show()

    print(f"✓ Saved mean_spectrum_ant{ant}.png")


if __name__ == "__main__":
    main()
