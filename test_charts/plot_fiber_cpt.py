#!/usr/bin/env python3
# script: plot_fiber
# Uso: ./plot_fiber input.npy [--out output.npy] [--frame-size 5376] [--plot]
# Carga un .npy 1D de muestras (posible complejo), agrupa en columnas de frame_size
# Resultado tiene shape (frame_size, n_frames). Si sobra insuficiente, se descarta.

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt



freq_spec = 672
spect_per_packet = 4
pols = 2
payload_per_packet = freq_spec * spect_per_packet * pols  # 5376

def main():
    p = argparse.ArgumentParser(description="Agrupa muestras 1D en frames de tamaño fijo.")
    p.add_argument("input", help=".npy de entrada (array 1D)")
    p.add_argument("--out", "-o", help="Guardar salida .npy (matriz) (opcional)")
    p.add_argument("--frame-size", "-f", type=int, default=5376, help="Tamaño de cada frame (por defecto 5376)")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print("Archivo no encontrado:", args.input, file=sys.stderr)
        sys.exit(2)

    data = np.load(args.input, allow_pickle=False)
    # Reshape to (n_frames, pols, freq_spec), flip endianness per 8 samples, then reshape back
    n_frames = data.shape[0]
    data_flat = data.reshape(-1)  # Flatten to 1D
    data_flat = data_flat.reshape(-1, 8)[:, ::-1].reshape(-1)  # Flip endianness
    data = data_flat.reshape(n_frames, pols, freq_spec)  # Reshape back to original shape


    spec = (np.abs(data[:,1,:])**2).mean(axis=0)
    plt.figure()
    plt.plot(spec)
    plt.title("Mean Spectrum - Polarization 1")
    plt.xlabel("Frequency Bins")
    plt.ylabel("Mean Amplitude")
    plt.grid()
    plt.savefig("mean_spectrum_pol1.png", dpi=150)
    print("✓ Saved plot to mean_spectrum_pol1.png")
    plt.show()



if __name__ == "__main__":
    main()