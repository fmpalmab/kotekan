import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


DELTA_FREQ = 0.3 # MHz
FREQ_0 = 300.0  # MHz
bandwidth = 201.3  # MHz



def unpack_4bit_complex(packed_u8: np.ndarray) -> np.ndarray:
    if packed_u8.dtype != np.uint8:
        packed_u8 = packed_u8.astype(np.uint8, copy=False)
    real = (packed_u8 >> 4).astype(np.int8)
    imag = (packed_u8 & 0x0F).astype(np.int8)
    real[real >= 8] -= 16
    imag[imag >= 8] -= 16

    return real.astype(np.float32) + 1j * imag.astype(np.float32)


def flip_endianness_groups_of_8(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    n = (flat.size // 8) * 8
    if n != flat.size:
        flat = flat[:n]
    flat = flat.reshape(-1, 8)[:, ::-1].reshape(-1)

    return flat.reshape((-1,) + x.shape[1:]) if x.ndim >= 2 else flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", help="Input HDF5 file created by test_convert_baseband.py")
    ap.add_argument("--n-time", type=int, default=0, help="Use only first N time samples (0 = all)")
    ap.add_argument("--out", default="mean_spectrum_2.png", help="Output PNG filename")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        if "baseband" not in f:
            raise RuntimeError("Dataset 'baseband' not found in HDF5.")

        dset = f["baseband"]
        print("baseband shape:", dset.shape, "dtype:", dset.dtype)
        print("baseband axis:", dset.attrs.get("axis", None))
        print("file attrs:", {k: f.attrs[k] for k in f.attrs.keys()})

        # Read (optionally a subset in time)
        if args.n_time and args.n_time > 0:
            packed = dset[: args.n_time, :, :]
        else:
            packed = dset[:, :, :]

    # Unpack to complex
    data = unpack_4bit_complex(packed)  # shape: (time, pol, freq)

    # Flip endianness in groups of 8 if requested
    data = flip_endianness_groups_of_8(data)

    sample_rate = 3.3 #us
    time = data.shape[0] * sample_rate * 1e-6  # seconds
    print(f"Total time duration: {time:.3f} seconds")
    # Visibility Cross Correlation:
    Cxy = np.mean(data[:,0,:] * np.conj(data[:,1,:]), axis=0) # (freq,) {Vx*conj Vy}
    phase = np.angle(Cxy)
    freqs = FREQ_0 + np.arange(0, Cxy.size) * DELTA_FREQ  # in MHz

    # Visibility delay
    lags = (np.linspace(-3.3/2, 3.3/2, Cxy.size))  # in seconds
    delay_spec = np.fft.ifft(Cxy) 
    delay_amp = np.abs(np.fft.fftshift(delay_spec))


    max_amp = np.max(delay_amp)
    max_tau = lags[np.argmax(delay_amp)]
    print(f"Max delay amplitude: {max_amp:.3f} at lag {max_tau:.6f} s")


    # Auto-correlation for pol 0
    Cxx = np.mean(data[:,0,:] * np.conj(data[:,0,:]), axis=0)
    delay_spec_xx = np.fft.ifft(Cxx)
    delay_amp_xx = np.abs(np.fft.fftshift(delay_spec_xx))

    plt.figure(figsize=(10, 4))
    plt.scatter(freqs, phase)
    plt.title(f"Visibility Phase - pol 0 & 1")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Phase (radians)")   
    plt.ylim(-np.pi, np.pi)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 4))
    plt.plot(lags, delay_amp)
    plt.scatter(max_tau, max_amp, color='red', s=100, label=f'Max at {max_tau:.3f} µs', zorder=5)
    plt.legend()
    plt.title(f"Cross Correlation - pol 0 & 1")
    plt.xlabel("Delay (µs)")
    plt.ylabel("|FFT(Vx*conj Vy)|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(lags * 1e6, delay_amp_xx)
    plt.title(f"Auto-correlation Delay Spectrum Magnitude - pol 0")
    plt.xlabel("Delay (microseconds)")
    plt.ylabel("|FFT(Vx*conj Vx)|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()