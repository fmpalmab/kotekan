import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Frame
RAW_NET = Path('/data/corr/network/user-System-Product-Name_netwok_capture_0000001.bin')
RAW_CORR = Path('/data/corr/host_corr/user-System-Product-Name_host_correlation_0000001.bin')


# Kotekan CHARTS parameters
NUM_ELEMENTS = 32
NUM_LOCAL_FREQ = 672
SAMPLES_PER_DATA_SET = 14976

# Kernel parameters
NR_POLARIZATIONS = 2
NR_RECEIVERS = NUM_ELEMENTS // NR_POLARIZATIONS   # 16
NUM_BASELINES = NR_RECEIVERS * (NR_RECEIVERS + 1) // 2   # 136


# Parameters to plot
ANT_REF = 16
PAIRS_TO_PLOT = [(ANT_REF, b) for b in range(8, 32)]   # 8..31 inclusive


# Unpack 4-bit signed 
def unpack(u8):
    real = (u8 >> 4).astype(np.int8)
    imag = (u8 & 0x0F).astype(np.int8)
    real[real >= 8] -= 16
    imag[imag >= 8] -= 16
    return real.astype(np.int16) + 1j * imag.astype(np.int16)


def baseline_index(recv_y, recv_x):
    return recv_y * (recv_y + 1) // 2 + recv_x


def elem_to_recv_pol(a): 
    recv = a // 2
    pol = a % 2
    return recv, pol

# Same Layout as GPU: [f, baseline, polY, polX]
def build_cpu_corr_in_gpu_layout(voltage_u8):
    x = unpack(voltage_u8)                     # [t,f,e]
    x = np.transpose(x, (1, 2, 0))            # [f,e,t]
    x = np.ascontiguousarray(x)
    x = x.reshape(NUM_LOCAL_FREQ, NR_RECEIVERS, NR_POLARIZATIONS, SAMPLES_PER_DATA_SET) #x[f, recv, pol, t]

    cpu_corr = np.zeros(
        (NUM_LOCAL_FREQ, NUM_BASELINES, NR_POLARIZATIONS, NR_POLARIZATIONS),
        dtype=np.complex128)

    # Compute correlations in the same order as the GPU kernel
    for recv_y in range(NR_RECEIVERS):
        Y = x[:, recv_y, :, :]

        for recv_x in range(recv_y + 1):
            X = x[:, recv_x, :, :]

            vis = np.einsum("fpt,fqt->fpq", Y, np.conj(X), optimize=True) # to not loss precision

            b = baseline_index(recv_y, recv_x)
            cpu_corr[:, b, :, :] = vis

    return cpu_corr # [f, baseline, polY, polX]

def extract_pair_from_gpu_layout(corr_gpu_layout, a, b):
    recv_a, pol_a = elem_to_recv_pol(a)
    recv_b, pol_b = elem_to_recv_pol(b)

    if recv_b <= recv_a:
        bidx = baseline_index(recv_a, recv_b)
        return corr_gpu_layout[:, bidx, pol_a, pol_b]
    else:
        bidx = baseline_index(recv_b, recv_a)
        return np.conj(corr_gpu_layout[:, bidx, pol_b, pol_a])


# Utils to read frames from raw files
def load_rawfile_payload_bytes(path):
    raw = np.fromfile(path, dtype=np.uint8)
    meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype=np.uint32)[0]) # 4 bytes of metadata size
    off = 4 + meta_size
    return raw[off:]

def load_network_payload_from_raw(path):
    payload = load_rawfile_payload_bytes(path)
    return payload.reshape(SAMPLES_PER_DATA_SET, NUM_LOCAL_FREQ, NUM_ELEMENTS)

def load_corr_payload_from_raw(path):
    payload = load_rawfile_payload_bytes(path)
    raw_i32 = np.frombuffer(payload.tobytes(), dtype=np.int32)
    corr_i32 = raw_i32.reshape(NUM_LOCAL_FREQ, NUM_BASELINES, NR_POLARIZATIONS, NR_POLARIZATIONS, 2)
    corr = corr_i32[..., 0].astype(np.int64) + 1j * corr_i32[..., 1].astype(np.int64)
    return corr_i32, corr


# get the correlations on the same layout 
voltage_frame = load_network_payload_from_raw(RAW_NET)
cpu_corr = build_cpu_corr_in_gpu_layout(voltage_frame)
gpu_corr = load_corr_payload_from_raw(RAW_CORR)[1]

print(f"CPU corr shape: {cpu_corr.shape}, GPU corr shape: {gpu_corr.shape}")


# Plots 
frequencies = np.linspace(300, 501.6, 672, endpoint=False)
n_cols = 4
n_plots = len(PAIRS_TO_PLOT)
n_rows = int(np.ceil(n_plots / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows), sharex=False, sharey=False)
axes_flat = np.atleast_1d(axes).flatten()

for idx, (a, b) in enumerate(PAIRS_TO_PLOT):
    cpu_pair = extract_pair_from_gpu_layout(cpu_corr, a, b)
    gpu_pair = extract_pair_from_gpu_layout(gpu_corr, a, b)


    ax = axes_flat[idx]
    ax.scatter(frequencies, np.abs(cpu_pair), label="CPU", s=15)
    ax.scatter(frequencies, np.abs(gpu_pair), label="GPU", s=3)
    ax.set_title(f"Amplitude ({a},{b})", fontsize=10)
    ax.set_xlabel("Frequency [MHz]", fontsize=8)
    ax.grid(True, alpha=0.5)
    ax.legend()

for k in range(n_plots, len(axes_flat)):
    axes_flat[k].axis("off")

fig.suptitle(
    f"Amp CPU-GPU antenna {ANT_REF}",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
plt.savefig(f"corr_cpu_gpu_amp_ant{ANT_REF}.png", dpi=300)
plt.close()


# Plot phase difference CPU-GPU
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows), sharex=False, sharey=False)
axes_flat = np.atleast_1d(axes).flatten()

for idx, (a, b) in enumerate(PAIRS_TO_PLOT):
    cpu_pair = extract_pair_from_gpu_layout(cpu_corr, a, b)
    gpu_pair = extract_pair_from_gpu_layout(gpu_corr, a, b)

    phase_diff = np.angle(cpu_pair * np.conj(gpu_pair))

    ax = axes_flat[idx]
    ax.scatter(frequencies, phase_diff, label="Phase Diff", s=15)
    ax.set_title(f"Phase Diff ({a},{b})", fontsize=10)
    ax.set_xlabel("Frequency [MHz]", fontsize=8)
    ax.grid(True, alpha=0.5)
    ax.legend()

for k in range(n_plots, len(axes_flat)):
    axes_flat[k].axis("off")

fig.suptitle({f"Phase Diff CPU-GPU antenna {ANT_REF}"}, fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
plt.savefig(f"corr_cpu_gpu_phase_ant{ANT_REF}.png", dpi=300)
plt.close()