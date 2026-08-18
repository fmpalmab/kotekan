# CHARTS-32 Kotekan

This directory contains the functional CHARTS-32 version of Kotekan and its supporting configurations and analysis tools.

The code targets the 32-element CHARTS model:

- 32 input elements
- 672 frequency channels
- NoClk RFSoC input handler
- Baseband recording and GPU-correlation pipelines
- Physical antenna ordering restored from the descending RFSoC element order

## Directory layout

### `config/`

- `32antennas_baseband.yaml` — baseband capture using `chartsBasebandReadout` and `BasebandWriter`.
- `32antennas_baseband_raw.yaml` — raw network-frame capture with `rawFileWrite`.
- `pipeline_32antennas_baseband_record.yaml` — runtime template used by the automatic raw-recording script.
- `32antennas_corr.yaml` — 32-element GPU-correlation pipeline.
- `pipeline_32antennas_corr.yaml` — GPU-correlation pipeline template.
- `32baseband_corr_and_baseband.yaml` — combined correlation/baseband configuration for development and validation.

### `examples/`

- `open_baseband_h5.ipynb` — inspect converted HDF5 baseband data, plot antenna spectra, and calculate correlations.
- `open_raw_corr.ipynb` — exploratory inspection of raw correlation data.

### `tests/`

- `run_32antennas_raw_record_NoClk.sh` — capture raw data for a requested duration and generate the 32-antenna spectrum grid.
- `run_32antennas_corr_gpu_NoClk.sh` — run the NoClk GPU-correlation capture and analysis pipeline.

## Main Python tools

- `convert_baseband_chunks_32.py` — convert large raw baseband captures into chunked HDF5 files and a virtual dataset.
- `convert_baseband_32.py` — simple monolithic raw-to-HDF5 converter.
- `open_h5_32.py` — command-line HDF5 inspection and quick spectrum/correlation plots.
- `plot_raw_baseband.py` — decode raw frames and generate the 32-antenna intensity spectrum grid.
- `plot_correlations_gpu.py` — read GPU correlation frames and plot selected visibility phases and matrices.
- `corr_gpu_vs_cpu.py` — compare CPU and GPU visibility phases from matching raw and correlation frames.

## Pipeline configurations

The two main configuration files are located in `charts/config/` and describe
complete Kotekan pipelines for the CHARTS-32 system. Both start with the same
RFSoC capture over DPDK, but produce different outputs:

- `32antennas_baseband.yaml` preserves the received voltages/baseband data and
  writes it to event-associated files.
- `32antennas_corr.yaml` reorders the voltages on the GPU, calculates
  visibilities between the 32 elements, integrates multiple results, and writes
  the correlations in binary format.

In the diagram below, each arrow represents the buffer name connecting two
stages:

    RFSoC/NIC
       |
       v
    dpdkCore + rfsocHandlerNoClk
       | network_capture_buf       (and mask_buffer_zeros)
       v
    zeroSamples
       | network_capture_buf with missing samples zeroed
       +--> chartsBasebandReadout --> baseband_output_buffer_0 --> BasebandWriter
       |
       +--> cudaProcess (GPU) --> host_correlation_buffer
                                      |
                                      +--> chartsAccumulate --> integrated_correlation_buffer
                                                                 |
                                                                 +--> rawFileWrite

The baseband and correlation branches belong to separate configuration files; the
diagram shows the two alternatives to make the split explicit.

### Common parameters

The following values describe the CHARTS-32 data format and are therefore not
ordinary tuning parameters:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `num_elements` | `32` | Number of input elements in the system (antennas in our case). |
| `n_channels_per_packet` | `168` | Number of channels transported in each RFSoC packet. |
| `packets_per_spectrum` | `4` | Number of packets forming one complete spectrum. |
| `num_local_freq` | `672` | Number of local frequency channels processed by the correlation pipeline. |
| `sample_size` | `21504` bytes | Size of a time block that `zeroSamples` can replace. This is derived from the RFSoC packet format. |

These values should not be changed in isolation: doing so would also change buffer
sizes and the interpretation of the data by `rfsocHandlerNoClk`, the CUDA kernels,
and the analysis tools. They should only be changed together with the corresponding
hardware or packet format.

The YAML files express derived sizes so that Kotekan calculates them:

    bytes_per_dataset = n_channels_per_packet * packets_per_spectrum * samples_per_data_set
    network_capture_buf.frame_size = bytes_per_dataset * num_elements

`buffer_depth` controls how many frames are kept in each buffer. Increasing it
provides more headroom against bursts or delays, but consumes more RAM; for the GPU
pipeline, GPU memory must also be considered. `cpu_affinity`, `numa_node`,
`lcore_cpu_map`, `main_lcore_cpu`, `num_mbufs`, `max_rx_pkt_len`, and `gpu_id`
are deployment parameters: they can be adapted to the target machine, but must
match the NIC, CPU, NUMA, and GPU topology.

### `32antennas_baseband.yaml`

This configuration keeps a history of incoming data and performs a baseband readout:

1. `dpdkCore` receives packets on port `0` through `rfsocHandlerNoClk`. The
   handler places them in `network_capture_buf`, reconstructs complete frames,
   and generates CHARTS metadata. `alignment: 4000` makes the first frame start
   at a known alignment point (for this config, it is not strictly necessary).
2. The handler records spectra with missing packets in `mask_buffer_zeros`.
   `zeroSamples` uses this mask and replaces incomplete blocks in
   `network_capture_buf` with `zero_value: 0`, so packet loss is not interpreted
   as valid signal.
3. `chartsBasebandReadout` retains up to `buffer_depth - 10` input frames (this is based on the CHIME model) and
   copies the selected interval to `baseband_output_buffer_0`. In this version,
   the stage contains an internal test trigger: once the requested number of frames
   specified by `max_dump_samples` is available, it performs one readout of the
   history.
4. `BasebandWriter` consumes the output buffer. With
   `root_path: /data/32_test`, it creates directories such as
   `baseband_raw_<event_id>` and files organized by event and frequency.
   `dump_timeout` controls how long write destinations remain open without
   receiving new frames.

Parameters normally adjusted in this file:

| Parameter | Current value | Use and constraints |
| --- | ---: | --- |
| `samples_per_data_set` | `15360` | Spectra per input frame. It must match the size expected by the handler and `baseband_readout`. |
| `spectra_per_dataset` | `15360` | Temporal size used to dimension the buffer and mask; it must remain consistent with `samples_per_data_set`. |
| `buffer_depth` | `80` | Network-buffer depth; affects the available history and RAM usage. |
| `baseband_buffer_depth` | `buffer_depth - 10` | Baseband-output depth; normally leave this as a derived expression. |
| `capture_n_frames` | `20` | In the current config, used to calculate `max_dump_samples = samples_per_data_set * capture_n_frames`, the target size of the test trigger. It does not stop `dpdkCore`. |
| `root_path` | `/data/32_test` | Output directory; it must exist and be writable. |
| `dump_timeout` | `1` s | Timeout for closing inactive destinations in `BasebandWriter`. |
| `log_level` | `info` | Can be raised to `debug`/`debug2` for diagnostics; these levels require a Debug build. |

The value `baseband_save.samples_per_data_set: 10321920` is expressed in bytes
per element even though the parameter name says *samples*. It is the value of
`bytes_per_dataset`, and together with `num_elements: 32` makes the expected
written-frame size match the baseband buffer. If the dataset size changes, this
value must be updated as well.

### `32antennas_corr.yaml`

The correlation pipeline consists of these stages:

1. `dpdkCore`, `rfsocHandlerNoClk`, and `zeroSamples` perform the same capture,
   frame assembly, and packet-loss handling as in the baseband configuration.
2. `cudaProcess` takes each frame from `network_capture_buf` and executes
   `cudaInputData`, `cudaSyncInput`, `cudaShuffleAstron`,
   `cudaCorrelatorAstron`, `cudaSyncOutput`, and `cudaOutputData`, in that
   order. The shuffle adapts the input order to the correlator layout; the
   correlator computes the N2 correlation matrix for the 32 elements; and
   `cudaOutputData` copies the result to `host_correlation_buffer` together
   with the input-frame metadata.
3. `chartsAccumulate` sums `10` correlation frames and produces each integrated
   result in `integrated_correlation_buffer`. With the current values, this is
   the buffer inspected by `restInspectFrame` and written by `rawFileWrite`.
4. `rawFileWrite` stores one file per integrated frame under
   `base_dir: /data/corr/test1708_2`, with a `.bin` extension and
   `host_correlation` as the base name.

Main parameters in this pipeline:

| Parameter | Current value | Use and constraints |
| --- | ---: | --- |
| `samples_per_data_set` | `153600` | Temporal dataset size received by the correlator; the config comment requires it to be divisible by `32`. |
| `buffer_depth` | `30` | Depth of the network and correlation buffers; affects RAM and back-pressure headroom. |
| `gpu_buffer_depth` | `6` | The depth actively used by `gpu_0.buffer_depth`; it limits the number of frames in flight on the GPU. |
| `block_size` | `2` | CUDA-kernel tuning parameter. Changing it requires reviewing `num_blocks` and performance. |
| `num_blocks` | `136` | For 32 elements and `block_size: 2`, this is `(16 * 17) / 2`; recalculate it if those values change. |
| `integration_frames` | `10` | Number of frames to integrate before writing to disk. |
| `charts_accumulate.num_frames_to_accumulate` | `10` | Parameter that actually controls the current integration; update it together with `integration_frames`. |
| `profiling` | `false` | Enables or disables CUDA-pipeline profiling. |
| `base_dir` | `/data/corr/test1708_2` | Binary-output directory. |




### How to use the configs

Before starting either pipeline, Kotekan must be built with DPDK (and CUDA for
correlation), the RFSoC NIC/port must be configured correctly, and the output
directories must be writable. From the repository root:


#### 1. Baseband capture and writing
    sudo ./kotekan -c charts/config/32antennas_baseband.yaml

For 32antennas_baseband.yaml, the most important parameters to change are:  
* `root_path` to the directory where you want to store the baseband data. If you do not change it and it already contains old data, it will be concatenated with the new data.
* `capture_n_frames` to the number of frames you want to capture. The default value is 20, which corresponds to 0.1 seconds of data.
* `spectra_per_dataset` and `samples_per_data_set` ONLY IF YOU KNOW WHAT YOU ARE DOING. 


#### 2. GPU correlation and integrated-visibility writing
    sudo ./kotekan -c charts/config/32antennas_corr.yaml

For 32antennas_corr.yaml, the most important parameters to change are:
* `base_dir` to the directory where you want to store the correlation data. If you do not change it, it will be overwritten with new data.
* `integration_frames` will determine how many frames are integrated before writing to disk. This also depends on the `samples_per_data_set` parameter, which is 153600 by default. Therefore, the integration time is `integration_frames * samples_per_data_set * 10/3 us`. The number of output frames will be `capture_n_frames / integration_frames`.
* `capture_n_frames` to the number of frames you want to capture. This is aligned with `samples_per_data_set`, so the total capture time is `capture_n_frames * samples_per_data_set * 10/3 us`. 

With the current values, both configurations continue receiving data until Kotekan
is stopped, for example with `Ctrl-C`; the baseband test trigger and the
correlation integration are not stop conditions by themselves. After a run, data is
stored under `root_path` for baseband or `base_dir` for correlation. To inspect
the results, use `open_h5_32.py` after converting baseband data to HDF5, and use
`plot_correlations_gpu.py` for the correlator's binary frames. Also you can use the repo charts-utils to 
insepct the correlations.

## Raw recording example

```bash
./charts/tests/run_32antennas_raw_record_NoClk.sh 5
```

The duration is converted into the required number of capture frames. Previous raw files and outputs belonging to this pipeline are replaced automatically.

## Data ordering

Raw RFSoC element indices are descending. The analysis and conversion tools use the common CHARTS-32 convention:

```text
raw element 31 -> physical antenna 0
raw element 30 -> physical antenna 1
...
raw element 0  -> physical antenna 31
```

This is still under discussion, but it is the current convention for the CHARTS-32 system. The correlation pipeline does not change the order of the elements, so the output correlation matrices are in the same order as the input voltages. The analysis tools can reorder them for plotting and comparison with other systems.