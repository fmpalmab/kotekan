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
