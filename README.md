
# Documentation (CHARTS)

Compiled docs are available at https://kotekan.readthedocs.io/.

# Build Instructions

Detailed instructions are available at https://kotekan.readthedocs.io/latest/compiling/general.html

Full list of CMake options: https://kotekan.readthedocs.io/latest/compiling/cmake_options.html

This project is built using cmake, so you will need to install cmake
before starting a build.

To build just the base framework:

	cd build
	cmake <options> ..
	make

Building minimun kotekan for CHARTS

Cmake build options (defaults shown in parentheses; most feature toggles accept `AUTO`, `ON`, or `OFF`, with `AUTO` probing for dependencies and falling back gracefully):

* `-DUSE_CUDA=<AUTO|ON|OFF>` (`AUTO`) - Build the CUDA backend and enable CUDA stages when `nvcc` and the CUDA toolkit are available. Adds `-DWITH_CUDA` on success.

* `-DUSE_DPDK=<AUTO|ON|OFF>` (`AUTO`) - Enable DPDK stages when `libdpdk>=19.11` is present via `pkg-config`.

* `-DUSE_HDF5=<AUTO|ON|OFF>` (`AUTO`) - Enable HDF5 output stages when HDF5, HighFive, and the runtime plugin directory are *all* available. Populates `KOTEKAN_HDF5_PLUGIN_DIR` for runtime use.

* `-DUSE_NUMA=<AUTO|ON|OFF>` (`ON`) - Link libnuma and enable NUMA-aware buffer handling. Required when DPDK is enabled.

* `-DWITH_TESTS=<AUTO|ON|OFF>` (`OFF`) - Build and link the helper stages from `lib/testing` into the kotekan binary (used by QA/example configs). Does not build unit tests.

**Examples (implemented on CHARTS):**

    cmake -DUSE_CUDA=ON -DUSE_DPDK=ON -DUSE_HDF5=ON -DUSE_NUMA=ON -DWITH_TESTS=ON ..

At the end of configuration, CMake prints a colorized feature summary indicating which features were enabled (found) or disabled (missing/explicitly off). Each feature row shows its toggle flag, e.g. `CUDA: ON (found, toggle: -DUSE_CUDA=ON/OFF)`. Use `-D<OPTION>=AUTO|ON|OFF` to auto-detect, require, or disable a feature present on your system.

To install kotekan:

	make install

# Running kotekan

**Using systemd (full install)**

To start kotekan

    sudo systemctl start kotekan

To stop kotekan

    sudo systemctl stop kotekan

**To run in debug mode, run from `ch_gpu/build/kotekan/`**

    sudo ./kotekan -c <config_file>.yaml

For example:

    sudo ./kotekan -c ../../kotekan/kotekan_gpu_replay.yaml

When installed kotekan's config files are located at /etc/kotekan/

**This read me is based on the official Kotekan Readme