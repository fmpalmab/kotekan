#ifndef CUDA_BEAM_TRACKER_V5_HPP
#define CUDA_BEAM_TRACKER_V5_HPP

#include "DataType.hpp" // for kotekan::int4x2_t

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace kotekan {

constexpr std::size_t MAX_TRACKER_BEAMS = 8;
constexpr std::size_t MAX_TRACKER_ANTENNAS = 256;

// 3D vector for directions
struct Direction3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 1.0f;
};

// 2D vector for direction rates (dl/dt, dm/dt in direction cosine space)
struct DirectionRate2D {
    float dl = 0.0f;
    float dm = 0.0f;
};

// Trajectory definition
struct BeamTrackerTrajectory {
    Direction3D direction_start{0.0f, 0.0f, 1.0f};
    DirectionRate2D direction_rate_per_sample{0.0f, 0.0f};
};

// Single-beam configuration for backward compatibility
struct BeamTrackerConfig {
    BeamTrackerTrajectory trajectory;
    std::size_t integration_spectra = 320;
    float spacing_m = 0.6f;
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 8; // 2, 4, or 8
    bool enable_cuda_graph = false;
    std::array<uint8_t, MAX_TRACKER_ANTENNAS> antenna_mask;
    BeamTrackerConfig() { antenna_mask.fill(1); }
};

// Multi-beam configuration supporting dynamic active slot counts
struct MultiBeamTrackerConfig {
    std::size_t num_active_beams = 1;
    std::array<BeamTrackerTrajectory, MAX_TRACKER_BEAMS> trajectories;
    std::size_t integration_spectra = 320;
    float spacing_m = 0.6f;
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 8;
    bool enable_cuda_graph = false;
    std::array<uint8_t, MAX_TRACKER_ANTENNAS> antenna_mask;
    MultiBeamTrackerConfig() { antenna_mask.fill(1); }
};

/**
 * @brief Launch the V5 Beam Tracker kernel for a single beam.
 */
void launch_beam_tracker_v5(
    const int4x2_t* d_packed,
    float* d_intensity,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    const std::vector<double>& frequencies_hz,
    const BeamTrackerConfig& config,
    cudaStream_t stream = nullptr,
    std::size_t window_offset = 0);

/**
 * @brief Launch the V5 Beam Tracker kernel for multiple dynamic beams with pre-allocated output slots.
 *
 * @param d_packed              Device pointer to packed voltages [time][freq][antenna]
 * @param d_intensity           Device pointer to output intensities [time][freq][max_beams_allocated]
 * @param n_time                Number of time samples
 * @param n_freq                Number of frequency channels
 * @param n_ant                 Number of antennas
 * @param max_beams_allocated   Total allocated beam stride in output buffer (e.g. 4 or 8)
 * @param frequencies_hz        Vector of physical frequencies in Hz
 * @param config                Multi-beam configuration with num_active_beams and trajectories
 * @param stream                CUDA stream
 * @param window_offset         Time window index offset
 */
void launch_beam_tracker_v5_multibeam(
    const int4x2_t* d_packed,
    float* d_intensity,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t max_beams_allocated,
    const std::vector<double>& frequencies_hz,
    const MultiBeamTrackerConfig& config,
    cudaStream_t stream = nullptr,
    std::size_t window_offset = 0);

/**
 * @brief Persistent Batched Beam Tracker Stream with optional CUDA graph execution.
 */
class CudaBeamTrackerV5Stream {
public:
    CudaBeamTrackerV5Stream(
        std::size_t n_time_per_batch,
        std::size_t n_freq,
        std::size_t n_ant,
        const std::vector<double>& frequencies_hz,
        const BeamTrackerConfig& config);
    ~CudaBeamTrackerV5Stream();

    CudaBeamTrackerV5Stream(const CudaBeamTrackerV5Stream&) = delete;
    CudaBeamTrackerV5Stream& operator=(const CudaBeamTrackerV5Stream&) = delete;

    void process_batch(
        std::size_t window_offset,
        const int4x2_t* host_packed,
        float* host_intensity);

    void process_batch_device(
        std::size_t window_offset,
        const int4x2_t* d_packed,
        float* d_intensity);

    float last_kernel_time_ms() const;

    cudaStream_t get_stream() const;
    int4x2_t* device_packed_buffer();
    float* device_intensity_buffer();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kotekan

#endif // CUDA_BEAM_TRACKER_V5_HPP
