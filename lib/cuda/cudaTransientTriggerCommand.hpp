#ifndef CUDA_TRANSIENT_TRIGGER_COMMAND_HPP
#define CUDA_TRANSIENT_TRIGGER_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaTransientTrigger.hpp"
#include "driver_types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaTransientTriggerCommand
 * @brief High-performance CUDA processing and ring buffering stage that continuously monitors
 *        cleaned baseband beams for transient events using Spectral Kurtosis and Inter-Beam
 *        Cross-Correlation, triggering automated candidate disk dumps from GPU VRAM.
 */
class cudaTransientTriggerCommand : public cudaCommand {
public:
    cudaTransientTriggerCommand(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers,
                                cudaDeviceInterface& device, int inst);
    ~cudaTransientTriggerCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    static void set_shared_config(const TransientTriggerConfig& config);
    static TransientTriggerConfig get_shared_config();
    static void trigger_manual();

private:
    void allocate_device_buffers();
    void free_device_buffers();
    void dispatch_candidate_dump(uint32_t trigger_frame, const TransientFrameMetrics& metrics);

    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _buffer_depth;
    int32_t _max_beams;

    std::string _gpu_mem_cleaned_beams;
    std::string _gpu_mem_output;

    // GPU VRAM Buffers
    float2* _d_ring_buffer = nullptr;
    float2* _d_metrics = nullptr;

    // Pinned Host Buffers
    float2* _h_metrics_pinned = nullptr;
    float2* _h_dump_pinned = nullptr;

    // Dedicated Stream for non-blocking candidate dumps
    cudaStream_t _dump_stream = nullptr;

    // Ring buffer state
    uint32_t _ring_buffer_depth = 32;
    uint32_t _pre_trigger_frames = 4;
    uint32_t _post_trigger_frames = 4;
    std::size_t _frame_elements = 0;
    std::size_t _frame_bytes = 0;

    uint32_t _frames_processed = 0;

    // Dump tracking state
    bool _dump_pending = false;
    uint32_t _dump_trigger_frame = 0;
    uint32_t _dump_target_frame = 0;
    TransientFrameMetrics _dump_metrics;

    // Thread-safe shared configuration and telemetry
    static std::mutex _global_mutex;
    static TransientTriggerConfig _shared_config;
    static bool _endpoints_registered;
    static std::atomic<bool> _manual_trigger_requested;
    static uint32_t _total_triggers_fired;
    static TransientFrameMetrics _last_trigger_metrics;
    static std::string _last_dump_path;
};

} // namespace kotekan

#endif // CUDA_TRANSIENT_TRIGGER_COMMAND_HPP
