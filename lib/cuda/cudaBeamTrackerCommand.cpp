#include "cudaBeamTrackerCommand.hpp"
#include "cudaBeamTrackerV5.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;
using kotekan::cudaBeamTrackerCommand;

REGISTER_CUDA_COMMAND(cudaBeamTrackerCommand);

namespace kotekan {

std::mutex cudaBeamTrackerCommand::_global_mutex;
MultiBeamTrackerConfig cudaBeamTrackerCommand::_shared_config;
bool cudaBeamTrackerCommand::_endpoints_registered = false;

cudaBeamTrackerCommand::cudaBeamTrackerCommand(
    Config& config, const std::string& unique_name,
    bufferContainer& host_buffers, cudaDeviceInterface& device, int inst)
    : cudaCommand(config, unique_name, host_buffers, device, inst,
                  no_cuda_command_state, "cudaBeamTrackerCommand") {

    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _max_beams = config.get_default<int>(unique_name, "max_beams", 1);
    _integration_spectra = config.get_default<int>(unique_name, "integration_spectra", 320);
    _spacing_m = config.get_default<float>(unique_name, "spacing_m", 0.6f);

    _gpu_mem_voltage = config.get_default<std::string>(unique_name, "gpu_mem_voltage", "voltage");
    _gpu_mem_intensity = config.get_default<std::string>(unique_name, "gpu_mem_intensity", "intensity");

    const float l0 = config.get_default<float>(unique_name, "source_l0", 0.0f);
    const float m0 = config.get_default<float>(unique_name, "source_m0", 0.0f);
    const float dl = config.get_default<float>(unique_name, "source_dl", 0.0f);
    const float dm = config.get_default<float>(unique_name, "source_dm", 0.0f);

    const float trans_sq = l0 * l0 + m0 * m0;
    const float n0 = (trans_sq <= 1.0f) ? std::sqrt(1.0f - trans_sq) : 0.0f;

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.num_active_beams = config.get_default<int>(unique_name, "initial_active_beams", 1);
        _shared_config.trajectories[0].direction_start = Direction3D{l0, m0, n0};
        _shared_config.trajectories[0].direction_rate_per_sample = DirectionRate2D{dl, dm};
        _shared_config.integration_spectra = static_cast<std::size_t>(_integration_spectra);
        _shared_config.spacing_m = _spacing_m;
        _shared_config.time_chunk_size = static_cast<std::size_t>(config.get_default<int>(unique_name, "time_chunk_size", 80));
        _shared_config.time_unroll = static_cast<std::size_t>(config.get_default<int>(unique_name, "time_unroll", 8));
        _shared_config.enable_cuda_graph = config.get_default<bool>(unique_name, "enable_cuda_graph", false);

        // Register Dynamic REST Endpoints once
        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. Update Trajectory for a specific beam slot
            rest.register_post_callback("/beam_tracker/set_trajectory", [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::size_t beam_id = j.value("beam_id", 0);
                    if (beam_id >= MAX_TRACKER_BEAMS) {
                        conn.send_error("beam_id exceeds MAX_TRACKER_BEAMS", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    std::lock_guard<std::mutex> lk(_global_mutex);
                    if (j.contains("source_l0")) _shared_config.trajectories[beam_id].direction_start.x = j["source_l0"];
                    if (j.contains("source_m0")) _shared_config.trajectories[beam_id].direction_start.y = j["source_m0"];
                    if (j.contains("source_dl")) _shared_config.trajectories[beam_id].direction_rate_per_sample.dl = j["source_dl"];
                    if (j.contains("source_dm")) _shared_config.trajectories[beam_id].direction_rate_per_sample.dm = j["source_dm"];

                    const float lx = _shared_config.trajectories[beam_id].direction_start.x;
                    const float my = _shared_config.trajectories[beam_id].direction_start.y;
                    const float tsq = lx * lx + my * my;
                    _shared_config.trajectories[beam_id].direction_start.z = (tsq <= 1.0f) ? std::sqrt(1.0f - tsq) : 0.0f;

                    INFO_NON_OO("Beam Tracker: Updated trajectory for beam slot {:d} (l0={:.5f}, m0={:.5f})", beam_id, lx, my);
                    conn.send_text_reply(fmt::format("Trajectory updated for beam slot {:d}\n", beam_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            });

            // 2. Enable / set number of active beams
            rest.register_post_callback("/beam_tracker/enable_beam", [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::size_t count = j.value("num_active_beams", 1);
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.num_active_beams = std::min(count, MAX_TRACKER_BEAMS);
                    INFO_NON_OO("Beam Tracker: Active beams set to {:d}", _shared_config.num_active_beams);
                    conn.send_text_reply(fmt::format("Active beams set to {:d}\n", _shared_config.num_active_beams));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            });

            // 3. Status Query
            rest.register_get_callback("/beam_tracker/status", [](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(_global_mutex);
                reply["num_active_beams"] = _shared_config.num_active_beams;
                reply["max_beams_capacity"] = MAX_TRACKER_BEAMS;
                reply["integration_spectra"] = _shared_config.integration_spectra;
                reply["spacing_m"] = _shared_config.spacing_m;

                reply["beams"] = nlohmann::json::array();
                for (std::size_t b = 0; b < _shared_config.num_active_beams; ++b) {
                    nlohmann::json b_info;
                    b_info["beam_id"] = b;
                    b_info["source_l0"] = _shared_config.trajectories[b].direction_start.x;
                    b_info["source_m0"] = _shared_config.trajectories[b].direction_start.y;
                    b_info["source_n0"] = _shared_config.trajectories[b].direction_start.z;
                    b_info["source_dl"] = _shared_config.trajectories[b].direction_rate_per_sample.dl;
                    b_info["source_dm"] = _shared_config.trajectories[b].direction_rate_per_sample.dm;
                    reply["beams"].push_back(b_info);
                }
                conn.send_json_reply(reply);
            });

            _endpoints_registered = true;
        }
    }

    const double freq_start_hz = config.get_default<double>(unique_name, "freq_start_hz", 300.0e6);
    const double freq_step_hz = config.get_default<double>(unique_name, "freq_step_hz", 300.0e3);

    _frequencies_hz.resize(static_cast<std::size_t>(_num_local_freq));
    for (int f = 0; f < _num_local_freq; ++f) {
        _frequencies_hz[f] = freq_start_hz + f * freq_step_hz;
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaBeamTrackerCommand");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_intensity, false, true, true));
}

cudaEvent_t cudaBeamTrackerCommand::execute(
    cudaPipelineState& pipestate, const std::vector<cudaEvent_t>&) {

    pre_execute();

    MultiBeamTrackerConfig current_config;
    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        current_config = _shared_config;
    }

    // Instantaneous bypass when 0 active beams
    if (current_config.num_active_beams == 0) {
        return record_end_event();
    }

    const std::size_t input_bytes = static_cast<std::size_t>(_num_elements) *
                                    static_cast<std::size_t>(_num_local_freq) *
                                    static_cast<std::size_t>(_samples_per_data_set) *
                                    sizeof(int4x2_t);
    void* input_memory = device.get_gpu_memory_array(_gpu_mem_voltage, pipestate.gpu_frame_id,
                                                     _gpu_buffer_depth, input_bytes);

    const std::size_t output_bytes = static_cast<std::size_t>(_num_local_freq) *
                                     static_cast<std::size_t>(_samples_per_data_set) *
                                     static_cast<std::size_t>(_max_beams) *
                                     sizeof(float);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_intensity, pipestate.gpu_frame_id,
                                                      _gpu_buffer_depth, output_bytes);

    record_start_event();

    const std::size_t windows_per_frame = static_cast<std::size_t>(
        (_samples_per_data_set + current_config.integration_spectra - 1) / current_config.integration_spectra);
    const std::size_t window_offset = static_cast<std::size_t>(pipestate.gpu_frame_id) * windows_per_frame;

    launch_beam_tracker_v5_multibeam(
        reinterpret_cast<const int4x2_t*>(input_memory),
        reinterpret_cast<float*>(output_memory),
        static_cast<std::size_t>(_samples_per_data_set),
        static_cast<std::size_t>(_num_local_freq),
        static_cast<std::size_t>(_num_elements),
        static_cast<std::size_t>(_max_beams),
        _frequencies_hz,
        current_config,
        device.getStream(cuda_stream_id),
        window_offset);

    return record_end_event();
}

} // namespace kotekan
