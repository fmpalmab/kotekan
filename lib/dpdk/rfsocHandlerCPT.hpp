#ifndef RFSOC_HANDLER_CPT_HPP
#define RFSOC_HANDLER_CPT_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chartsMetadata.hpp"
#include "dpdkCore.hpp"
#include "json.hpp"
#include "prometheusMetrics.hpp"

#include <arpa/inet.h>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <endian.h>
#include <memory>
#include <rte_mbuf.h>
#include <string>
#include <sys/time.h>
#include <vector>

#include <util.h>

class rfsocHandlerCPT : public dpdkRXhandler {
public:
    rfsocHandlerCPT(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container, int port);

    int handle_packet(struct rte_mbuf* mbuf) override;
    void update_stats() override;

protected:
    Buffer* out_buf = nullptr;
    uint8_t* out_frame = nullptr;
    int out_frame_id = 0;

    Buffer* mask_buf = nullptr;
    uint8_t* mask_frame = nullptr;
    int mask_frame_id = 0;
    std::string mask_name;

    static constexpr uint32_t ETH_IP_UDP_HDR = 42;
    static constexpr uint32_t RFSOC_HEADER = 64;
    static constexpr uint32_t PACKET_SIZE = 5482;
    static constexpr uint32_t PAYLOAD_OFFSET = ETH_IP_UDP_HDR + RFSOC_HEADER;
    static constexpr uint32_t SPECTRUM_SIZE = 672;
    static constexpr uint32_t BLOCKS_PER_PACKET = 8;
    static constexpr uint32_t NUM_POL = 2;
    static constexpr uint32_t SAMPLES_PER_PACKET = BLOCKS_PER_PACKET / NUM_POL;
    static constexpr uint32_t SAMPLE_SIZE = NUM_POL * SPECTRUM_SIZE;

    bool zero_new_frames = true;
    uint64_t samples_per_frame = 0;
    uint64_t frame_start_seq = 0;
    uint64_t current_frame_valid_samples = 0;

    bool got_first_packet = false;
    bool got_time0_fpga = false;
    bool warned_time0_change = false;
    uint64_t cur_seq = 0;
    uint64_t last_seq = 0;
    uint64_t time0_fpga = 0;
    uint32_t timestamp_sec = 0;
    uint32_t timestamp_micro = 0;

    uint64_t alignment = 0;
    uint64_t num_frames_captured = 0;
    uint64_t capture_n_frames = 0;

    uint32_t num_local_freq = SPECTRUM_SIZE;
    uint32_t coarse_freq_start = 0;
    std::vector<int> frame_coarse_freq;

    uint64_t rx_lost_packets_total = 0;
    uint64_t rx_packets_total = 0;
    uint64_t rx_bytes_total = 0;
    uint64_t rx_out_of_order_total = 0;
    uint64_t rx_error_total = 0;
    uint64_t rx_len_error_total = 0;
    uint64_t rx_samples_total = 0;
    uint64_t rx_lost_samples_total = 0;

    uint64_t rx_bytes_last = 0;
    uint64_t rx_lost_packets_last = 0;
    uint64_t rx_packets_last = 0;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_samples_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_samples_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_bytes_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_error_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_len_error_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_out_of_order_total_metric;

    inline uint64_t extract_seq_le64(const uint8_t* p) const {
        uint64_t v = 0;
        std::memcpy(&v, p, sizeof(uint64_t));
        return le64toh(v);
    }

    inline uint32_t extract_timestamp_le32(const uint8_t* p) const {
        uint32_t t = 0;
        std::memcpy(&t, p, sizeof(uint32_t));
        return le32toh(t);
    }

    inline bool check_packet_basic(struct rte_mbuf* mbuf) {
        if (unlikely(mbuf == nullptr)) {
            rx_error_total++;
            return false;
        }

        const uint32_t pkt_len = rte_pktmbuf_pkt_len(mbuf);
        if (unlikely(pkt_len != PACKET_SIZE)) {
            rx_error_total++;
            rx_len_error_total++;
            return false;
        }

        rx_packets_total++;
        rx_bytes_total += pkt_len;
        return true;
    }

    bool open_frame(uint64_t new_frame_start_seq);
    bool close_frame();
    bool advance_to_seq(uint64_t seq);
    bool copy_packet(struct rte_mbuf* mbuf);
    void set_frame_metadata();
    void parse_header(struct rte_mbuf* mbuf);
};

inline rfsocHandlerCPT::rfsocHandlerCPT(kotekan::Config& config,
                                        const std::string& unique_name,
                                        kotekan::bufferContainer& buffer_container,
                                        int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port),
    rx_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_packets_total", unique_name, {"port"})),
    rx_samples_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_samples_total", unique_name, {"port"})),
    rx_lost_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_lost_packets_total", unique_name, {"port"})),
    rx_lost_samples_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_lost_samples_total", unique_name, {"port"})),
    rx_bytes_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_bytes_total", unique_name, {"port"})),
    rx_error_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_error_total", unique_name, {"port"})),
    rx_len_error_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_len_error_total", unique_name, {"port"})),
    rx_out_of_order_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_out_of_order_packets_total", unique_name, {"port"})) {

    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buffer"));
    if (!out_buf) {
        FATAL_ERROR("rfsocHandlerCPT: Could not find output buffer {:s} for handler {:s}",
                    config.get<std::string>(unique_name, "out_buffer"), unique_name);
    }
    out_buf->register_producer(unique_name.c_str());

    if (out_buf->metadata_pool == nullptr || out_buf->metadata_pool->type_name != "chartsMetadata") {
        FATAL_ERROR("rfsocHandlerCPT: Output buffer {:s} must use chartsMetadata.",
                    out_buf->buffer_name);
    }

    mask_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "mask_buf"));
    if (!mask_buf) {
        FATAL_ERROR("rfsocHandlerCPT: Could not find mask buffer {:s} for handler {:s}",
                    config.get<std::string>(unique_name, "mask_buf"), unique_name);
    }
    mask_name = unique_name + "_mask";
    mask_buf->register_producer(mask_name.c_str());
    mask_buf->zero_frames();

    if (mask_buf->metadata_pool == nullptr || mask_buf->metadata_pool->type_name != "chartsMetadata") {
        FATAL_ERROR("rfsocHandlerCPT: Mask buffer {:s} must use chartsMetadata.",
                    mask_buf->buffer_name);
    }

    if (out_buf->frame_size % SAMPLE_SIZE != 0) {
        FATAL_ERROR("rfsocHandlerCPT: Output frame size {:d} is not divisible by sample size {:d}.",
                    out_buf->frame_size, SAMPLE_SIZE);
    }
    samples_per_frame = out_buf->frame_size / SAMPLE_SIZE;

    if (mask_buf->frame_size != samples_per_frame) {
        FATAL_ERROR("rfsocHandlerCPT: Mask frame size {:d} must match samples per frame {:d}.",
                    mask_buf->frame_size, samples_per_frame);
    }

    alignment = config.get_default<uint64_t>(unique_name, "alignment", samples_per_frame);
    if (alignment == 0) {
        alignment = samples_per_frame;
    }

    capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);
    zero_new_frames = config.get_default<bool>(unique_name, "zero_new_frames", true);

    frame_coarse_freq.resize(num_local_freq);
    for (uint32_t i = 0; i < num_local_freq; ++i) {
        frame_coarse_freq[i] = static_cast<int>(coarse_freq_start + i);
    }

    rx_bytes_last = rx_bytes_total;
    INFO("rfsocHandlerCPT: Using chartsMetadata, samples_per_frame={:d}, sample_size={:d}.",
         samples_per_frame, SAMPLE_SIZE);
}

inline void rfsocHandlerCPT::parse_header(struct rte_mbuf* mbuf) {
    const uint8_t* pkt = rte_pktmbuf_mtod(mbuf, const uint8_t*);
    const uint8_t* rfsoc_header = pkt + ETH_IP_UDP_HDR;

    cur_seq = extract_seq_le64(rfsoc_header);
    timestamp_micro = extract_timestamp_le32(rfsoc_header + 9);
    timestamp_sec = extract_timestamp_le32(rfsoc_header + 13);

    const uint64_t packet_time0 =
        ((uint64_t)timestamp_sec) * 1000000ULL + ((uint64_t)timestamp_micro);
    if (!got_time0_fpga) {
        time0_fpga = packet_time0;
        got_time0_fpga = true;
        INFO("rfsocHandlerCPT: RFSoC time0 timestamp: {:d} us", time0_fpga);
    } else if (packet_time0 != time0_fpga && !warned_time0_change) {
        WARN("rfsocHandlerCPT: RFSoC time0 changed from {:d} to {:d}; keeping first value.",
             time0_fpga, packet_time0);
        warned_time0_change = true;
    }
}

inline void rfsocHandlerCPT::set_frame_metadata() {
    struct timeval now;
    gettimeofday(&now, nullptr);

    out_buf->allocate_new_metadata_object(out_frame_id);
    auto out_meta = std::dynamic_pointer_cast<chartsMetadata>(out_buf->get_metadata(out_frame_id));
    if (!out_meta) {
        FATAL_ERROR("rfsocHandlerCPT: Failed to cast output metadata to chartsMetadata.");
    }

    out_meta->set_first_packet_recv_time(now);
    out_meta->set_fpga_seq_num(frame_start_seq);
    out_meta->set_time_downsampling_fpga(1);
    out_meta->set_coarse_freq(frame_coarse_freq);
    out_meta->set_lost_timesamples(0);
    out_meta->set_time0_fpga(time0_fpga);
    out_meta->dims = 3;
    out_meta->type = kotekan::int4x2;
    out_meta->set_array_dimension(0, static_cast<int>(samples_per_frame), "T");
    out_meta->set_array_dimension(1, NUM_POL, "P");
    out_meta->set_array_dimension(2, SPECTRUM_SIZE, "F");
    out_meta->set_strides_simple();

    mask_buf->allocate_new_metadata_object(mask_frame_id);
    auto mask_meta =
        std::dynamic_pointer_cast<chartsMetadata>(mask_buf->get_metadata(mask_frame_id));
    if (!mask_meta) {
        FATAL_ERROR("rfsocHandlerCPT: Failed to cast mask metadata to chartsMetadata.");
    }

    mask_meta->set_first_packet_recv_time(now);
    mask_meta->set_fpga_seq_num(frame_start_seq);
    mask_meta->set_time_downsampling_fpga(1);
    mask_meta->set_coarse_freq(frame_coarse_freq);
    mask_meta->set_lost_timesamples(0);
    mask_meta->set_time0_fpga(time0_fpga);
    mask_meta->dims = 1;
    mask_meta->type = kotekan::uint8;
    mask_meta->set_array_dimension(0, static_cast<int>(samples_per_frame), "T");
    mask_meta->set_strides_simple();
}

inline bool rfsocHandlerCPT::open_frame(uint64_t new_frame_start_seq) {
    out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
    if (out_frame == nullptr) {
        rx_error_total++;
        return false;
    }

    mask_frame = mask_buf->wait_for_empty_frame(mask_name.c_str(), mask_frame_id);
    if (mask_frame == nullptr) {
        rx_error_total++;
        return false;
    }

    if (zero_new_frames) {
        std::memset(out_frame, 0, out_buf->frame_size);
    }
    std::memset(mask_frame, 1, mask_buf->frame_size);

    frame_start_seq = new_frame_start_seq;
    current_frame_valid_samples = 0;
    set_frame_metadata();
    return true;
}

inline bool rfsocHandlerCPT::close_frame() {
    const uint64_t lost_samples = samples_per_frame - current_frame_valid_samples;
    rx_lost_samples_total += lost_samples;

    auto out_meta = std::dynamic_pointer_cast<chartsMetadata>(out_buf->get_metadata(out_frame_id));
    if (out_meta) {
        out_meta->set_lost_timesamples(static_cast<int32_t>(lost_samples));
    }
    auto mask_meta =
        std::dynamic_pointer_cast<chartsMetadata>(mask_buf->get_metadata(mask_frame_id));
    if (mask_meta) {
        mask_meta->set_lost_timesamples(static_cast<int32_t>(lost_samples));
    }

    out_buf->mark_frame_full(unique_name, out_frame_id);
    mask_buf->mark_frame_full(mask_name.c_str(), mask_frame_id);

    out_frame = nullptr;
    mask_frame = nullptr;
    out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
    mask_frame_id = (mask_frame_id + 1) % mask_buf->num_frames;

    num_frames_captured++;
    if (capture_n_frames != 0 && num_frames_captured >= capture_n_frames) {
        INFO("rfsocHandlerCPT: Reached the configured number of frames to capture ({:d}).",
             capture_n_frames);
        return false;
    }
    return true;
}

inline bool rfsocHandlerCPT::advance_to_seq(uint64_t seq) {
    if (out_frame == nullptr || mask_frame == nullptr) {
        const uint64_t aligned_start = seq - (seq % alignment);
        return open_frame(aligned_start);
    }

    while (seq >= frame_start_seq + samples_per_frame) {
        const uint64_t next_frame_start = frame_start_seq + samples_per_frame;
        if (!close_frame()) {
            return false;
        }
        if (!open_frame(next_frame_start)) {
            return false;
        }
    }
    return true;
}

inline bool rfsocHandlerCPT::copy_packet(struct rte_mbuf* mbuf) {
    uint8_t pols_seen[SAMPLES_PER_PACKET] = {0}; // bitfield of which pols have been seen for each sample in the packet

    // Each packet contains 8 blocks of data, each block is one pol of one time sample. 
    // We loop through the blocks in order, which allows us to track which pols we've seen 
    // for each sample and mark samples as valid once we've seen both pols.
    for (uint32_t block = 0; block < BLOCKS_PER_PACKET; block++) {
        const uint32_t pol = block % NUM_POL;
        const uint32_t sample_in_packet = block / NUM_POL;
        const uint64_t spec_id = cur_seq + sample_in_packet;

        if (!advance_to_seq(spec_id)) {
            return false;
        }

        if (spec_id < frame_start_seq) {
            rx_out_of_order_total++;
            continue;
        }

        const uint64_t spec_loc = spec_id - frame_start_seq;
        const uint64_t dst_offset = spec_loc * SAMPLE_SIZE + pol * SPECTRUM_SIZE;
        if (unlikely(dst_offset + SPECTRUM_SIZE > (uint64_t)out_buf->frame_size)) {
            rx_error_total++;
            WARN("rfsocHandlerCPT: Port {:d}; destination offset {:d} exceeds frame size {:d}.",
                 port, dst_offset + SPECTRUM_SIZE, out_buf->frame_size);
            return false;
        }

        uint8_t* dst = out_frame + dst_offset;
        const uint32_t src_offset = PAYLOAD_OFFSET + block * SPECTRUM_SIZE;
        const void* ret = rte_pktmbuf_read(mbuf, src_offset, SPECTRUM_SIZE, dst);
        if (unlikely(ret == nullptr)) {
            rx_error_total++;
            WARN("rfsocHandlerCPT: Port {:d}; failed to read packet seq {:d}, block {:d}.",
                 port, cur_seq, block);
            return false;
        }

        if (ret != dst) {
            std::memcpy(dst, ret, SPECTRUM_SIZE);
        }

        pols_seen[sample_in_packet] |= uint8_t(1u << pol);
        if (pols_seen[sample_in_packet] == 0x03) {
            mask_frame[spec_loc] = 0;
            current_frame_valid_samples++;
            rx_samples_total++;
        }
    }

    return true;
}

inline int rfsocHandlerCPT::handle_packet(struct rte_mbuf* mbuf) {
    if (unlikely(!check_packet_basic(mbuf))) {
        return 0;
    }

    parse_header(mbuf);

    if (!got_first_packet) {
        got_first_packet = true;
        last_seq = cur_seq;
        INFO("rfsocHandlerCPT: Starting capture at packet seq {:d}, frame start {:d}.",
             cur_seq, cur_seq - (cur_seq % alignment));
        return copy_packet(mbuf) ? 0 : -1;
    }

    if (unlikely(cur_seq <= last_seq)) {
        rx_out_of_order_total++;
        return 0;
    }

    const uint64_t expected_seq = last_seq + SAMPLES_PER_PACKET;
    if (unlikely(cur_seq > expected_seq)) {
        const uint64_t missing_samples = cur_seq - expected_seq;
        const uint64_t missing_packets =
            (missing_samples + SAMPLES_PER_PACKET - 1) / SAMPLES_PER_PACKET;
        rx_lost_packets_total += missing_packets;
        DEBUG("rfsocHandlerCPT: Lost {:d} packet(s), missing {:d} sample(s), last_seq {:d}, "
              "cur_seq {:d}.",
              missing_packets, missing_samples, last_seq, cur_seq);
    }

    if (unlikely(!copy_packet(mbuf))) {
        return -1;
    }

    last_seq = cur_seq;
    return 0;
}

inline void rfsocHandlerCPT::update_stats() {
    std::vector<std::string> port_label = {std::to_string(port)};
    rx_packets_total_metric.labels(port_label).set(rx_packets_total);
    rx_samples_total_metric.labels(port_label).set(rx_samples_total);
    rx_lost_packets_total_metric.labels(port_label).set(rx_lost_packets_total);
    rx_lost_samples_total_metric.labels(port_label).set(rx_lost_samples_total);
    rx_bytes_total_metric.labels(port_label).set(rx_bytes_total);
    rx_error_total_metric.labels(port_label).set(rx_error_total);
    rx_len_error_total_metric.labels(port_label).set(rx_len_error_total);
    rx_out_of_order_total_metric.labels(port_label).set(rx_out_of_order_total);

    double time_now = e_time();
    static double last_status_message_time = 0.0;
    const double status_cadence = 1.0;

    if ((time_now - last_status_message_time) > status_cadence) {
        const uint64_t d_packets = rx_packets_total - rx_packets_last;
        const uint64_t d_lost = rx_lost_packets_total - rx_lost_packets_last;
        const uint64_t d_bytes = rx_bytes_total - rx_bytes_last;
        const double mbps = (double)d_bytes * 8.0 / (status_cadence * 1e6);

        INFO("CPT RFSoC port {:d} | RX pkt {:d} (+{:d}) | lost pkt {:d} (+{:d}) | "
             "lost samples {:d} | out-of-order {:d} | {:.2f} Mb/s | samples {:d}",
             port, rx_packets_total, d_packets, rx_lost_packets_total, d_lost,
             rx_lost_samples_total, rx_out_of_order_total, mbps, rx_samples_total);

        rx_packets_last = rx_packets_total;
        rx_lost_packets_last = rx_lost_packets_total;
        rx_bytes_last = rx_bytes_total;
        last_status_message_time = time_now;
    }
}

#endif // RFSOC_HANDLER_CPT_HPP
