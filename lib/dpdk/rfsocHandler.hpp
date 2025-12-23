#ifndef RFSOC_HANDLER_HPP
#define RFSOC_HANDLER_HPP

#include "Config.hpp"
#include "dpdkCore.hpp"
#include "prometheusMetrics.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>

#include "json.hpp"

#include <mutex>

class rfsocHandler : public dpdkRXhandler {
public:
    /// Default constructor
    rfsocHandler(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container, int port);

        int handle_packet(struct rte_mbuf* mbuf) override;
        void update_stats() override;

protected:

        // Output buffer

        Buffer* out_buf = nullptr;
        uint8_t* out_frame = nullptr;
        int out_frame_id = 0;

        // Layout 
        uint32_t packet_payload_size;
        uint32_t packets_per_frame;
        uint32_t packet_index_in_frame = 0;

        // Header offsets
        static constexpr uint32_t ETH_IP_UDP_HDR = 42;
        static constexpr uint32_t SEQ_OFFSET = 0;
        static constexpr uint32_t SEQ_SIZE = 8; // little endian uint64_t

        std::chrono::steady_clock::time_point last_report;

        // Lost packet metric
        uint64_t rx_lost_packets_total = 0;
        uint64_t rx_packets_total = 0;

        // Sequence tracking
        bool got_first_packet = false;
        uint64_t last_seq = 0;
        uint64_t cur_seq = 0;

        // Prometheus metrics
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_packets_total_metric;
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_packets_total_metric;
        
    };

inline rfsocHandler::rfsocHandler(kotekan::Config& config, const std::string& unique_name,
                                      kotekan::bufferContainer& buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port),
    rx_packets_total_metric(
        kotekan::prometheus::Metrics::instance().add_gauge(
            "rfsoc_rx_packets_total", "Total number of packets received by rfsoc handler",
            {"port"})),
    rx_lost_packets_total_metric(
        kotekan::prometheus::Metrics::instance().add_gauge(
            "rfsoc_rx_lost_packets_total", "Total number of packets lost in rfsoc handler",
            {"port"}))
    {
    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"));

    if (out_buf == nullptr) {
        FATAL_ERROR("rfsocHandler: Could not find output buffer {:s} in buffer container",
                    config.get<std::string>(unique_name, "out_buf"));
    }

    out_buf->register_producer(unique_name.c_str());

    packet_payload_size = config.get<uint32_t>(unique_name, "packet_payload_size");
    packets_per_frame = config.get<uint32_t>(unique_name, "packets_per_frame");

    last_report = std::chrono::steady_clock::now();
}
int rfsocHandler::handle_packet(struct rte_mbuf* mbuf) {

    // Sequence number is a little endian uint64_t at offset 42 + SEQ_OFFSET
    uint64_t seq_num_le;
    std::memcpy(&seq_num_le,
                rte_pktmbuf_mtod_offset(mbuf, uint8_t*, ETH_IP_UDP_HDR + SEQ_OFFSET),
                SEQ_SIZE);
    cur_seq = le64toh(seq_num_le);

    INFO("rfsocHandler: Port {:d} - Received packet with seq: {:d}", port, cur_seq);
    if (!got_first_packet) {
        last_seq = cur_seq;
        got_first_packet = true;
    } else {
        if (cur_seq > last_seq + 1) {
            rx_lost_packets_total += (cur_seq - (last_seq - 1));
        }else if (cur_seq < last_seq) {
            // Out of order packet
            WARN("rfsocHandler: Port {:d} got out of order packet. Current seq: {:d}, last seq: {:d}",
                 port, cur_seq, last_seq);
        }
    }
    last_seq = cur_seq;
    rx_packets_total++;

    if (!out_frame) {
        out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            return -1;
        packet_index_in_frame = 0;
    
    }

    // Copy payload
    uint32_t payload_offset = ETH_IP_UDP_HDR + SEQ_SIZE;
    uint32_t dst_offset = packet_index_in_frame * packet_payload_size;

    copy_block(&mbuf, out_frame+dst_offset, packet_payload_size, (int*)&payload_offset);

    packet_index_in_frame++;

    if (packet_index_in_frame == packets_per_frame) {
        out_buf->mark_frame_full(unique_name.c_str(), out_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
        out_frame = nullptr;
        packet_index_in_frame = 0;
    }
    return 0;
}

inline void rfsocHandler::update_stats() {

    std::vector<std::string> port_label = {std::to_string(port)};

    rx_packets_total_metric.labels(port_label).set(rx_packets_total);
    rx_lost_packets_total_metric.labels(port_label).set(rx_lost_packets_total);

    //Print every 10 seconds
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_report);
    if (duration.count() >= 5) {
        INFO("rfsocHandler: Port {:d} - Total packets received: {:d}, Total packets lost: {:d}",
             port, rx_packets_total, rx_lost_packets_total); last_report = now;
    }
}

#endif // RFSOC_HANDLER_HPP
