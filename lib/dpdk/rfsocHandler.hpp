#ifndef RFSOC_HANDLER_HPP
#define RFSOC_HANDLER_HPP

#include "Config.hpp"
#include "dpdkCore.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "prometheusMetrics.hpp"
#include "chordMetadata.hpp"
#include "BasebandMetadata.hpp"   // for BasebandMetadata
#include "json.hpp"
#include <util.h>
#include <packet_copy.h>
#include <endian.h>
#include <arpa/inet.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>  

// This is the RFoC Handler Class for the 32 antenna CHARTS deployment

class rfsocHandler : public dpdkRXhandler {
public:
    /// Default constructor
    rfsocHandler(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container, int port);

        int handle_packet(struct rte_mbuf* mbuf) override;
        void update_stats() override {};
        
protected:

        // Output buffer
        Buffer* out_buf = nullptr;
        uint8_t* out_frame = nullptr;
        int out_frame_id = 0;

        // Config packet constants
        static constexpr uint32_t ETH_IP_UDP_HDR = 42;
        uint32_t packet_size = 5482; // ip/udp + rfsoc header + payload


        
        bool zero_new_frames = true;
        uint64_t packet_index = 0;

        uint32_t packets_written_in_frame = 0;
        

        //RFSoC parameters
        static constexpr uint32_t subbands = 4;
        uint32_t bytes_per_sb = 0;
        uint32_t bytes_per_spec = 0;
        uint32_t samples_per_spec = 0;
        static constexpr uint32_t rfsoc_header = 64;
        uint32_t payload_len = packet_size - ETH_IP_UDP_HDR - rfsoc_header;
        uint32_t payload_offset = ETH_IP_UDP_HDR + rfsoc_header;

        // Packet tracking
        bool first_packet = false;
        uint64_t cur_seq = 0;
        uint64_t last_seq = 0;

        uint64_t cur_spec = 0;
        uint64_t last_spec = 0;
        uint64_t frame_start_spec = 0;

        uint8_t subband = 0;
        uint32_t timestamp_sec = 0;
        uint32_t timestamp_micro = 0;
        
        uint64_t frame_start_seq = 0;
        uint32_t header_offset = 0;
        
        // Lost packet metric
        uint64_t rx_lost_packets_total = 0;
        uint64_t rx_packets_total = 0;
        uint64_t rx_bytes_total = 0;
        uint64_t rx_out_of_order_total = 0;
        uint64_t rx_error_total = 0;
        uint64_t rx_len_error_total = 0;

        uint64_t rx_bytes_last = 0;
        uint64_t rx_lost_packets_last = 0;
        uint64_t rx_out_of_order_last = 0;
        uint64_t rx_packets_last = 0;

        Buffer* mask_buf = nullptr; //to track the missing samples
        uint8_t* mask_frame = nullptr;
        int mask_frame_id = 0;

        std::string mask_name;
        std::vector<uint8_t> packets_seen; // to track received packets in a spectrum

        uint64_t num_frames_captured;
        uint64_t capture_n_frames;

        // Alignment (startup)
        bool got_first_packet = false;
        uint64_t alignment = 0;  

        // For test and start
        double warmup_time = 3.0; // seconds
        std::chrono::steady_clock::time_point start_time;
        bool in_warmup = true;

        std::chrono::steady_clock::time_point last_report;
        std::chrono::steady_clock::time_point last_rate_time;
        std::chrono::steady_clock::time_point last_stats_time;


        // Baseband Metadata pointer
        uint64_t event_id = 0;
        uint64_t freq_id = 0;

        inline uint64_t extract_seq_le64(const uint8_t* p) const {
            uint64_t v = 0;
            std::memcpy(&v, p, sizeof(uint64_t));
            return le64toh(v);
        }
        
        inline uint8_t extract_subband_le8(const uint8_t* p) const {
            uint8_t s = 0;
            std::memcpy(&s, p, sizeof(uint8_t));
            return s;
        }

        inline uint64_t extract_timestamp_le32(const uint8_t* p) const {
            uint64_t t = 0;
            std::memcpy(&t, p, sizeof(uint32_t));
            return le32toh(t);
        }


        inline bool check_packet_basic(struct rte_mbuf* mbuf){
            if (unlikely(mbuf == nullptr)) {
                rx_error_total +=1;
                return false;
        }

        const uint32_t pkt_len = rte_pktmbuf_pkt_len(mbuf);
        if (unlikely(pkt_len != packet_size)) {
            rx_error_total +=1;
            rx_len_error_total +=1;
            return false;
        }

        rx_packets_total +=1;
        rx_bytes_total += pkt_len;
        return true;
    };



    inline void print_status() {    
        const auto now = std::chrono::steady_clock::now();
        const double dt =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - last_stats_time).count(); 
            
        if (dt < 1.0) {
            return;
        }

        const uint64_t pkts_diff = rx_packets_total - rx_packets_last;
        const uint64_t lost_diff = rx_lost_packets_total - rx_lost_packets_last;
        const uint64_t ooo_diff  = rx_out_of_order_total - rx_out_of_order_last; 
        const uint64_t bytes_diff = rx_bytes_total - rx_bytes_last;    // Rates
        const double pps = pkts_diff / dt;
        const double loss_pps = lost_diff / dt;
        const double mbps = 8.0 * bytes_diff / dt / 1e6;  
        const uint64_t expected_pkts = pkts_diff + lost_diff;
        const double loss_pct =
            (expected_pkts > 0) ? (100.0 * lost_diff / expected_pkts) : 0.0;   
            
        WARN(
        "rfsocHandler | {:.2f}s | "
        "RX: {:.0f} pkt/s | Lost: {:.2f} pkt/s ({:.3f}%) | "
        "OOO: {:.2f} pkt/s | Rate: {:.2f} Mbps | ErrTot: {:d}",
        dt,
        pps,
        loss_pps, loss_pct,
        ooo_diff / dt,
        mbps,
        rx_error_total
        );   
         // Update snapshots
        rx_packets_last = rx_packets_total;
        rx_lost_packets_last = rx_lost_packets_total;
        rx_out_of_order_last = rx_out_of_order_total;
        rx_bytes_last = rx_bytes_total;
        last_stats_time = now;
    }
    bool align_first_packet(uint64_t seq);

    inline bool handle_lost_samples(int64_t lost_units);

    bool advance_frame(uint64_t new_seq, bool first_time=false);

    bool copy_packet(struct rte_mbuf* mbuf);
};

inline rfsocHandler::rfsocHandler(kotekan::Config& config, const std::string& unique_name,
                                      kotekan::bufferContainer& buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port){

        out_buf = buffer_container.get_buffer(
            config.get<std::string>(unique_name, "out_buffer"));

        if (!out_buf)
            FATAL_ERROR("rfsocHandlerCPT: Could not find output buffer {:s} for handler {:s}",
                        config.get<std::string>(unique_name, "out_buffer"), unique_name);
        
        out_buf->register_producer(unique_name.c_str());

        mask_buf = buffer_container.get_buffer(
            config.get<std::string>(unique_name, "mask_buf"));

        if (!mask_buf)
            FATAL_ERROR("rfsocHandler: Could not find mask buffer {:s} for handler {:s}",
                        config.get<std::string>(unique_name, "mask_buf"), unique_name);

        mask_name = unique_name + "_mask";
        mask_buf->register_producer(mask_name.c_str());
        mask_buf->zero_frames();

        alignment = config.get_default<uint64_t>(unique_name, "alignment", 0);
    
        // Number of frames to capture before stopping, 0 = unlimited
        capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);
        
        // [FIX] Inicializar contador de frames a 0
        num_frames_captured = 0;

        first_packet = false;
        last_report = std::chrono::steady_clock::now();
        last_rate_time = std::chrono::steady_clock::now();
        last_stats_time = std::chrono::steady_clock::now();
        rx_bytes_last = rx_bytes_total;
        header_offset = payload_offset;
        
        bytes_per_sb = payload_len;
        bytes_per_spec = subbands * bytes_per_sb; 

        packets_seen.resize(mask_buf->frame_size, 0); // to track received packets in a spectrum

        start_time = std::chrono::steady_clock::now();

} ;


inline int rfsocHandler::handle_packet(struct rte_mbuf* mbuf) {


    if (in_warmup) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
        if (elapsed_time >= warmup_time) {
            in_warmup = false;
            INFO("rfsocHandler: Warmup period of {:.2f} seconds ended. Starting capture.", warmup_time);
        } else {
            return 0; // discard packets during warmup
        }
    }

    if (unlikely(!check_packet_basic(mbuf))) return 0;

    const uint8_t* pkt = rte_pktmbuf_mtod(mbuf, const uint8_t*);
    cur_seq = extract_seq_le64(pkt + ETH_IP_UDP_HDR);
    subband = extract_subband_le8(pkt + ETH_IP_UDP_HDR + 8);
    cur_spec = (cur_seq >> 2);  //  one spec are 4 packets

    timestamp_sec = extract_timestamp_le32(pkt + ETH_IP_UDP_HDR + 9);
    timestamp_micro = extract_timestamp_le32(pkt + ETH_IP_UDP_HDR + 13);

    if (unlikely(subband >= subbands)) {    
        INFO("rfsocHandler: Invalid subband number {:d} in packet. Discarding packet.", subband);
        rx_error_total +=1; 
        return 0;
    }


    if (unlikely(!got_first_packet)) {
        if (!align_first_packet(cur_seq)) {
            return 0;   // descartar
        }
    }
    const int64_t seq_diff = (int64_t)cur_seq - (int64_t)last_seq;
    if (unlikely(seq_diff <= 0)) {

        rx_out_of_order_total++;
        return 0;
    }

    if (unlikely(seq_diff > 1)) {
        if (unlikely(!handle_lost_samples(seq_diff - 1)))
            return -1;
        //printf("RFSOC Handler: Lost packets detected. Lost: %ld\n", seq_diff - 1);
    }


    if (unlikely(!copy_packet(mbuf))) {
        INFO("rfsocHandler: Stopping capture due to frame advance failure.");
        return -1;
    }
    print_status();

    last_seq = cur_seq;

    return 0;
}

inline bool rfsocHandler::align_first_packet(uint64_t seq) {

    if (alignment == 0){
        FATAL_ERROR("rfsocHandler: Alignment parameter must be set and greater than zero.");
    }

    if ((seq % alignment) <= 1000) {
            INFO("rfsocHandler: Aligned at sequence {:d}", seq);
            last_seq = seq - seq % alignment; // The alignment must be multiple of the subbands
            got_first_packet = true;

            const uint32_t start_spec = (last_seq >> 2); // spec start (4 packets)

            if (unlikely(!advance_frame(start_spec, true))) {
                got_first_packet = false;
                return false;
            }
            return true;
        }
        return false;
    }



inline bool rfsocHandler::handle_lost_samples(int64_t lost_units) {

    uint64_t missing_seq = (uint64_t)last_seq + 1;

    while (lost_units > 0) {

        const uint64_t miss_spec = (missing_seq >> 2); // spec missing (4 packets)

        const uint64_t spec_offset = miss_spec - frame_start_spec;

        // Check if we need to advance the frame
        if (spec_offset * bytes_per_spec >= out_buf->frame_size) {
            if (!advance_frame(miss_spec)) return false;
        }

        rx_lost_packets_total++;
        missing_seq++;
        lost_units--;
    }
    return true;
}


inline bool rfsocHandler::advance_frame(uint64_t new_spec, bool first_time) {

    std::fill(packets_seen.begin(), packets_seen.end(), 0); // reset tracking
    if (!first_time) {
        //out_buf->allocate_new_metadata_object(out_frame_id);
        out_buf->mark_frame_full(unique_name, out_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;

        // Mask buffer handling
        mask_buf->mark_frame_full(mask_name.c_str(), mask_frame_id);
        mask_frame_id = (mask_frame_id + 1) % mask_buf->num_frames;

        num_frames_captured++;

    }
    if (capture_n_frames != 0 && num_frames_captured >= capture_n_frames) {
        INFO("rfsocHandler: Reached the configured number of frames to capture ({:d}). Stopping capture.",
             capture_n_frames);
        return false; // stop capturing
    }

    out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
    if (out_frame == nullptr) {
        return false;
    }

    mask_frame = mask_buf->wait_for_empty_frame(mask_name.c_str(), mask_frame_id);
    if (mask_frame == nullptr) {
        return false;
    }

    std::memset(mask_frame, 1, mask_buf->frame_size);
    frame_start_spec = new_spec;

    // Set metadata values
    BasebandMetadata* out_metadata = nullptr;
    out_buf->allocate_new_metadata_object(out_frame_id);
    out_metadata = (BasebandMetadata*)(out_buf->get_metadata(out_frame_id).get());
    out_metadata->event_id = 0;
    out_metadata->freq_id = 0;

    return true;
}


inline bool rfsocHandler::copy_packet(struct rte_mbuf* mbuf) {

    const uint64_t spec_id = cur_seq >> 2; // one spec are 4 packets

    if (spec_id < frame_start_spec) {
        rx_out_of_order_total++;
        return true; 
    }

    uint64_t spec_loc = spec_id - frame_start_spec; // location in specs

    if (spec_loc >= packets_seen.size()) {
        if (!advance_frame(spec_id)) return false;
        spec_loc = 0;
    }

    const size_t byte_offset = spec_loc * bytes_per_spec + subband * bytes_per_sb; // location in bytes
    int pkt_offset = payload_offset;

    if (unlikely(byte_offset + bytes_per_sb > out_buf->frame_size)) {
        rx_error_total +=1;
        return true; // fuera de rango
    }

    copy_block(&mbuf, &out_frame[byte_offset], bytes_per_sb, &pkt_offset);


    // Track received packets to the mask buffer
    uint8_t& seen = packets_seen[spec_loc];
    const uint8_t bit = uint8_t(1u << subband);
    if (seen & bit) {
        // duplicado de esa subband
        return true;
    }
    seen |= bit;

    if (seen == 0x0F) { // 4 subbands recibidas
        mask_frame[spec_loc] = 0;
        printf("Frame spec complete\n");
    }
    return true;
}


#endif 

// RFSOC_HANDLER_CPT_HPP