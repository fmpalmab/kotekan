#ifndef RFSOC_HANDLER_CPT_HPP
#define RFSOC_HANDLER_CPT_HPP

#include "Config.hpp"
#include "dpdkCore.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "vdif_functions.h"
#include "prometheusMetrics.hpp"
#include "chordMetadata.hpp"
#include "json.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>    


class rfsocHandlerCPT : public dpdkRXhandler {
public:
    /// Default constructor
    rfsocHandlerCPT(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container, int port);

        int handle_packet(struct rte_mbuf* mbuf) override;
        void update_stats() override {};
protected:

        // Output buffer

        Buffer* out_buf = nullptr;
        uint8_t* out_frame = nullptr;
        int out_frame_id = 0;

        // Header offsets
        static constexpr uint32_t ETH_IP_UDP_HDR = 42;
        static constexpr uint32_t VDIF_HEADER_LEN = 32;

        // Layout ;
        const uint32_t blocks_per_packet = 8;
        const uint32_t spectrum_size = 672;
        const uint32_t num_pols = 2;
        const uint32_t payload_size = blocks_per_packet * spectrum_size;
        const uint32_t vdif_packet_size = VDIF_HEADER_LEN + spectrum_size;
        uint32_t frame_offset_bytes = 0;
        
        // state
        uint64_t vdif_frame_counter = 0;

        // VDIF header
        uint32_t station_id;
        const uint8_t log_num_chan = 10; // 1024 channels
        const uint32_t bits_depth = 7;   // 8 bits per sample

        std::chrono::steady_clock::time_point last_report;

        // Lost packet metric
        uint64_t rx_lost_packets_total = 0;
        uint64_t rx_packets_total = 0;

        inline void write_vdif_header(uint8_t* vdif_packet, uint32_t pol,
                                     uint64_t frame_counter, uint64_t now_sec);

    };

inline rfsocHandlerCPT::rfsocHandlerCPT(kotekan::Config& config, const std::string& unique_name,
                                      kotekan::bufferContainer& buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port){

        out_buf = buffer_container.get_buffer(
            config.get<std::string>(unique_name, "out_buffer"));

        if (!out_buf)
            FATAL_ERROR("rfsocHandlerCPT: Could not find output buffer {:s} for handler {:s}",
                        config.get<std::string>(unique_name, "out_buffer"), unique_name);
        
        out_buf->register_producer(unique_name.c_str());

        station_id = config.get_default<uint32_t>(unique_name, "station_id", 0); // Check this    
        }

inline int rfsocHandlerCPT::handle_packet(struct rte_mbuf* mbuf) {

    rx_packets_total++; 

    if (!out_frame) {
        out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            return -1;
    }


    if (!out_frame) {
        ERROR("rfsocHandlerCPT: Could not get output frame {:d}", out_frame_id);
        return -1;
    }

    INFO("rfsocHandlerCPT: Processing packet for output frame {:d}", out_frame_id);

    uint64_t now_sec = static_cast<uint64_t>(time(nullptr));
    uint32_t payload_offset = ETH_IP_UDP_HDR;
 

    for (uint32_t block = 0; block < blocks_per_packet; ++block) {
        uint32_t pol = block % 2;
        uint8_t* vdif_packet = out_frame + frame_offset_bytes;

        if (frame_offset_bytes + vdif_packet_size > out_buf->frame_size) {
            ERROR("rfsocHandlerCPT: Not enough space in output buffer frame {:d} for VDIF packets",
                  out_frame_id);
            return -1;
        }

        // Write VDIF header
        write_vdif_header(vdif_packet, pol, vdif_frame_counter, now_sec);
        vdif_frame_counter++;

        int src_offset = payload_offset + block * spectrum_size;
        uint32_t pkt_len = rte_pktmbuf_pkt_len(mbuf);

        if (src_offset + spectrum_size > pkt_len) {
            ERROR("rfsocHandlerCPT: Packet too small to read spectrum data at offset {:d}, packet length {:d}",
                  src_offset, pkt_len);
            return -1;
        }

        // rte_pktmbuf_read(mbuf, src_offset, spectrum_size, vdif_packet + VDIF_HEADER_LEN);

        uint8_t tmp[672];
        rte_pktmbuf_read(mbuf, src_offset, spectrum_size, tmp);
        memcpy(vdif_packet + VDIF_HEADER_LEN, tmp, spectrum_size);

        frame_offset_bytes += vdif_packet_size;
    }

    // Move to next output frame
    INFO("VDIF Frame offset: {:d}, Out buffer frame size: {:d}", 
         frame_offset_bytes, out_buf->frame_size);

    if (frame_offset_bytes == out_buf->frame_size) {

        out_buf->allocate_new_metadata_object(out_frame_id);
        auto meta = get_chord_metadata(out_buf, out_frame_id);

        if (meta == nullptr) {
            ERROR("rfsocHandlerCPT: No chordMetadata found for output frame {:d}", out_frame_id);
            return -1;
        }
        meta->set_lost_timesamples(0); // No lost samples tracking yet

        out_buf->mark_frame_full(unique_name, out_frame_id);
        out_frame = nullptr;
        frame_offset_bytes = 0;
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
    }

    return 0;
}

inline void rfsocHandlerCPT::write_vdif_header(uint8_t* vdif_packet, uint32_t pol,
                                            uint64_t frame_counter, uint64_t now_sec) {

    struct VDIFHeader* vdif_header = (struct VDIFHeader*)vdif_packet;

    vdif_header->invalid = 0;
    vdif_header->legacy = 0;
    vdif_header->vdif_version = 1;
    vdif_header->data_type = 1;
    vdif_header->unused = 0;

    vdif_header->ref_epoch = 36;
    vdif_header->frame_len = vdif_packet_size / 8; // In 8 byte words
    vdif_header->data_frame = frame_counter & 0xFFFFFF;

    vdif_header->log_num_chan = log_num_chan;
    vdif_header->bits_depth = bits_depth;
    vdif_header->station_id = station_id;
    vdif_header->thread_id = pol;
    vdif_header->seconds = now_sec & 0x3FFFFFFF;

    vdif_header->edv = 0;  
    vdif_header->eud1 = vdif_header->eud2 = vdif_header->eud3 = vdif_header->eud4 = 0;

    vdif_header->seconds = now_sec & 0x3FFFFFFF;



}

#endif // RFSOC_HANDLER_CPT_HPP