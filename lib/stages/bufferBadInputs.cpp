#include "bufferBadInputs.hpp"

#include "Config.hpp"         // for Config
#include "N2Util.hpp"         // for frameID
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"         // for Buffer
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for DEBUG, ERROR

#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <memory>     // for __shared_ptr_access, shared_ptr

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferBadInputs);

bufferBadInputs::bufferBadInputs(Config& config_, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&bufferBadInputs::main_thread, this)) {

    uint32_t num_elements = config.get<uint32_t>(unique_name, "num_elements");
    input_mask_len = sizeof(uint8_t) * num_elements;

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    N2::frameID frame_id(out_buf);
}

bufferBadInputs::~bufferBadInputs() {}

bool bufferBadInputs::update_bad_inputs_callback(nlohmann::json& json) {

    DEBUG("update_bad_inputs_callback(): Update to bad inputs list.");

    // Get the next output frame
    uint8_t* host_mask = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
    // "Zero" bad inputs mask - aka, fill it with ones meaning
    // nothing is currently masked
    std::memset(host_mask, 1U, input_mask_len);

    try {
        // Rely on getting inputs in cylinder order
        // TODO: is there any easy way to check this?
        bad_inputs_cylinder = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list {:s}", e.what());
        return false;
    }

    // Add current bad input mask
    for (int element : bad_inputs_cylinder) {
        if (element < input_mask_len && element >= 0) {
            host_mask[element] = 0;
        } else {
            ERROR("Got a bad input with invalid index");
            return false;
        }
    }

    // Create new metadata
    out_buf->allocate_new_metadata_object(frame_id);

    // Set no. of bad inputs in the metadata
    get_chord_metadata(out_buf, frame_id)->set_rfi_num_bad_inputs(bad_inputs_cylinder.size());

    out_buf->mark_frame_full(unique_name, frame_id);

    DEBUG("update_bad_inputs_callback(): Bad inputs buffered.");

    // Increment frame ID
    frame_id++;

    return true;
}

void bufferBadInputs::main_thread() {
    // Listen for bad input list updates
    std::string badInputs = config.get<std::string>(unique_name, "updatable_config/bad_inputs");
    configUpdater::instance().subscribe(
        badInputs,
        std::bind(&bufferBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));
}
