#include "parseReorderDefault.hpp"

#include "DataType.hpp"        // for DataType, KOTEKAN_FLOAT16, float16_t
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, CHORD_META_MAX_FREQ
#include "kotekanLogging.hpp"  // for INFO, DEBUG, ERROR

#include <algorithm> // for std::copy
#include <assert.h>  // for assert
#include <stdint.h>  // for int8_t, uint32_t, uint8_t, int16_t, int32_t, uint64_t
#include <string>    // for std::string
#include <strings.h> // for bzero
#include <unistd.h>  // for sleep
#include <vector>    // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(parseReorderDefault);

parseReorderDefault::parseReorderDefault(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&parseReorderDefault::main_thread, this)),
    _out_buf(get_buffer("out_buf")), _name(config.get<std::string>(unique_name, "name")),
    _input_reorder(std::get<0>(parse_reorder_default(config, unique_name))) {

    _out_buf->register_producer(unique_name);

    if (_out_buf->frame_size != _input_reorder.size() * sizeof(_input_reorder[0])) {
        throw std::invalid_argument("parseReorderDefault: incorrect frame size");
    }
}


void parseReorderDefault::main_thread() {
    int abs_frame_id = 0;

    bool first_time = true;
    while (!stop_thread) {

        if (first_time && abs_frame_id > 0) {
            sleep(1);
            continue;
        }

        const int frame_id = abs_frame_id % _out_buf->num_frames;
        int32_t* const frame =
            reinterpret_cast<int32_t*>(_out_buf->wait_for_empty_frame(unique_name, frame_id));
        if (frame == nullptr)
            break;

        std::copy(_input_reorder.begin(), _input_reorder.end(), frame);

        _out_buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(_out_buf, frame_id);

        chordmeta->set_frame_counter(abs_frame_id);

        // TODO: this is not quite correct. Really if going to cylinder order
        // the array is {4,2,256} {"C", "P", "D"}
        _out_buf->allocate_ndarray_frame_desc(kotekan::int32, _name, {2, 1024}, {"P", "D"});
        chordmeta->set_from_frame_desc(_out_buf->get_ndarray_frame_desc());
        chordmeta->check_frame_desc(_out_buf->get_ndarray_frame_desc());

        _out_buf->mark_frame_full(unique_name, frame_id);

        abs_frame_id += 1;

        first_time = false;
    }
}
