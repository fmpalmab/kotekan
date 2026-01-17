#ifndef N2FRAME_TO_VISFRAME_HPP
#define N2FRAME_TO_VISFRAME_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"
#include "version.h"
#include "visUtil.hpp"

#include <stdint.h> // for int32_t, uint32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @brief Merges and expands CHIME lost_samples buffers (1 bytes per sample for 4
 * frequencies) into a CHORD style package loss mask (1 bit per downsampled
 * time, per 4 frequencies, per polarization, per dish).
 *
 * @par Buffers
 * @buffer pl_mask_buf Kotekan buffer for package loss mask
 *     @buffer_format [time / 2 / 64][freq / 4][polr][dish / 8][time / 2 % 64]
 *     @buffer_metadata chordMetadata
 * @buffer lost_samples_buf Array of flags which indicate if a sample in a given location is lost
 *     @buffer_format Array of flags uint8_t flags which are either 0 (unset) or 1 (set)
 *     @buffer_metadata chimeMetadata
 *
 * @conf  in_lost_sample_buffers    Buffers to hold the lost samples buffer. For
 * example: in_lost_sample_buffers:
 *                                        - lost_samples_buffer_0
 *                                        - lost_samples_buffer_1
 *                                        - lost_samples_buffer_2
 *                                        - lost_samples_buffer_3
 *
 * @author Roland Haas
 */
class n2FrameToVisFrame : public kotekan::Stage {
public:
    /// Standard constructor
    n2FrameToVisFrame(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~n2FrameToVisFrame();

    /// Main thead which zeros the data from the lost_samples_buf
    void main_thread() override;

private:
    void register_base_dataset_states(std::string& instrument_name,
                                      std::vector<std::pair<uint32_t, freq_ctype>>& freqs,
                                      std::vector<input_ctype>& inputs,
                                      std::vector<prod_ctype>& prods);

private:
    datasetManager& dm = datasetManager::instance();

    // the base states (input, prod, freq, meta)
    std::vector<state_id_t> base_dataset_states;

    /// The buffer with the output of N2Accumulate
    Buffer* n2_buf;

    /// The target buffer for a visFrame
    Buffer* vis_buf;
};

#endif /* N2FRAME_TO_VISFRAME_HPP */
