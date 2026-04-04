#ifndef DSABEAMFORMERSTAGE_HPP
#define DSABEAMFORMERSTAGE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <string>

/**
 * @class dsaBeamformerStage
 * @brief Stage for wrapping the devincody/DSAbeamformer implementation.
 *
 * @par Buffers
 * @buffer in_buf The input voltage buffer
 * @buffer out_buf The output beam buffer
 */
class dsaBeamformerStage : public kotekan::Stage {
public:
    dsaBeamformerStage(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~dsaBeamformerStage();
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
};

#endif // DSABEAMFORMERSTAGE_HPP
