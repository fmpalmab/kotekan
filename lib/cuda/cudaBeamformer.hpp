#ifndef CUDA_BEAMFORMER_HPP
#define CUDA_BEAMFORMER_HPP

#include "Config.hpp"              // for Config
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface

#include <cublas_v2.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

/***************************************************
				CHARTS Constants & Types
***************************************************/

#define N_BEAMS 32
#define N_ANTENNAS 32
#define N_FREQUENCIES 672
#define HALF_FOV 3.5

#define N_POL 1				
#define N_CX 2				

#ifndef N_AVERAGING
	#define N_AVERAGING 300
#endif

#define TOT_CHANNELS 672
#define START_F 0.300
#define END_F 0.5016
#define BW_PER_CHANNEL ((END_F - START_F)/TOT_CHANNELS)

#define C_SPEED 299792458.0
#ifndef PI
	#define PI 3.14159265358979
#endif

#define N_BITS 8
#define MAX_VAL 127

/* Fixed derivation macros */
#define N_INPUTS_PER_OUTPUT (N_POL*N_AVERAGING)

typedef char2 CxInt8_t;
typedef char char4_t[4]; 
typedef char char8_t[8]; 
typedef CxInt8_t cuChar4_t[4];

class antenna {
public:
	float x = 0;
	float y = 0;
	float z = 0;
	antenna(){}
	friend std::istream & operator >> (std::istream &in, antenna &a) {
        in >> a.x >> a.y >> a.z;
        return in;
    }
};

class beam_direction {
public:
	float theta = 0;
	float phi = 0;
	beam_direction(){}
	beam_direction(float th, float ph) : theta(th), phi(ph) {}
	friend std::istream & operator >> (std::istream &in, beam_direction &a) {
        in >> a.theta >> a.phi;
        return in;
    }
};

/**
 * @class cudaBeamformer
 * @brief cudaCommand for performing beamforming via cuBLAS and Kotekan arrays.
 */
class cudaBeamformer : public cudaCommand {
public:
    cudaBeamformer(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                         int inst);
    ~cudaBeamformer();
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    int _num_elements;
    int _num_local_freq;
    int _samples_per_data_set;
    std::string _gpu_mem_voltage;
    std::string _gpu_mem_beam_output;
    std::string _antenna_positions_file;
    std::string _beam_directions_file;

    // CUBLAS pointers
    void *d_fourier_coefficients;
    void *d_inv_max_value;
    void *d_zero;

    cublasHandle_t handle;

    int fourier_coefficients_rows;
    int fourier_coefficients_cols;
    int fourier_coefficients_stride;
    int B_cols;
    int B_rows;
    int B_ld_cublas;
    int B_stride_cublas;
    int C_rows;
    int C_cols;
    int C_stride;
};

#endif // CUDA_BEAMFORMER_HPP
