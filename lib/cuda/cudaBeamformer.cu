#include "cudaBeamformer.hpp"


#include "cuda.h"             
#include "gpuCommand.hpp"     
#include "kotekanLogging.hpp" 

#include <stdexcept>
#include <iostream>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaBeamformer);

__global__
void expand_input(char const * __restrict__ input, char *output, int input_size) {
	__shared__ float shmem_in[32];
	__shared__ double shmem_out[32];

	char4_t *char_shmem_in;
	cuChar4_t *char_shmem_out;

	char_shmem_in = reinterpret_cast<char4_t *>(shmem_in);
	char_shmem_out = reinterpret_cast<cuChar4_t *>(shmem_out);

	int local_idx = threadIdx.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (global_idx < input_size/sizeof(float)) {
		shmem_in[local_idx] = ((float *) input)[global_idx]; 

		for (int i = 0; i < 4; i++) {
			char temp = char_shmem_in[local_idx][i];

			char high = (temp >> 4); 
			char low = (temp << 4);  
			low = (low >> 4); 	 	 

			char_shmem_out[local_idx][i].x = high; 
			char_shmem_out[local_idx][i].y = low;
		}

		((double *) output)[global_idx] = shmem_out[local_idx];	

		global_idx += gridDim.x * blockDim.x;
	}
}

dim3 detect_dimGrid(N_OUTPUTS_PER_GEMM, N_FREQUENCIES, 1);
dim3 detect_dimBlock(N_BEAMS, 1,1);

__global__
void detect_sum(cuComplex const * __restrict__ input, int n_avg,  float * __restrict__ output) {
	const int n_beams = blockDim.x;	 	
	const int n_freq = gridDim.y;   	
	const int n_batch = gridDim.x;		

	const int batching_idx = blockIdx.x;
	const int freq_idx = blockIdx.y;	
	const int beam_idx = threadIdx.x;	

	__shared__ float shmem[N_BEAMS];
	shmem[beam_idx] = 0;			

	const int input_idx  	 = freq_idx * n_batch * n_avg * n_beams 	
								+ batching_idx * n_avg * n_beams  		
								+ beam_idx;							   	

	const int output_idx = batching_idx * n_freq * n_beams + freq_idx * n_beams + beam_idx;

	for (int i = input_idx; i < input_idx + n_avg*n_beams; i += n_beams) {
		shmem[beam_idx] += input[i].x*input[i].x + input[i].y*input[i].y;
	}

	output[output_idx] = shmem[beam_idx]; 
}

cudaBeamformer::cudaBeamformer(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers,
                                           cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state, "beamformer", "") {
    
    _num_elements = config.get_default<int>(unique_name, "num_elements", N_ANTENNAS);
    _num_local_freq = config.get_default<int>(unique_name, "num_local_freq", N_FREQUENCIES);
    _samples_per_data_set = config.get_default<int>(unique_name, "samples_per_data_set", 288);
    
    _gpu_mem_beam_output = config.get_default<std::string>(unique_name, "gpu_mem_beam_output", "beam_output");
    _antenna_positions_file = config.get<std::string>(unique_name, "antenna_positions");
    _beam_directions_file = config.get<std::string>(unique_name, "beam_positions");

    set_command_type(gpuCommandType::KERNEL);

    if (inst == 0) {
        // Initialize CuBLAS handle
        cublasCreate(&handle);

        fourier_coefficients_rows = N_BEAMS;
        fourier_coefficients_cols = _num_elements;
        fourier_coefficients_stride  = fourier_coefficients_rows * fourier_coefficients_cols;
        B_cols = _samples_per_data_set * N_POL;
        B_rows = fourier_coefficients_cols;
        B_ld_cublas = B_rows * _num_local_freq;
        B_stride_cublas = B_rows;
        
        C_rows = fourier_coefficients_rows;
        C_cols = B_cols;
        C_stride = C_rows*C_cols;

        // Allocate internal persistent arrays
        // Note: Kotekan device.get_gpu_memory is for buffer data. We use raw cudaMalloc for internal persistent arrays.
        cudaMalloc(&d_fourier_coefficients, fourier_coefficients_rows * fourier_coefficients_cols * _num_local_freq * sizeof(CxInt8_t));
        cudaMalloc(&d_inv_max_value, sizeof(cuComplex));
        cudaMalloc(&d_zero, sizeof(cuComplex));

        h_inv_max_value.x = 1.0/MAX_VAL;
        h_inv_max_value.y = 0;
        h_zero.x = 0;
        h_zero.y = 0;

        antenna *pos = new antenna[_num_elements];
        beam_direction *dir = new beam_direction[N_BEAMS];
        
        std::ifstream ant_file(_antenna_positions_file);
        if(!ant_file.is_open()) {
            throw std::runtime_error("Failed to open antenna positions file: " + _antenna_positions_file);
        }
        int num_ants_in_file;
        ant_file >> num_ants_in_file;
        for (int i=0; i<std::min(_num_elements, num_ants_in_file); i++) {
            ant_file >> pos[i];
        }
        ant_file.close();

        std::ifstream beam_file(_beam_directions_file);
        if(!beam_file.is_open()) {
            throw std::runtime_error("Failed to open beam directions file: " + _beam_directions_file);
        }
        int num_beams_in_file;
        beam_file >> num_beams_in_file;
        for (int i=0; i<std::min(N_BEAMS, num_beams_in_file); i++) {
            beam_file >> dir[i];
        }
        beam_file.close();

        CxInt8_t *fourier_coefficients = new CxInt8_t[fourier_coefficients_cols*fourier_coefficients_rows*_num_local_freq];
        float bw_per_channel = BW_PER_CHANNEL;

        for (int i = 0; i < _num_local_freq; i++){
            float freq = START_F + i * bw_per_channel;
            float wavelength = C_SPEED/(1E9*freq);
            for (int j = 0; j < _num_elements; j++){
                for (int k = 0; k < N_BEAMS; k++){
                    fourier_coefficients[i*fourier_coefficients_stride + j*N_BEAMS + k].x = round(MAX_VAL*cos(-2*PI*(pos[j].x*sin(dir[k].theta) + pos[j].y*sin(dir[k].phi))/wavelength));
                    fourier_coefficients[i*fourier_coefficients_stride + j*N_BEAMS + k].y = round(MAX_VAL*sin(-2*PI*(pos[j].x*sin(dir[k].theta) + pos[j].y*sin(dir[k].phi))/wavelength));
                }
            }
        }

        // Copy up to device
        cudaMemcpy(d_fourier_coefficients, fourier_coefficients, fourier_coefficients_rows*fourier_coefficients_cols*_num_local_freq*sizeof(CxInt8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inv_max_value, &h_inv_max_value, sizeof(cuComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zero, &h_zero, sizeof(cuComplex), cudaMemcpyHostToDevice);

        delete[] pos;
        delete[] dir;
        delete[] fourier_coefficients;
    }
}

cudaBeamformer::~cudaBeamformer() {
    cublasDestroy(handle);
    cudaFree(d_fourier_coefficients);
    cudaFree(d_inv_max_value);
    cudaFree(d_zero);
}

cudaEvent_t cudaBeamformer::execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>&) {
    pre_execute();

    size_t input_frame_len = (size_t)_num_elements * _num_local_freq * _samples_per_data_set;
    void* input_memory = device.get_gpu_memory(_gpu_mem_voltage, input_frame_len);

    // Calculate detecting float output size 
    size_t output_len = (size_t)N_BEAMS * _num_local_freq * (_samples_per_data_set / N_AVERAGING) * sizeof(float);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_beam_output, gpu_frame_id, _gpu_buffer_depth, output_len);

    // Allocate frame-specific scratch matrices B and C mapped dynamically across the processing limits
    size_t B_len = (size_t)B_rows * _num_local_freq * B_cols * sizeof(CxInt8_t);
    size_t C_len = (size_t)C_rows * _num_local_freq * C_cols * sizeof(cuComplex);
    void* d_B = device.get_gpu_memory_array(_gpu_mem_voltage + "_scratch_B", gpu_frame_id, _gpu_buffer_depth, B_len);
    void* d_C = device.get_gpu_memory_array(_gpu_mem_voltage + "_scratch_C", gpu_frame_id, _gpu_buffer_depth, C_len);

    record_start_event();
    
    cudaStream_t stream = device.getStream(cuda_stream_id);
    cublasSetStream(handle, stream);

    // 1. Expand input
    // The legacy code iterated over fixed bounds, we compute directly over the frame 
    int input_complex_samples = _num_elements * _num_local_freq * _samples_per_data_set * N_POL; 
    int blocks_needed = (input_complex_samples / 8 + 31) / 32; // input_size in loop is divided by 4-byte chunk (which is 8 complex samples)
    if (blocks_needed == 0) blocks_needed = 1;
    
    expand_input<<<blocks_needed, 32, 0, stream>>>((char const *)input_memory,
                                          (char *) d_B, 
                                          input_complex_samples * 2); // Pass bytes

    // 2. Beamforming GEMM
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                fourier_coefficients_rows, B_cols, fourier_coefficients_cols,
                                d_inv_max_value,
                                d_fourier_coefficients, CUDA_C_8I, fourier_coefficients_rows, fourier_coefficients_stride,
                                d_B, CUDA_C_8I, B_ld_cublas, B_stride_cublas,
                                d_zero,
                                d_C, CUDA_C_32F, C_rows, C_stride,
                                _num_local_freq, CUDA_C_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // 3. Detect and Sum
    dim3 detect_dimGrid((_samples_per_data_set / N_AVERAGING), _num_local_freq, 1);
    detect_sum<<<detect_dimGrid, detect_dimBlock, 0, stream>>>((cuComplex const*)d_C, N_INPUTS_PER_OUTPUT, (float*)output_memory);

    return record_end_event();
}
