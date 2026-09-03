#include "cudaDirectBeamTracker.hpp"
#include "cudaUtils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kotekan {

std::vector<float2> generate_sky_grid_directions(float step) {
    std::vector<float2> grid;
    if (step <= 0.0f) step = 0.02f;

    const int n_steps = static_cast<int>(std::ceil(2.0f / step));
    for (int i = 0; i <= n_steps; ++i) {
        const float l = -1.0f + i * step;
        for (int j = 0; j <= n_steps; ++j) {
            const float m = -1.0f + j * step;
            const float r2 = l * l + m * m;
            if (r2 <= 1.0f) {
                grid.push_back(make_float2(l, m));
            }
        }
    }
    return grid;
}

int lookup_nearest_sky_grid(float l, float m, const std::vector<float2>& grid_directions) {
    if (grid_directions.empty()) return -1;

    float min_dist_sq = 1e9f;
    int best_idx = 0;

    for (std::size_t i = 0; i < grid_directions.size(); ++i) {
        const float dl = grid_directions[i].x - l;
        const float dm = grid_directions[i].y - m;
        const float dist_sq = dl * dl + dm * dm;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

namespace {

__device__ __forceinline__ float2 unpack_int4_streaming(const std::uint8_t* ptr) {
#if defined(__CUDA_ARCH__)
    std::uint32_t byte_val;
    // Cache streaming modifier (ld.global.cs):
    // Instructs Blackwell / Ada cache controller to evict streaming data first,
    // preventing the 3.3 GB input frame from evicting weights and sky grid from L2 cache.
    asm("ld.global.cs.u8 %0, [%1];" : "=r"(byte_val) : "l"(ptr));
    int r, i;
    asm("bfe.s32 %0, %1, 0, 4;" : "=r"(r) : "r"(byte_val));
    asm("bfe.s32 %0, %1, 4, 4;" : "=r"(i) : "r"(byte_val));
    return make_float2(__int2float_rn(r), __int2float_rn(i));
#else
    const std::uint32_t byte_val = *ptr;
    const int r = (static_cast<int>(byte_val) << 28) >> 28;
    const int i = ((static_cast<int>(byte_val) << 24) >> 28);
    return make_float2(static_cast<float>(r), static_cast<float>(i));
#endif
}

__global__ void generate_steering_weights_kernel(
    float2* __restrict__ weights,
    const DirectDirection3D* __restrict__ directions,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const std::uint8_t* __restrict__ antenna_mask,
    const float2* __restrict__ calibration_gains,
    const std::size_t num_beams,
    const std::size_t n_freq,
    const std::size_t n_ant) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_beams * n_freq * n_ant;
    if (idx >= total) return;

    const std::size_t ant = idx % n_ant;
    const std::size_t rest = idx / n_ant;
    const std::size_t freq = rest % n_freq;
    const std::size_t beam = rest / n_freq;

    if (antenna_mask != nullptr && __ldg(&antenna_mask[ant]) == 0) {
        weights[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    const DirectDirection3D dir = directions[beam];
    const float3 pos = antenna_positions[ant];
    const double wave_num = __ldg(&wavenumbers[freq]);

    const double delay_m = static_cast<double>(pos.x) * dir.x +
                           static_cast<double>(pos.y) * dir.y +
                           static_cast<double>(pos.z) * dir.z;
    const float phase = static_cast<float>(wave_num * delay_m);
    float s, c;
    __sincosf(phase, &s, &c);

    float wr = c;
    float wi = s;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_ant + ant]);
        // w_cal = w * g
        const float cal_r = wr * g.x - wi * g.y;
        const float cal_i = wr * g.y + wi * g.x;
        wr = cal_r;
        wi = cal_i;
    }

    weights[idx] = make_float2(wr, wi);
}

__global__ void precompute_sky_grid_kernel(
    float2* __restrict__ grid_weights,
    const float2* __restrict__ grid_lms,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const std::uint8_t* __restrict__ antenna_mask,
    const float2* __restrict__ calibration_gains,
    const std::size_t num_grid_points,
    const std::size_t n_freq,
    const std::size_t n_ant) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_grid_points * n_freq * n_ant;
    if (idx >= total) return;

    const std::size_t ant = idx % n_ant;
    const std::size_t rest = idx / n_ant;
    const std::size_t freq = rest % n_freq;
    const std::size_t grid_pt = rest / n_freq;

    if (antenna_mask != nullptr && __ldg(&antenna_mask[ant]) == 0) {
        grid_weights[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    const float2 lm = __ldg(&grid_lms[grid_pt]);
    const float l = lm.x;
    const float m = lm.y;
    const float trans_sq = l * l + m * m;
    const float n = (trans_sq <= 1.0f) ? sqrtf(1.0f - trans_sq) : 0.0f;

    const float3 pos = antenna_positions[ant];
    const double wave_num = __ldg(&wavenumbers[freq]);

    const double delay_m = static_cast<double>(pos.x) * l +
                           static_cast<double>(pos.y) * m +
                           static_cast<double>(pos.z) * n;
    const float phase = static_cast<float>(wave_num * delay_m);
    float s, c;
    __sincosf(phase, &s, &c);

    float wr = c;
    float wi = s;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_ant + ant]);
        const float cal_r = wr * g.x - wi * g.y;
        const float cal_i = wr * g.y + wi * g.x;
        wr = cal_r;
        wi = cal_i;
    }

    grid_weights[idx] = make_float2(wr, wi);
}

template <int N_ANT, int B_TILE, int TIME_UNROLL>
__global__ void __launch_bounds__(256, 4)
direct_beamformer_fused_multibeam_kernel(
    float2* __restrict__ voltages,
    const float2* __restrict__ weights,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t time_chunk_size,
    const std::size_t num_active_beams,
    const std::size_t max_beams_stride,
    const std::size_t total_warps) {

    constexpr unsigned int ANT_PER_LANE = static_cast<unsigned int>(N_ANT / 32);
    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;

    if (warp_id >= total_warps) {
        return;
    }

    const std::size_t num_beam_tiles = (num_active_beams + B_TILE - 1) / B_TILE;
    const std::size_t tile_idx = warp_id % num_beam_tiles;
    const std::size_t rest = warp_id / num_beam_tiles;
    const std::size_t freq = rest % n_freq;
    const std::size_t chunk_idx = rest / n_freq;

    const std::size_t b_base = tile_idx * B_TILE;
    const unsigned int active_in_tile = (b_base + B_TILE <= num_active_beams)
                                            ? B_TILE
                                            : static_cast<unsigned int>(num_active_beams - b_base);

    // 1. Load precomputed weights for all B_TILE beams into registers for this warp
    float w_r[B_TILE][ANT_PER_LANE];
    float w_i[B_TILE][ANT_PER_LANE];
    float nw_i[B_TILE][ANT_PER_LANE];

    #pragma unroll
    for (int b = 0; b < B_TILE; ++b) {
        const std::size_t b_curr = b_base + b;
        const std::size_t w_base = (b_curr < num_active_beams)
                                       ? ((b_curr * n_freq + freq) * N_ANT)
                                       : 0;
        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            if (b < active_in_tile) {
                const unsigned int elem = lane + a * 32U;
                const float2 w = __ldg(&weights[w_base + elem]);
                w_r[b][a] = w.x;
                w_i[b][a] = w.y;
                nw_i[b][a] = -w.y;
            } else {
                w_r[b][a] = 0.0F;
                w_i[b][a] = 0.0F;
                nw_i[b][a] = 0.0F;
            }
        }
    }

    const std::size_t t_start = chunk_idx * time_chunk_size;
    if (t_start >= n_time) {
        return;
    }
    const std::size_t t_end = (t_start + time_chunk_size < n_time)
                                  ? (t_start + time_chunk_size)
                                  : n_time;

    const std::size_t t_stride = n_freq * N_ANT;
    const std::size_t voltage_stride = n_freq * max_beams_stride;

    const std::uint8_t* packed_ptr = packed + (t_start * n_freq + freq) * N_ANT + lane;
    float2* voltages_ptr = voltages + (t_start * n_freq + freq) * max_beams_stride + b_base;

    std::size_t t = t_start;

    for (; t + (TIME_UNROLL - 1) < t_end; t += TIME_UNROLL) {
        float s_r[B_TILE][TIME_UNROLL] = {0.0F};
        float s_i[B_TILE][TIME_UNROLL] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const unsigned int a_offset = a * 32U;

            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                // LOAD RAW VOLTAGE SAMPLE ONCE WITH NON-TEMPORAL CACHE STREAMING (ld.global.cs)
                const float2 p = unpack_int4_streaming(&packed_ptr[k * t_stride + a_offset]);

                // MULTIPLY-ACCUMULATE ACROSS ALL B_TILE BEAMS SIMULTANEOUSLY
                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    const float wra = w_r[b][a];
                    const float wia = w_i[b][a];
                    const float nwi = nw_i[b][a];

                    s_r[b][k] = fmaf(wra, p.x, fmaf(nwi, p.y, s_r[b][k]));
                    s_i[b][k] = fmaf(wra, p.y, fmaf(wia, p.x, s_i[b][k]));
                }
            }
        }

        // Intra-warp shuffle reduction across all B_TILE beams
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int b = 0; b < B_TILE; ++b) {
                #pragma unroll
                for (int k = 0; k < TIME_UNROLL; ++k) {
                    s_r[b][k] += __shfl_down_sync(full_mask, s_r[b][k], offset);
                    s_i[b][k] += __shfl_down_sync(full_mask, s_i[b][k], offset);
                }
            }
        }

        // Coalesced write across lanes 0..B_TILE-1 to contiguous beam output slots
        if (lane < B_TILE && lane < active_in_tile) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                voltages_ptr[k * voltage_stride + lane] = make_float2(s_r[lane][k], s_i[lane][k]);
            }
        }

        packed_ptr += TIME_UNROLL * t_stride;
        voltages_ptr += TIME_UNROLL * voltage_stride;
    }

    // Remainder loop
    for (; t < t_end; ++t) {
        float s_r[B_TILE] = {0.0F};
        float s_i[B_TILE] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float2 p = unpack_int4_streaming(&packed_ptr[a * 32U]);

            #pragma unroll
            for (int b = 0; b < B_TILE; ++b) {
                s_r[b] = fmaf(w_r[b][a], p.x, fmaf(nw_i[b][a], p.y, s_r[b]));
                s_i[b] = fmaf(w_r[b][a], p.y, fmaf(w_i[b][a], p.x, s_i[b]));
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int b = 0; b < B_TILE; ++b) {
                s_r[b] += __shfl_down_sync(full_mask, s_r[b], offset);
                s_i[b] += __shfl_down_sync(full_mask, s_i[b], offset);
            }
        }

        if (lane < B_TILE && lane < active_in_tile) {
            voltages_ptr[lane] = make_float2(s_r[lane], s_i[lane]);
        }

        packed_ptr += t_stride;
        voltages_ptr += voltage_stride;
    }
}

} // namespace

void set_l2_persisting_weights_policy(cudaStream_t stream, const void* ptr, std::size_t bytes) {
#if CUDART_VERSION >= 11000
    if (ptr == nullptr || bytes == 0) return;

    int dev_id = 0;
    if (cudaGetDevice(&dev_id) != cudaSuccess) return;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev_id) != cudaSuccess) return;

    if (prop.persistingL2CacheMaxSize > 0) {
        const std::size_t window_size = std::min(bytes, static_cast<std::size_t>(prop.persistingL2CacheMaxSize));
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, window_size);

        cudaStreamAttrValue attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
        attr.accessPolicyWindow.num_bytes = window_size;
        attr.accessPolicyWindow.hitRatio = 1.0F;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
#else
    (void)stream; (void)ptr; (void)bytes;
#endif
}

void launch_direct_beamformer(
    const int4x2_t* d_packed,
    const float2* d_weights,
    float2* d_voltages,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    std::size_t time_chunk_size,
    std::size_t time_unroll,
    std::size_t beam_tile_size,
    cudaStream_t stream) {

    if (num_active_beams == 0) {
        return; // Zero active beams: instantaneous bypass
    }

    const std::size_t active_beams = std::min(num_active_beams, MAX_DIRECT_BEAMS);
    const std::size_t stride_beams = std::max(max_beams_stride, active_beams);

    // Auto-select optimal B_TILE (4, 2, or 1) based on active beam count
    std::size_t b_tile = 1;
    if (beam_tile_size >= 4 && active_beams >= 4) {
        b_tile = 4;
    } else if (beam_tile_size >= 2 && active_beams >= 2) {
        b_tile = 2;
    }

    const std::size_t num_chunks = (n_time + time_chunk_size - 1) / time_chunk_size;
    const std::size_t num_beam_tiles = (active_beams + b_tile - 1) / b_tile;
    const std::size_t total_warps = num_chunks * n_freq * num_beam_tiles;

    // 256 threads per block (8 warps) for Blackwell SM sub-core latency hiding
    constexpr int WARPS_PER_BLOCK = 8;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(d_packed);

    auto dispatch_kernel_combo = [&](auto ant_tag, auto b_tile_tag) {
        constexpr int N_A = decltype(ant_tag)::value;
        constexpr int B_T = decltype(b_tile_tag)::value;

        if constexpr (B_T == 4) {
            direct_beamformer_fused_multibeam_kernel<N_A, 4, 4><<<grid_dim, block_dim, 0, stream>>>(
                d_voltages, d_weights, packed_bytes, n_time, n_freq,
                time_chunk_size, active_beams, stride_beams, total_warps);
        } else if constexpr (B_T == 2) {
            if (time_unroll >= 8) {
                direct_beamformer_fused_multibeam_kernel<N_A, 2, 8><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            } else {
                direct_beamformer_fused_multibeam_kernel<N_A, 2, 4><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            }
        } else {
            if (time_unroll >= 8) {
                direct_beamformer_fused_multibeam_kernel<N_A, 1, 8><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            } else {
                direct_beamformer_fused_multibeam_kernel<N_A, 1, 4><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            }
        }
    };

    auto dispatch_b_tile = [&](auto ant_tag) {
        if (b_tile == 4) dispatch_kernel_combo(ant_tag, std::integral_constant<int, 4>{});
        else if (b_tile == 2) dispatch_kernel_combo(ant_tag, std::integral_constant<int, 2>{});
        else dispatch_kernel_combo(ant_tag, std::integral_constant<int, 1>{});
    };

    switch (n_ant) {
        case 32:  dispatch_b_tile(std::integral_constant<int, 32>{}); break;
        case 64:  dispatch_b_tile(std::integral_constant<int, 64>{}); break;
        case 128: dispatch_b_tile(std::integral_constant<int, 128>{}); break;
        case 256: dispatch_b_tile(std::integral_constant<int, 256>{}); break;
        default:
            throw std::invalid_argument("Unsupported n_ant for Direct Beamformer: must be 32, 64, 128, or 256");
    }
}

void launch_generate_steering_weights(
    float2* d_weights,
    const DirectDirection3D* d_directions,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_ant,
    cudaStream_t stream) {

    const std::size_t total_weights = num_beams * n_freq * n_ant;
    if (total_weights == 0) return;

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size = static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    generate_steering_weights_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_weights,
        d_directions,
        d_wavenumbers,
        d_antenna_positions,
        d_antenna_mask,
        d_calibration_gains,
        num_beams,
        n_freq,
        n_ant);
}

void launch_precompute_sky_grid(
    float2* d_grid_weights,
    const float2* d_grid_lms,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_grid_points,
    std::size_t n_freq,
    std::size_t n_ant,
    cudaStream_t stream) {

    const std::size_t total_weights = num_grid_points * n_freq * n_ant;
    if (total_weights == 0) return;

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size = static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    precompute_sky_grid_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_grid_weights,
        d_grid_lms,
        d_wavenumbers,
        d_antenna_positions,
        d_antenna_mask,
        d_calibration_gains,
        num_grid_points,
        n_freq,
        n_ant);
}

} // namespace kotekan
