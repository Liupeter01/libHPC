#include <cudaHelper.cuh>

namespace cudahelper {

          __global__ void kernel_clear_u32(uint32_t* __restrict ptr, std::size_t n) {
                    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    std::size_t stride = blockDim.x * gridDim.x;

                    for (std::size_t i = idx; i < n; i += stride) {
                              ptr[i] = 0;
                    }
          }

          __device__  uint32_t warp_reduce_sum(uint32_t v) {
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                              v += __shfl_down_sync(0xffffffffu, v, off);
                    return v;
          }

          __device__ uint32_t warp_exclusive_scan(uint32_t v, int lane) {
                    return warp_scan<true>(v, lane);
          }

          __device__ uint32_t warp_inclusive_scan(uint32_t v, int lane){
                    return warp_scan<false>(v, lane);
          }

}