#pragma once
#ifndef _CUDA_UTILS_CUH_
#define _CUDA_UTILS_CUH_
#include <cuda.h>
#include <cuda_runtime.h>

namespace gpu {
namespace util {
template <typename _Ty, std::size_t BlockSize>
__global__ void kernel_transpose(_Ty *__restrict out, const _Ty *__restrict in,
                                 const uint32_t width, const uint32_t height) {

  // before transpose
          const uint32_t x = blockIdx.x * BlockSize + threadIdx.x;
          const uint32_t y = blockIdx.y * BlockSize + threadIdx.y;

  __shared__ volatile _Ty tile[BlockSize][BlockSize + 1];

  tile[threadIdx.y][threadIdx.x] = _Ty(0);
  __syncthreads();

  if (x <width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();

  // after transpose
  const uint32_t tx = blockIdx.y * BlockSize + threadIdx.x;
  const uint32_t ty = blockIdx.x * BlockSize + threadIdx.y;

  if (tx < height && ty < width) {
    out[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
  }
}
} // namespace util
} // namespace gpu

#endif //_CUDA_UTILS_CUH_
