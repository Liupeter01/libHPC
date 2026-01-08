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
  const int x = int(blockIdx.x * BlockSize + threadIdx.x);
  const int y = int(blockIdx.y * BlockSize + threadIdx.y);

  __shared__ volatile _Ty tile[BlockSize][BlockSize + 1];

  tile[threadIdx.y][threadIdx.x] = _Ty(0);
  __syncthreads();

  if (x < int(width) && y < int(height)) {
    tile[threadIdx.y][threadIdx.x] = in[uint32_t(y) * width + uint32_t(x)];
  }
  __syncthreads();

  // after transpose
  const int tx = int(blockIdx.y * BlockSize + threadIdx.x);
  const int ty = int(blockIdx.x * BlockSize + threadIdx.y);

  if (tx < int(height) && ty < int(width)) {
    out[uint32_t(ty) * height + uint32_t(tx)] = tile[threadIdx.x][threadIdx.y];
  }
}
} // namespace util
} // namespace gpu

#endif //_CUDA_UTILS_CUH_
