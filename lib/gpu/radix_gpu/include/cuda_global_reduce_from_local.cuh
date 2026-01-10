#pragma once
#ifndef _CUDA_GLOBAL_REDUCE_FROM_LOCAL_CUH_
#define _CUDA_GLOBAL_REDUCE_FROM_LOCAL_CUH_
#include <common.hpp>

namespace sort {
namespace gpu {
namespace radix {
namespace details {

namespace v1 {

template <std::size_t BinSize>
__global__ void
kernel_global_reduce_from_local_v1(uint32_t *__restrict local,
                                   std::size_t local_length,
                                   uint32_t *__restrict global) {

  if (threadIdx.x < BinSize) {
    uint32_t value = local[blockIdx.x * BinSize + threadIdx.x];
    atomicAdd(&global[threadIdx.x], value);
  }
}

} // namespace v1

namespace v2 {

template <std::size_t BinSize>
__global__ void kernel_global_reduce_from_local_v2(uint32_t *__restrict local,
                                                   std::size_t local_length,
                                                   uint32_t *__restrict global,
                                                   std::size_t numBlocks) {

  uint32_t bin = blockIdx.x;   // 0..BinSize-1
  uint32_t lane = threadIdx.x; // 0..31

  // localT row start for this bin
  const uint32_t *row = local + bin * numBlocks;

  // warp-stride loop ( each lane accumulates a strided chunk of blocks )
  uint32_t sum = 0;
  for (int b = lane; b < numBlocks; b += 32) {
    sum += row[b];
  }

#pragma unroll 4
  // warp reduce (all lanes are active)
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }

  if (lane == 0) {
    global[bin] = sum;
  }
}
} // namespace v2

namespace v3 {

template <std::size_t BlockSize = 32>
__global__ void
kernel_global_reduce_from_local_v3(const uint32_t *__restrict in,
                                   uint32_t *__restrict out,
                                   std::size_t in_stride,  // pitch
                                   std::size_t in_real,    //
                                   std::size_t out_stride, // pitch
                                   std::size_t out_real    //
) {

  /*
  global memory (1D):

  block k:
  ------------------------------------------------
  | warp 0 lane 0..31  |  k*1024 + 0..31
  | warp 1 lane 0..31  |  k*1024 + 32..63
  | warp 2 lane 0..31  |  k*1024 + 64..95
  | ...
  | warp 31 lane 0..31 |  k*1024 + 992..1023
  ------------------------------------------------
  */

  __shared__ volatile uint32_t warp_sum[BlockSize];

  uint32_t lane = threadIdx.x;    // 0..31
  uint32_t warp_id = threadIdx.y; // 0..31
  uint32_t block_off = blockIdx.x;
  uint32_t bin = blockIdx.y;

  if (block_off >= out_real)
    return;

  // ranging from 0-1023
  uint32_t in_index =
      block_off * (BlockSize * BlockSize) + warp_id * BlockSize + lane;
  uint32_t x = (in_index < in_real) ? in[bin * in_stride + in_index] : 0;

  uint32_t sum = x;

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, off);

  if (lane == 0)
    warp_sum[warp_id] = sum;

  __syncthreads();

  if (warp_id == 0) {
    uint32_t v = (lane < 32) ? warp_sum[lane] : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);

    if (lane == 0) {
      if (out_real == 1)
        out[bin] = v;
      else
        out[bin * out_stride + block_off] = v;
    }
  }
}

} // namespace v3

} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
