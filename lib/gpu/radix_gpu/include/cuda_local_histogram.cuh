#pragma once
#ifndef _CUDA_LOCAL_HISTOGRAM_CUH_
#define _CUDA_LOCAL_HISTOGRAM_CUH_
#include <common.hpp>

namespace sort {
namespace gpu {
namespace radix {
namespace details {

namespace v1 {
template <std::size_t BinSize>
__global__ void
kernel_local_histogram_v1(uint32_t *__restrict global_data,
                          std::size_t data_length, uint32_t *__restrict local,
                          std::size_t local_length, std::size_t rshift) {

  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint32_t cache[BinSize];

  // init shared histogram
  if (threadIdx.x < BinSize)
    cache[threadIdx.x] = 0;

  __syncthreads();

  // each thread handles one element
  if (idx < data_length) {
    uint32_t value = global_data[idx];
    uint32_t offset = (value >> rshift) & (BinSize - 1);
    atomicAdd(&cache[offset], 1);
  }

  __syncthreads();

  // write block-local histogram to global
  const uint32_t out = blockIdx.x * BinSize + threadIdx.x;
  if (threadIdx.x < BinSize)
    local[out] = cache[threadIdx.x];
}
} // namespace v1

namespace v2 {
template <std::size_t BinSize>
__global__ void
kernel_local_histogram_v2(uint32_t *__restrict global_data,
                          std::size_t data_length, uint32_t *__restrict local,
                          std::size_t local_length, std::size_t rshift) {

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = (tid & 31);
  const uint32_t warp_id = (tid >> 5);

  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint32_t cache[BinSize];

  // init shared histogram
  if (threadIdx.x < BinSize)
    cache[threadIdx.x] = 0;

  __syncthreads();

  // each thread handles one element
  // if (idx < data_length) {
  //          uint32_t value = global_data[idx];
  //          uint32_t offset = (value >> rshift) & (BinSize - 1);
  //          atomicAdd(&cache[offset], 1);
  //}

  unsigned active = __activemask();
  bool valid = (idx < data_length);
  uint32_t value = valid ? global_data[idx] : 0;
  uint32_t mask = __ballot_sync(active, valid);

  uint32_t my_bin = valid ? ((value >> rshift) & (BinSize - 1)) : 0xffffffffu;
  uint32_t group = __match_any_sync(active, my_bin);
  group &= mask;

  uint32_t warp_count = __popc(group);

  // the lowest bit lane in this wrap needs to write wrap_count
  int leader = __ffs((int)group) - 1;
  if (valid && (lane == leader)) {
    atomicAdd(&cache[my_bin], warp_count);
  }
  __syncthreads();

  // write block-local histogram to global
  const uint32_t out = blockIdx.x * BinSize + threadIdx.x;
  if (threadIdx.x < BinSize)
    local[out] = cache[threadIdx.x];
}
} // namespace v2

namespace v3 {} // namespace v3

} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
