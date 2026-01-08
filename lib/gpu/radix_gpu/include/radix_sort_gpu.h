#pragma once
#ifndef _RADIX_SORT_GPU_H_
#define _RADIX_SORT_GPU_H_
#include <algorithm>
#include <cudaAllocator.hpp>
#include <cuda_radix_sort.cuh>
#include <random>

namespace sort {
namespace gpu {
namespace radix {
namespace details {
namespace helper {

void generate_random(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &vec,
    const std::size_t numbers);
}
} // namespace details

void radix_sort(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input);
} // namespace radix
} // namespace gpu
} // namespace sort

#endif //_RADIX_SORT_GPU_H_
