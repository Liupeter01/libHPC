#pragma once
#ifndef _CUDA_RADIX_SORT_CUH_
#define _CUDA_RADIX_SORT_CUH_
#include <common.hpp>
#include <cuda_radix_sort_v1.cuh>
#include <cuda_radix_sort_v2.cuh>
#include <cuda_radix_sort_v3.cuh>

namespace sort {
namespace gpu {
namespace radix {
namespace details {

__global__ void kernel_startup();
void __kernel_startup();

} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
