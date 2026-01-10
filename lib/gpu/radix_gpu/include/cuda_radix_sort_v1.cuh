#pragma once
#ifndef _CUDA_RADIX_SORT_V1_CUH_
#define _CUDA_RADIX_SORT_V1_CUH_
#include <common.hpp>

namespace sort {
namespace gpu {
namespace radix {
namespace details {
namespace v1 {

void __radix_sort_v1(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input);
} // namespace v1
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
