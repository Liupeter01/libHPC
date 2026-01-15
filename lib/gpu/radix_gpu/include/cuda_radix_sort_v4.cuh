#pragma once
#ifndef _CUDA_RADIX_SORT_V4_CUH_
#define _CUDA_RADIX_SORT_V4_CUH_
#include <common.hpp>

namespace sort {
namespace gpu {
namespace radix {
namespace details {
namespace v4 {
void __radix_sort_v4(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input);
} // namespace v3
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
