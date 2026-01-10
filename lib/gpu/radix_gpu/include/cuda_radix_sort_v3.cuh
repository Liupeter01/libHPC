#pragma once
#ifndef _CUDA_RADIX_SORT_V3_CUH_
#define _CUDA_RADIX_SORT_V3_CUH_
#include <common.hpp>

namespace sort {
namespace gpu {
namespace radix {
namespace details {
namespace v3 {

void __radix_sort_v3(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input);

} // namespace v3
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
