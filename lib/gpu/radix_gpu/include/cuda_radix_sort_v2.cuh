#pragma once
#ifndef _CUDA_RADIX_SORT_V2_CUH_
#define _CUDA_RADIX_SORT_V2_CUH_
#include <common.hpp>

namespace sort {
namespace gpu {
namespace radix {
namespace details {
namespace v2 {

void __radix_sort_v2(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input);

} // namespace v2
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
