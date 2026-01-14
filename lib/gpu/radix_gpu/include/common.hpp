#pragma once
#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <cuda.h>
#include <cudaAllocator.hpp>
#include <cudaHelper.cuh>
#include <cudaUtils.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <vector>

namespace sort {
namespace gpu {
namespace radix {
namespace details {
static constexpr inline std::intptr_t constexpr_log2(std::intptr_t n) {
  return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1);
}

static inline uint32_t div_up_u32(uint32_t a, uint32_t b) {
          return (a + b - 1u) / b;
}
static inline uint32_t align_up_u32(uint32_t a, uint32_t align) {
          return ((a + align - 1u) / align) * align;
}
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort

#endif
