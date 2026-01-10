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
#include <type_traits>
#include <vector>

namespace sort {
          namespace gpu {
                    namespace radix {
                              namespace details {
                                        static constexpr inline std::intptr_t constexpr_log2(std::intptr_t n) {
                                                  return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1);
                                        }
                              } // namespace details
                    } // namespace radix
          } // namespace gpu
} // namespace sort

#endif