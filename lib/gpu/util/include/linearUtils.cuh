#pragma once
#ifndef _LINEAR_UTILS_CUH_
#define _LINEAR_UTILS_CUH_
#include <cuda.h>
#include <cuda_runtime.h>

namespace cudahelper {
namespace util {
__device__ std::size_t global_linear_stride();
__device__ std::size_t global_linear_tid();

template <typename _Ty, typename Callable>
__device__ __forceinline__ _Ty my_atomic(_Ty *address, _Ty value,
                                         Callable callable) {

  _Ty expected, old = *address;
  do {
    expected = old;
    old = atomicCAS(address, expected, callable(expected, value));
  } while (expected != old);
  return old;
}
} // namespace util
} // namespace cudahelper

#endif //_LINEAR_UTILS_CUH_
