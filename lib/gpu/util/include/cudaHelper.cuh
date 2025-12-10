#pragma once
#ifndef _CUDA_HELPER_HPP_
#define _CUDA_HELPER_HPP_
#include <string>
#include <stdexcept>
#include <system_error>
#include <linearUtils.cuh>

namespace cudahelper {
          inline const std::error_category & cudaErrorCategory() noexcept{
                    struct MyCudaErrorCategory : std::error_category {
                              std::string message(int val) const override { return cudaGetErrorString(static_cast<cudaError_t>(val)); }
                              const char* name() const noexcept override { return "CUDA"; }
                    };
                    static  MyCudaErrorCategory ins;
                    return ins;
          }

          inline std::error_code makeCudaError(const cudaError_t error) noexcept{
                    return { static_cast<int>(error), cudaErrorCategory() };
          }
          inline void throwCudaError(const cudaError_t e, const char* file, int line) {
                    throw std::system_error(makeCudaError(e),
                              std::string(file ? file : "[??]") + ":" + std::to_string(line));
          }

#define GLOBAL_LINEAR_TID     util::global_linear_tid()
#define GLOBAL_LINEAR_STRIDE   util::global_linear_stride()

          template<typename Func>
          __global__ void parallel_for(std::size_t size, Func func) {
                    const auto tid = GLOBAL_LINEAR_TID;
                    const auto stride = GLOBAL_LINEAR_STRIDE;
                    for (auto i = tid; i < size; i += stride) {
                              func(i);
                    }
          }

          template<std::size_t N, typename Func>
          __global__ void parallel_for(Func func) {
                    parallel_for(N, func);
          }

          // Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
          inline cudaError_t checkCuda(cudaError_t result){
                    if (result != cudaSuccess) {
                              cudahelper::throwCudaError(result, __FILE__, __LINE__);
                    }
                    return result;
          }
}

#endif //_CUDA_HELPER_HPP_