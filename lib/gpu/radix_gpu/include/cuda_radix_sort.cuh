#pragma once
#ifndef _CUDA_TUT_STALL_LG_HPP_
#define _CUDA_TUT_STALL_LG_HPP_
#include <vector>
#include <cudaAllocator.hpp>
#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cudaHelper.cuh>
#include <cudaAllocator.hpp>
#include <nvtx3/nvToolsExt.h>
#include <cudaUtils.cuh>

namespace sort {
          namespace gpu {
                    namespace radix {
                              namespace details {

                                        __global__ void kernel_startup();

                                        static constexpr inline std::intptr_t constexpr_log2(std::intptr_t n) {
                                                  return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1);
                                        }

                                        namespace v1 {
                                                  template< std::size_t BinSize>
                                                  __global__ void kernel_local_histogram_v1(uint32_t* __restrict global_data,
                                                            std::size_t data_length,
                                                            uint32_t* __restrict local,
                                                            std::size_t local_length,
                                                            std::size_t rshift) {

                                                            const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                                                            __shared__ uint32_t cache[BinSize];

                                                            //init shared histogram
                                                            if (threadIdx.x < BinSize)
                                                                      cache[threadIdx.x] = 0;

                                                            __syncthreads();

                                                            // each thread handles one element
                                                            if (idx < data_length) {
                                                                      uint32_t value = global_data[idx];
                                                                      uint32_t offset = (value >> rshift) & (BinSize - 1);
                                                                      atomicAdd(&cache[offset], 1);
                                                            }

                                                            __syncthreads();

                                                            // write block-local histogram to global
                                                            const uint32_t out = blockIdx.x * BinSize + threadIdx.x;
                                                            if (threadIdx.x < BinSize)
                                                                      local[out] = cache[threadIdx.x];
                                                  }

                                                  template< std::size_t BinSize >
                                                  __global__ void kernel_global_reduce_from_local_v1(uint32_t* __restrict local,
                                                            std::size_t local_length,
                                                            uint32_t* __restrict global) {

                                                            if (threadIdx.x < BinSize) {
                                                                      uint32_t value = local[blockIdx.x * BinSize + threadIdx.x];
                                                                      atomicAdd(&global[threadIdx.x], value);
                                                            }
                                                  }

                                                  template< std::size_t BinSize>
                                                  __global__ void  kernel_radix_scatter_v1(uint32_t* __restrict in,
                                                            uint32_t* __restrict out,
                                                            std::size_t data_length,
                                                            uint32_t* __restrict local_offset,
                                                            std::size_t local_length,
                                                            uint32_t* __restrict global_base,
                                                            std::size_t global_length,
                                                            std::size_t rshift) {

                                                            static_assert((BinSize % 32) == 0, "BinSize must be multiple of warpSize");
                                                            constexpr int NumWarps = int(BinSize >> 5);

                                                            const uint32_t tid = threadIdx.x;      // 0..BinSize-1
                                                            const uint32_t block = blockIdx.x;
                                                            const uint32_t idx = block * blockDim.x + tid; // blockDim == BinSize

                                                            const int lane = int(tid & 31);
                                                            const int warp_id = int(tid >> 5);

                                                            // blockBase[bin] = local_offset[block][bin]
                                                            __shared__ uint32_t blockBase[BinSize];                            
                                                            __shared__ uint32_t bits [/*BinSize = */BinSize]  [/*Warp id = */ NumWarps];

                                                            uint32_t v = 0;
                                                            if (tid < BinSize && idx < local_length)
                                                                      v = local_offset[idx];

                                                            if (tid < BinSize) {
                                                                      blockBase[tid] = v;

#pragma unroll
                                                                      for (int w = 0; w < NumWarps; ++w) 
                                                                                bits[tid][w] = 0;
                                                            }

                                                            __syncthreads();

                                                            /*Must Not Contain Early return at this stage!*/
                                                            bool valid = (idx < data_length);
                                                            uint32_t value = valid ? in[idx] : 0;
                                                            uint32_t mask = __ballot_sync(0xffffffffu, valid);

                                                            uint32_t my_bin = valid ? ((value >> rshift) & (BinSize - 1)) : 0xffffffffu;
                                                            uint32_t group = __match_any_sync(0xffffffffu, my_bin);
                                                            group &= mask;

                                                            uint32_t warp_rank = __popc(group & ((1u << lane) - 1u));
                                                            uint32_t warp_count = __popc(group);

                                                            //the lowest bit lane in this wrap needs to write wrap_count
                                                            int leader = __ffs((int)group) - 1;
                                                            if (valid && (lane == leader)) {
                                                                      bits[my_bin][warp_id] = warp_count;
                                                            }
                                                            __syncthreads();

                                                            if (tid < BinSize) {
                                                                      uint32_t running = 0;
#pragma unroll
                                                                      for (int w = 0; w < NumWarps; ++w) {
                                                                                uint32_t c = bits[tid][w];
                                                                                bits[tid][w] = running;   // prefix base for that warp
                                                                                running += c;
                                                                      }
                                                            }
                                                            __syncthreads();

                                                            if (!valid) return;

                                                            uint32_t warp_base = bits[my_bin][warp_id];
                                                            uint32_t local_rank = warp_base + warp_rank;

                                                            uint32_t pos = global_base[my_bin] + blockBase[my_bin] + local_rank;

                                                            if (pos >= data_length) asm("trap;");
                                                            // Safety: pos should be < data_length if bases/offsets are correct.
                                                            out[pos] = value;
                                                  }

                                                  void __radix_sort_v1(std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>& input);
                                        }

                                        void __kernel_startup();
                              }
                    }
          }
}

#endif