#pragma once
#ifndef _CUDA_RADIX_SCATTER_CUH_
#define _CUDA_RADIX_SCATTER_CUH_
#include <common.hpp>

namespace sort {
          namespace gpu {
                    namespace radix {
                              namespace details {

                                        namespace v1 {

                                                  template <std::size_t BinSize>
                                                  __global__ void kernel_radix_scatter_v1(
                                                            uint32_t* __restrict in, uint32_t* __restrict out, std::size_t data_length,
                                                            uint32_t* __restrict local_offset, std::size_t local_length,
                                                            uint32_t* __restrict global_base, std::size_t global_length,
                                                            std::size_t rshift) {

                                                            static_assert((BinSize % 32) == 0, "BinSize must be multiple of warpSize");
                                                            constexpr int NumWarps = int(BinSize >> 5);

                                                            const uint32_t tid = threadIdx.x; // 0..BinSize-1
                                                            const uint32_t block = blockIdx.x;
                                                            const uint32_t idx = block * blockDim.x + tid; // blockDim == BinSize

                                                            const int lane = int(tid & 31);
                                                            const int warp_id = int(tid >> 5);

                                                            // blockBase[bin] = local_offset[block][bin]
                                                            __shared__ uint32_t blockBase[BinSize];
                                                            __shared__ uint32_t bits[/*BinSize = */ BinSize][/*Warp id = */ NumWarps];

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

                                                            // the lowest bit lane in this wrap needs to write wrap_count
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
                                                                                bits[tid][w] = running; // prefix base for that warp
                                                                                running += c;
                                                                      }
                                                            }
                                                            __syncthreads();

                                                            if (!valid)
                                                                      return;

                                                            uint32_t warp_base = bits[my_bin][warp_id];
                                                            uint32_t local_rank = warp_base + warp_rank;

                                                            uint32_t pos = global_base[my_bin] + blockBase[my_bin] + local_rank;

                                                            if (pos >= data_length)
                                                                      asm("trap;");
                                                            // Safety: pos should be < data_length if bases/offsets are correct.
                                                            out[pos] = value;
                                                  }

                                        } // namespace v1

                                        namespace v2 {

                                                  // kernel_radix_scatter_solve_bank_conflict_v2
                                                  template <std::size_t BinSize>
                                                  __global__ void kernel_radix_scatter_v2(
                                                            uint32_t* __restrict in, uint32_t* __restrict out, std::size_t data_length,
                                                            uint32_t* __restrict local_offset, std::size_t local_length,
                                                            uint32_t* __restrict global_base, std::size_t global_length,
                                                            std::size_t rshift) {

                                                            static_assert((BinSize % 32) == 0, "BinSize must be multiple of warpSize");
                                                            constexpr uint32_t NumWarps = BinSize >> 5;

                                                            const uint32_t tid = threadIdx.x; // 0..BinSize-1
                                                            const uint32_t block = blockIdx.x;
                                                            const uint32_t idx = block * blockDim.x + tid; // blockDim == BinSize

                                                            const uint32_t lane = (tid & 31);
                                                            const uint32_t warp_id = (tid >> 5);

                                                            // blockBase[bin] = local_offset[block][bin]
                                                            __shared__ uint32_t blockBase[BinSize];

                                                            //fix bank conflict
                                                            __shared__ uint32_t bits[/*Warp id = */ NumWarps][/*BinSize = */ BinSize + 1];

                                                            uint32_t v = 0;
                                                            if (tid < BinSize && idx < local_length)
                                                                      v = local_offset[idx];

                                                            if (tid < BinSize) {
                                                                      blockBase[tid] = v;

#pragma unroll 4
                                                                      for (int w = 0; w < NumWarps; ++w)
                                                                                bits[w][tid] = 0;
                                                            }

                                                            __syncthreads();

                                                            /*Must Not Contain Early return at this stage!*/
                                                            bool valid = (idx < data_length);
                                                            unsigned active = __activemask();
                                                            uint32_t value = valid ? in[idx] : 0;
                                                            uint32_t mask = __ballot_sync(active, valid);

                                                            uint32_t my_bin = valid ? ((value >> rshift) & (BinSize - 1)) : 0xffffffffu;
                                                            uint32_t group = __match_any_sync(active, my_bin);
                                                            group &= mask;

                                                            uint32_t warp_rank = __popc(group & ((1u << lane) - 1u));
                                                            uint32_t warp_count = __popc(group);

                                                            // the lowest bit lane in this wrap needs to write wrap_count
                                                            int leader = __ffs((int)group) - 1;
                                                            if (valid && (lane == leader)) {
                                                                      bits[warp_id][my_bin] = warp_count;
                                                            }
                                                            __syncthreads();

                                                            if (tid < BinSize) {
                                                                      uint32_t running = 0;
#pragma unroll 4
                                                                      for (int w = 0; w < NumWarps; ++w) {
                                                                                uint32_t c = bits[w][tid];
                                                                                bits[w][tid] = running; // prefix base for that warp
                                                                                running += c;
                                                                      }
                                                            }
                                                            __syncthreads();

                                                            if (!valid)
                                                                      return;

                                                            uint32_t warp_base = bits[warp_id][my_bin];
                                                            uint32_t local_rank = warp_base + warp_rank;

                                                            uint32_t pos = global_base[my_bin] + blockBase[my_bin] + local_rank;

                                                            if (pos >= data_length)
                                                                      asm("trap;");
                                                            // Safety: pos should be < data_length if bases/offsets are correct.
                                                            out[pos] = value;
                                                  }

                                        } // namespace v2

                                        namespace v3 {

                                        } // namespace v3

                              } // namespace details
                    } // namespace radix
          } // namespace gpu
} // namespace sort

#endif