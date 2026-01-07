#include <radix_sort_gpu.h>
#include <cuda_radix_sort.cuh>

namespace sort {
          namespace gpu {
                    namespace radix {

                              namespace details {
                                        namespace helper {

                                                  void generate_random(std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>& vec,
                                                            const std::size_t numbers) {
                                                            vec.clear();
                                                            vec.resize(numbers);
                                                            std::generate(vec.begin(), vec.end(),
                                                                      [uni = std::uniform_int_distribution<uint32_t>(100, std::numeric_limits<uint32_t>::max() - 100),
                                                                      rng = std::mt19937{}]() mutable { return uni(rng); });
                                                  }
                                        }
                              }

                              void radix_sort(std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>& input) {
                                        nvtxRangePushA("RadixSortGPUv1");
                                        ::sort::gpu::radix::details::v1::__radix_sort_v1(input);
                                        nvtxRangePop();
                              }
                    }
          }
}