#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cudaAllocator.hpp>
#include <cuda_radix_sort.cuh>
#include <libmorton/morton.h>
#include <omp.h>
#include <radix_sort_gpu.h>
#include <vector>

static void BM_gpu_radix_sort_v1(benchmark::State &bm) {
  for (auto _ : bm) {

    // benchmark::DoNotOptimize(ret);
  }
}

// static void BM_stl_sort_baseline(benchmark::State& bm) {
//           for (auto _ : bm) {
//                     static constexpr std::size_t ARRAY_SIZE = 100000000;
//                     std::vector<uint32_t> sort_test_array;
//                     ::sort::gpu::radix::details::helper::generate_random(sort_test_array,
//                     ARRAY_SIZE); std::sort(sort_test_array.data(),
//                               sort_test_array.data() +
//                               sort_test_array.size());
//                     benchmark::DoNotOptimize(sort_test_array);
//           }
// }

// BENCHMARK(BM_floatingpoint);
// BENCHMARK_MAIN();

static constexpr std::size_t ARRAY_SIZE = 100000000;

void v1() {

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>
      sort_test_array;
  ::sort::gpu::radix::details::helper::generate_random(sort_test_array,
                                                       ARRAY_SIZE);
  ::sort::gpu::radix::details::v1::__radix_sort_v1(sort_test_array);
}

void v2() {

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>
      sort_test_array;
  ::sort::gpu::radix::details::helper::generate_random(sort_test_array,
                                                       ARRAY_SIZE);
  ::sort::gpu::radix::details::v2::__radix_sort_v2(sort_test_array);
}

void v3() {

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>
      sort_test_array;
  ::sort::gpu::radix::details::helper::generate_random(sort_test_array,
                                                       ARRAY_SIZE);
  ::sort::gpu::radix::details::v3::__radix_sort_v3(sort_test_array);
}

void radix_sort_test() {
  ::sort::gpu::radix::details::__kernel_startup();

  // printf("\n===============v1================\n");
  // v1();
  // printf("\n===============v2================\n");
  // v2();
  // printf("\n===============v3================\n");
  v3();
}

int main() {

  radix_sort_test();
  return 0;
}
