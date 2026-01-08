#include <algorithm>
#include <cpu.hpp>
#include <cudaAllocator.hpp>
#include <gpu.cuh>
#include <gtest/gtest.h>
#include <radix_sort_gpu.h>
#include <vector>

class RadixLocalCountTest : public ::testing::TestWithParam<size_t> {};

TEST_P(RadixLocalCountTest, LocalCountMatchesCPU) {
  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  // input
  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  // CPU reference
  std::vector<uint32_t> cpu_local_count;
  cpu_local_count_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      numBlocks, cpu_local_count);

  // GPU result
  std::vector<uint32_t> gpu_local_count;
  gpu_local_count_func(gpu_array.data(), N, gpu_local_count);

  ASSERT_EQ(cpu_local_count.size(), gpu_local_count.size());

  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local_count[b * BinSize + bin],
                gpu_local_count[b * BinSize + bin])
          << "Mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }
}

TEST_P(RadixLocalCountTest, LocalCountIgnoresPadding) {
  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  //  padding
  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  // CPU reference original N
  std::vector<uint32_t> cpu_local_count;
  cpu_local_count_ref(
      std::vector<uint32_t>(gpu_array.begin(), gpu_array.begin() + N),
      numBlocks, cpu_local_count);

  // GPU result
  std::vector<uint32_t> gpu_local_count;
  gpu_local_count_func(gpu_array.data(), N, gpu_local_count);

  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local_count[b * BinSize + bin],
                gpu_local_count[b * BinSize + bin])
          << "Padding mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RadixEdgeCases, RadixLocalCountTest,
                         ::testing::Values(1, 111, 256, 297, 500, 512));
