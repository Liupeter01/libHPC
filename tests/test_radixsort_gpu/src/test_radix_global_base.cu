#include <algorithm>
#include <cpu.hpp>
#include <cudaAllocator.hpp>
#include <gpu.cuh>
#include <gtest/gtest.h>
#include <radix_sort_gpu.h>
#include <vector>

class RadixGlobalBaseTest : public ::testing::TestWithParam<size_t> {};

TEST_P(RadixGlobalBaseTest, GlobalBaseMatchesCPU) {
  size_t N = GetParam();

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  gpu_global_base(gpu_array.data(), N, gpu_base);

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, GlobalBaseIgnoresPadding) {
  size_t N = GetParam();
  constexpr size_t BinSize = 256;

  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  std::vector<uint32_t> cpu_base, gpu_base;

  //  original N
  cpu_global_base_ref(
      std::vector<uint32_t>(gpu_array.begin(), gpu_array.begin() + N),
      cpu_base);

  gpu_global_base(gpu_array.data(), N, gpu_base);

  for (size_t i = 0; i < 256; ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch with padding at N=" << N << " bin=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(RadixEdgeCases, RadixGlobalBaseTest,
                         ::testing::Values(1, 111, 256, 297, 500, 512));
