#include <algorithm>
#include <cpu.hpp>
#include <cudaAllocator.hpp>
#include <gpu.cuh>
#include <gtest/gtest.h>
#include <radix_sort_gpu.h>
#include <vector>

class RadixLocalOffsetTest : public ::testing::TestWithParam<size_t> {};

TEST_P(RadixLocalOffsetTest, GlobalLocalOffsetCPU) {

  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  constexpr size_t BlockSize = 32;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  // input
  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  // ---------- CPU reference ----------
  std::vector<uint32_t> cpu_local;
  cpu_local_count_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      numBlocks, cpu_local);
  cpu_local_offset_ref(cpu_local, numBlocks);

  // ---------- GPU ----------
  std::vector<uint32_t> gpu_local;
  gpu_local_offset_func<BlockSize>(gpu_array.data(), N, gpu_local);

  // ---------- compare ----------
  ASSERT_EQ(cpu_local.size(), gpu_local.size());
  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local[b * BinSize + bin], gpu_local[b * BinSize + bin])
          << "Mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }
}

TEST_P(RadixLocalOffsetTest, LocalOffsetIgnoresPadding) {

  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  constexpr size_t BlockSize = 32;

  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);
  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  // CPU reference: only original N
  std::vector<uint32_t> cpu_local;
  cpu_local_count_ref(
      std::vector<uint32_t>(gpu_array.begin(), gpu_array.begin() + N),
      numBlocks, cpu_local);
  cpu_local_offset_ref(cpu_local, numBlocks);

  // GPU
  std::vector<uint32_t> gpu_local;
  gpu_local_offset_func<BlockSize>(gpu_array.data(), N, gpu_local);

  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local[b * BinSize + bin], gpu_local[b * BinSize + bin])
          << "Padding mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RadixEdgeCases, RadixLocalOffsetTest,
                         ::testing::Values(1, 111, 256, 297, 500, 512, 3987,
                                           1024 * 256 + 57));
