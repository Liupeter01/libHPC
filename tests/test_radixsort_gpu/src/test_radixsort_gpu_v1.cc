#include <algorithm>
#include <gtest/gtest.h>
#include <vector>
#include <radix_sort_gpu.h>
#include <cudaAllocator.hpp>

TEST(RadixSortPlatformGPU, KernelStartupOnly) {
          ::sort::gpu::radix::details::__kernel_startup();
}

TEST(RadixSortPlatformGPU, TestRadixSortGPUv1) {
          static constexpr std::size_t ARRAY_SIZE = 100000000;
          std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> sort_test_array;
          ::sort::gpu::radix::details::helper::generate_random(sort_test_array, ARRAY_SIZE);
          ::sort::gpu::radix::radix_sort(sort_test_array);
          EXPECT_TRUE(std::is_sorted(sort_test_array.begin(), sort_test_array.end()));
}
