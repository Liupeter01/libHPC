#include <algorithm>
#include <cudaAllocator.hpp>
#include <gtest/gtest.h>
#include <radix_sort_gpu.h>
#include <vector>

TEST(RadixSortPlatformGPU, TestRadixSortGPUv3) {
          static constexpr std::size_t ARRAY_SIZE = 100000000;
          std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>
                    sort_test_array;
          ::sort::gpu::radix::details::helper::generate_random(sort_test_array,
                    ARRAY_SIZE);
          ::sort::gpu::radix::details::v3::__radix_sort_v3(sort_test_array);
          EXPECT_TRUE(std::is_sorted(sort_test_array.begin(), sort_test_array.end()));
}