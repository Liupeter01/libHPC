#include <algorithm>
#include <cudaAllocator.hpp>
#include <gtest/gtest.h>
#include <radix_sort_gpu.h>
#include <vector>

TEST(RadixSortPlatformGPU, TestRadixSortGPUv4) {
  static constexpr std::size_t ARRAY_SIZE = 100000000;
  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>>
      sort_test_array;
  auto now = std::chrono::high_resolution_clock::now();
  ::sort::gpu::radix::details::helper::generate_random(sort_test_array,
                                                       ARRAY_SIZE);
  std::cout << "Fill Data Time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - now)
                   .count()
            << "ms\n";
  ::sort::gpu::radix::details::v4::__radix_sort_v4(sort_test_array);
  EXPECT_TRUE(std::is_sorted(sort_test_array.begin(), sort_test_array.end()));
}
