#include <algorithm>
#include <gtest/gtest.h>
#include <radix_sort_cpu.hpp>

TEST(RadixSortPlatformCPU, TestRadixSortCPUMutliV1) {
          static constexpr std::size_t ARRAY_SIZE = 10000000;
          std::vector<uint32_t> sort_test_array;
          sort::radix::details::helper::generate_random(sort_test_array, ARRAY_SIZE);
          sort::radix::details::radix_sort_cache_thread_v1(sort_test_array.data(), sort_test_array.size());
          EXPECT_TRUE(std::is_sorted(sort_test_array.begin(), sort_test_array.end()));
}