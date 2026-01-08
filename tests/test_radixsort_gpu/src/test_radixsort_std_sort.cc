#include <algorithm>
#include <gtest/gtest.h>
#include <radix_sort_cpu.hpp>

TEST(RadixSortPlatformGPU, TestRadixSortSTLBaseLine) {
  static constexpr std::size_t ARRAY_SIZE = 100000000;
  std::vector<uint32_t> sort_test_array;
  sort::radix::details::helper::generate_random(sort_test_array, ARRAY_SIZE);
  std::sort(sort_test_array.data(),
            sort_test_array.data() + sort_test_array.size());
  EXPECT_TRUE(std::is_sorted(sort_test_array.begin(), sort_test_array.end()));
}
