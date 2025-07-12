#include <gtest/gtest.h>
#include <thread>
#include <thread_pool.hpp>

TEST(ThreadPoolTest, CommitWithoutRef) {
  int m = -1;

  auto fut = ThreadPool::Instance()->commit([](int &m) { m = 1000; }, m);
  fut.get();

  EXPECT_EQ(m, -1);
}
