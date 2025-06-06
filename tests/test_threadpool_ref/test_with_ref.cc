#include <thread>
#include <functional>
#include <gtest/gtest.h>
#include <thread_pool.hpp>

TEST(ThreadPoolTest, CommitWithRef) {
          int m = -1;

          auto fut = ThreadPool::Instance()->commit([](int& m) { m = 1000; }, std::ref(m));
          fut.get();

          EXPECT_EQ(m, 1000);
}