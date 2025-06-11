#include <gtest/gtest.h>
#include <hierarchical_lock.hpp>
#include <thread>

TEST(HierarchicalLockTest, MultiThreadedLockOrder) {
  hierarchical_lock lock1(1000);
  hierarchical_lock lock2(500);

  std::thread th1([&lock1, &lock2]() {
    EXPECT_NO_THROW(
        {
          lock1.lock();
          lock2.lock();
          lock2.unlock();
          lock1.unlock();
        },
        std::runtime_error);
  });

  std::thread th2([&lock1, &lock2]() {
    EXPECT_THROW(
        {
          lock2.lock();
          lock1.lock();
          lock1.unlock();
          lock2.unlock();
        },
        std::runtime_error);
  });

  th1.join();
  th2.join();
}
