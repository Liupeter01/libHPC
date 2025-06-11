#include <gtest/gtest.h>
#include <hierarchical_lock.hpp>

TEST(HierarchicalLockTest, ValidLockOrder) {
  hierarchical_lock lock1(1000);
  hierarchical_lock lock2(500);

  EXPECT_NO_THROW({
    lock1.lock();
    lock2.lock();
    lock2.unlock();
    lock1.unlock();
  });
}

TEST(HierarchicalLockTest, InvalidLockOrder) {
  hierarchical_lock lock1(1000);
  hierarchical_lock lock2(500);

  // there should be a throw
  EXPECT_THROW(
      {
        lock2.lock();
        lock1.lock();
        lock1.unlock();
        lock2.unlock();
      },
      std::runtime_error);
}
