#include <circular_queue_lockfree.hpp>
#include <gtest/gtest.h>
#include <thread>

struct MyClass {
  int a;
};

#define TEST_POP_TRUE                                                          \
  {                                                                            \
    MyClass out;                                                               \
    EXPECT_TRUE(queue.pop(out));                                               \
  }

#define TEST_POP_FALSE                                                         \
  {                                                                            \
    MyClass out;                                                               \
    EXPECT_FALSE(queue.pop(out));                                              \
  }

TEST(ConcurrentCircularQueue, SingleThreadPushAndPop) {
  concurrency::ConcurrentCircularQueue<MyClass, 3> queue;

  MyClass a, b;

  EXPECT_TRUE(queue.push(a));
  EXPECT_TRUE(queue.push(a));
  EXPECT_TRUE(queue.push(a));
  EXPECT_FALSE(queue.push(a));

  TEST_POP_TRUE
  TEST_POP_TRUE
  TEST_POP_TRUE
  TEST_POP_FALSE
}
