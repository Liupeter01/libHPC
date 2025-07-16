#include <gtest/gtest.h>
#include <queue_lockfree.hpp>
#include <thread>

struct MyClass {
  int a;
};

#define TEST_POP_TRUE                                                          \
  {                                                                            \
    EXPECT_TRUE(queue.pop().has_value());                                      \
  }

#define TEST_POP_FALSE                                                         \
  {                                                                            \
    EXPECT_FALSE(queue.pop().has_value());                                     \
  }

TEST(LockFreeRefQueueTest, SingleThreadCompleteTest) {
  concurrency::ConcurrentQueue<MyClass> queue;

  MyClass a;
  a.a = 100;

  queue.push(a);
  queue.push(a);
  queue.push(a);

  queue.pop();
  queue.pop();
  queue.pop();

  EXPECT_TRUE(queue.empty());
}

TEST(LockFreeRefQueueTest, SingleThreadEmptyTest) {
  concurrency::ConcurrentQueue<MyClass> queue;

  MyClass a;
  a.a = 100;

  EXPECT_FALSE(queue.pop().has_value());

  queue.push(a);
  queue.push(a);
  queue.push(a);

  EXPECT_TRUE(queue.pop().has_value());
  EXPECT_TRUE(queue.pop().has_value());
  EXPECT_TRUE(queue.pop().has_value());
  EXPECT_FALSE(queue.pop().has_value());

  EXPECT_TRUE(queue.empty());
}
