#include <gtest/gtest.h>
#include <queue_lockfree.hpp>
#include <thread>

struct MyClass {
  int a;
};

#define TESTCOUNT 50000

TEST(LockFreeRefQueueTest, SingleThreadReferenceCounterTest) {
  concurrency::ConcurrentQueue<MyClass> queue;

  MyClass a;
  a.a = 100;

  for (std::size_t i = 0; i < TESTCOUNT; ++i) {
    EXPECT_TRUE(queue.empty());
    queue.pop();
    queue.pop();
    queue.pop();

    EXPECT_TRUE(queue.empty());
    queue.push(a);

    EXPECT_FALSE(queue.empty());
    queue.pop();
  }

  EXPECT_TRUE(queue.empty());
}

TEST(LockFreeRefQueueTest, SingleThreadCompleteTest) {
  concurrency::ConcurrentQueue<MyClass> queue;

  MyClass a;
  a.a = 100;

  for (std::size_t i = 0; i < TESTCOUNT; ++i) {
    EXPECT_TRUE(queue.empty());
    queue.push(a);
    queue.push(a);
    EXPECT_FALSE(queue.empty());
    queue.push(a);

    queue.pop();
    queue.pop();
    EXPECT_FALSE(queue.empty());
    queue.pop();

    EXPECT_TRUE(queue.empty());
  }
}

TEST(LockFreeRefQueueTest, SingleThreadEmptyTest) {
  concurrency::ConcurrentQueue<MyClass> queue;

  MyClass a;
  a.a = 100;

  for (std::size_t i = 0; i < TESTCOUNT; ++i) {
    EXPECT_FALSE(queue.pop().has_value());
    EXPECT_TRUE(queue.empty());

    queue.push(a);
    queue.push(a);
    queue.push(a);

    EXPECT_TRUE(queue.pop().has_value());
    EXPECT_TRUE(queue.pop().has_value());
    EXPECT_TRUE(queue.pop().has_value());
    EXPECT_FALSE(queue.pop().has_value());
  }
  EXPECT_TRUE(queue.empty());
}

TEST(LockFreeRefQueueTest, SingleThreadCorrectValueTest) {
          concurrency::ConcurrentQueue<int> queue;

          for (std::size_t i = 0; i < TESTCOUNT; ++i) {
                    EXPECT_FALSE(queue.pop().has_value());
                    EXPECT_TRUE(queue.empty());

                    queue.push(i);

                    auto opt = queue.pop();
                    EXPECT_TRUE(opt.has_value());
                    EXPECT_EQ(*opt.value(), i);

                    EXPECT_TRUE(queue.empty());
                    EXPECT_FALSE(queue.pop().has_value());
          }
          EXPECT_TRUE(queue.empty());
}
