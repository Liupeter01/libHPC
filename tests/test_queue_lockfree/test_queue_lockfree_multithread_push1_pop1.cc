#include <gtest/gtest.h>
#include <queue_lockfree.hpp>
#include <thread>

#define NUMBER 2000000

TEST(LockFreeRefQueueTest, MultiThreadPush1Pop1) {
  concurrency::ConcurrentQueue<int> queue;

  auto producer = [&queue]() {
    for (std::size_t i = 0; i < NUMBER; ++i) {
      queue.push(i);
    }
  };

  auto consumer = [&queue]() {
    std::size_t popped = 0;
    while (popped < NUMBER) {
      auto val = queue.pop();
      if (val.has_value()) {
        ++popped;
        continue;
      }
      std::this_thread::yield();
    }
  };

  std::thread th1(producer);
  std::thread th2(consumer);

  th1.join();
  th2.join();

  EXPECT_TRUE(queue.empty());
}
