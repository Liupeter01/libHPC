#include <atomic>
#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <queue_lockfree.hpp>
#include <random>
#include <thread>

#define TESTCOUNT 10
#define NUMBER 2000000

TEST(LockFreeRefQueueTest, MultiThreadPush2Pop4_Stable_WithNotify) {
  for (std::size_t i = 0; i < TESTCOUNT; ++i) {
    concurrency::ConcurrentQueue<int> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<std::size_t> total_popped = 0;
    const std::size_t target = NUMBER * 2;

    // Producer
    auto producer = [&queue, &cv]() {
      for (std::size_t i = 0; i < NUMBER; ++i) {
        queue.push(i);
        cv.notify_one(); // notify a waiting consumer
      }
    };

    // Consumer
    auto consumer = [&]() {
      while (true) {
        if (total_popped.load(std::memory_order_acquire) >= target)
          break;

        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]() {
          return total_popped.load(std::memory_order_acquire) >= target ||
                 !queue.empty();
        });

        while (auto val = queue.pop()) {
          ++total_popped;
        }
      }
    };

    std::vector<std::thread> producers, consumers;
    for (int i = 0; i < 2; ++i) {
      producers.emplace_back(std::thread(producer));
    }
    for (int i = 0; i < 4; ++i) {
      consumers.emplace_back(std::thread(consumer));
    }

    for (auto &p : producers)
      p.join();
    for (auto &c : consumers)
      c.join();

    EXPECT_TRUE(queue.empty());
  }
}
