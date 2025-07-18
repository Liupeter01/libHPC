#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <queue_lockfree.hpp>
#include <random>
#include <thread>

#define TESTCOUNT 10
#define NUMBER 2000000

void MultiThreadPush3Pop2_Stable_WithNotify(){
          concurrency::ConcurrentQueue<int> queue;

          std::mutex mtx;
          std::condition_variable cv;
          std::atomic<std::size_t> total_popped = 0;
          const std::size_t target = NUMBER * 3;

          auto producer = [&]() {
                    for (std::size_t i = 0; i < NUMBER; ++i) {
                              queue.push(i);
                              cv.notify_one();
                    }
                    };

          auto consumer = [&]() {
                    while (true) {
                              if (total_popped.load(std::memory_order_acquire) >= target)
                                        break;

                              std::unique_lock<std::mutex> lock(mtx);
                              cv.wait(lock, [&]() {
                                        return total_popped.load(std::memory_order_acquire) >= target ||
                                                  !queue.empty();
                                        });

                              if (total_popped.load(std::memory_order_acquire) >= target)
                                        break;

                              if (auto val = queue.pop(); val) {
                                        ++total_popped;
                              }
                    }
                    };

          std::vector<std::thread> producers, consumers;
          for (int i = 0; i < 3; ++i)
                    producers.emplace_back(producer);
          for (int i = 0; i < 2; ++i)
                    consumers.emplace_back(consumer);

          for (auto& t : producers) t.join();
          for (auto& t : consumers) t.join();

          EXPECT_TRUE(queue.empty());
}

TEST(LockFreeRefQueueTest, MultiThreadPush3Pop2_Multiple_Times) {
          for (std::size_t i = 0; i < TESTCOUNT; ++i) {
                    MultiThreadPush3Pop2_Stable_WithNotify();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
}