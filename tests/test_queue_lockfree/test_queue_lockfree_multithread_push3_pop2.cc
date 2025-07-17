#include <gtest/gtest.h>
#include <queue_lockfree.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>

#define TESTCOUNT 10
#define NUMBER 2000000

TEST(LockFreeRefQueueTest, MultiThreadPush3Pop2_Stable_WithEmptyCheck) {

          for (std::size_t i = 0; i < TESTCOUNT; ++i) {
                    concurrency::ConcurrentQueue<int> queue;

                    std::mutex mtx;
                    std::condition_variable cv;
                    std::atomic<std::size_t> total_popped = 0;
                    const std::size_t target = NUMBER * 3;

                    auto producer = [&]() {
                              for (std::size_t i = 0; i < NUMBER; ++i) {
                                        queue.push(i);
                                        cv.notify_all(); 
                              }
                              };

                    auto consumer = [&]() {
                              std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
                              while (true) {
                                        if (total_popped.load() >= target) break;

                                        auto val = queue.pop();
                                        if (val.has_value()) {
                                                  ++total_popped;
                                        }
                                        else {
                                                  lock.lock();
                                                  cv.wait_for(lock, std::chrono::microseconds(10));  
                                                  lock.unlock();
                                        }
                              }
                              };

                    std::vector<std::thread> producers;
                    std::vector<std::thread> consumers;
                    for (int i = 0; i < 3; ++i) 
                              producers.emplace_back(std::thread(producer));
                    for (int i = 0; i < 2; ++i) 
                              consumers.emplace_back(std::thread(consumer));

                    for (auto& t : producers) t.join();
                    for (auto& t : consumers) t.join();

                    EXPECT_TRUE(queue.empty());
          }
}