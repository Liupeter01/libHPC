#include <gtest/gtest.h>
#include <queue_lockfree.hpp>
#include <thread>
#include <atomic>
#include <chrono>

#define NUMBER 2000000

TEST(LockFreeRefQueueTest, MultiThreadPush3Pop2_Stable_WithEmptyCheck) {
          concurrency::ConcurrentQueue<int> queue;

          auto producer = [&queue]() {
                    for (std::size_t i = 0; i < NUMBER; ++i) {
                              queue.push(i);
                    }
                    };

          std::vector<std::thread> producer_list;
          producer_list.emplace_back(producer);
          producer_list.emplace_back(producer);
          producer_list.emplace_back(producer);

          std::thread th4([&queue, producer_number = producer_list.size()]() {
                    thread_local std::size_t popped = 0;
                    while (popped < NUMBER * producer_number / 2) {
                              if (queue.empty()) {
                                        std::this_thread::sleep_for(std::chrono::microseconds(2));
                                        continue;
                              }
                              auto val = queue.pop();
                              if (val.has_value()) {
                                        ++popped;
                                        continue;
                              }
                    }
                    });

          std::thread th5([&queue, producer_number = producer_list.size()]() {
                    thread_local std::size_t popped = 0;
                    while (popped < NUMBER * producer_number / 2) {
                              if (queue.empty()) {
                                        std::this_thread::sleep_for(std::chrono::microseconds(2));
                              }
                              auto val = queue.pop();
                              if (val.has_value()) {
                                        ++popped;
                                        continue;
                              }
                    }
                    });

  

          th4.join();
          th5.join();

          for (auto& th : producer_list) {
                    if (th.joinable()) {
                              th.join();
                    }
          }

          EXPECT_TRUE(queue.empty());
}