#include <gtest/gtest.h>
#include <queue_lockfree.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>

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
                    std::size_t popped = 0;
                    thread_local std::mt19937 rng(std::random_device{}());
                    thread_local std::uniform_int_distribution<int> dist(1, 10);

                    while (popped < NUMBER * producer_number / 2) {
                              auto val = queue.pop();
                              if (val.has_value()) {
                                        ++popped;
                              }
                              else {
                                        std::this_thread::sleep_for(std::chrono::microseconds(dist(rng)));
                              }
                    }
                    });

          std::thread th5([&queue, producer_number = producer_list.size()]() {
                    std::size_t popped = 0;
                    thread_local std::mt19937 rng(std::random_device{}());
                    thread_local std::uniform_int_distribution<int> dist(5, 15);
                    while (popped < NUMBER * producer_number / 2) {
                              auto val = queue.pop();
                              if (val.has_value()) {
                                        ++popped;
                              }
                              else {
                                        std::this_thread::sleep_for(std::chrono::microseconds(dist(rng)));
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