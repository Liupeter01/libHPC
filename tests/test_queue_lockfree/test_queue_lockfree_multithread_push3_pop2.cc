#include <queue_lockfree.hpp>
#include <gtest/gtest.h>
#include <thread>

#define NUMBER 2000000

TEST(LockFreeRefQueueTest, MultiThreadPush3Pop2) {
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
                              auto val = queue.pop();
                              if (val.has_value()) {
                                        ++popped;
                                        continue;
                              }
                              std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                    });

          std::thread th5([&queue, producer_number = producer_list.size()]() {
                    thread_local std::size_t popped = 0;
                    while (popped < NUMBER * producer_number / 2) {
                              auto val = queue.pop();
                              if (val.has_value()) {
                                        ++popped;
                                        continue;
                              }
                              std::this_thread::sleep_for(std::chrono::microseconds(1));
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