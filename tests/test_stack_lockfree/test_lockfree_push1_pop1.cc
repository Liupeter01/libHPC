#include <thread>
#include <gtest/gtest.h>
#include <stack_lockfree.hpp>

#define NUMBER 2000000

TEST(LockFreeStackTest, OneThreadForPushAndPop) {

          concurrency::ConcurrentStack<std::size_t> list;

          auto producer = [&list]() {
                    for (std::size_t i = 0; i < NUMBER; ++i) {
                              list.push(i);
                    }
                    };

          auto consumer = [&list]() {
                    std::size_t popped = 0;
                    while (popped < NUMBER) {
                              auto val = list.pop();
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

          EXPECT_EQ(list.size(), 0);
}