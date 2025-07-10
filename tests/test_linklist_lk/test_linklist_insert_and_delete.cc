#include <thread>
#include <gtest/gtest.h>
#include <linklist_lk.hpp>

TEST(LinkListLkTest, MultiThreadInsertAndDelete) {
          concurrency::LinkListLK<std::size_t> list;

          std::thread th1([&list]() {
                    for (std::size_t i = 0; i < 100; ++i) {
                              if (i % 2 == 0)
                                        list.push_front(i);
                              else
                                        list.push_back(i);
                    }
                    });

          std::thread th2([&list]() {
                    for (std::size_t i = 100; i < 300; ++i) {
                              if (i % 2 == 0)
                                         list.push_back(i);
                              else
                                        list.push_front(i);
                    }
                    });

          std::thread th3([&list]() {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                    for (std::size_t i = 0; i < 300; ++i) {
                              list.remove_if([i](const auto& value) { return value == i; });
                    }
                    });


          th1.join();
          th2.join();
          th3.join();

          EXPECT_EQ(list.size(), 0);
}