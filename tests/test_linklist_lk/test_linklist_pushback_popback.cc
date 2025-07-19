#include <gtest/gtest.h>
#include <linklist_lk.hpp>
#include <thread>

TEST(LinkListLkTest, MultiThreadPushBackPopBack) {

  concurrency::LinkListLK<std::size_t> list;
  std::thread th1([&list]() {
    for (std::size_t i = 0; i < 10000; ++i) {
      list.push_back(i);
    }
  });

  std::thread th2([&list]() {
    std::this_thread::sleep_for(std::chrono::microseconds(2));
    for (std::size_t i = 0; i < 10000; ++i) {
      list.remove_if([i](const auto &value) { return value == i; });
    }
  });

  th1.join();
  th2.join();

  EXPECT_EQ(list.size(), 0);
}
