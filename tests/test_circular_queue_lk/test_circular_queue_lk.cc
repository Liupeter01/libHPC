#include <thread>
#include <gtest/gtest.h>
#include <circular_queue_lk.hpp>

struct MyClass {
          MyClass() :p(new char[5] {}) ,value(3000){}
          ~MyClass() { delete []p; }

          MyClass(const MyClass& other)
                    : p(new char[5]), value(other.value) {
                    std::copy(other.p, other.p + 5, p);
          }

          MyClass& operator=(const MyClass& other) {
                    if (this != &other) {
                              delete[] p;
                              p = new char[5];
                              std::copy(other.p, other.p + 5, p);
                              value = other.value;
                    }
                    return *this;
          }

          // Move constructor
          MyClass(MyClass&& other) noexcept : p(other.p), value(other.value) {
                    other.p = nullptr;
                    other.value = 0;
          }

          // Move assignment
          MyClass& operator=(MyClass&& other) noexcept {
                    if (this != &other) {
                              delete[] p;
                              p = other.p;
                              value = other.value;

                              other.p = nullptr;
                              other.value = 0;
                    }
                    return *this;
          }


          char* p;
          std::size_t value;
};

TEST(CircularQueueLkTest, SingleThreadPushAndPop) {
          CircularQueueLk < MyClass, 5> queue;

          MyClass a, b;

          // Push copy
          EXPECT_TRUE(queue.push(a));

          // Push move
          EXPECT_TRUE(queue.push(std::move(b)));

          for (int i = 3; i <= 5; ++i) {
                    MyClass tmp;
                    EXPECT_TRUE(queue.push(std::move(tmp)));
          }

          // Should be full now
          MyClass extra;
          EXPECT_FALSE(queue.push(extra));

          for (int i = 0; i < 5; ++i) {
                    MyClass out;
                    EXPECT_TRUE(queue.pop(out));
          }

          MyClass out;
          EXPECT_FALSE(queue.pop(out));
}