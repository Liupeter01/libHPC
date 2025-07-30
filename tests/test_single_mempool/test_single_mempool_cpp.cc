#include <gtest/gtest.h>
#include <single_mempool.hpp>

struct Trivial {
          int a;
          int* p;
};

struct NonTrivial {
          NonTrivial(const std::string &s):p(new int),c(s) {}
          ~NonTrivial() { delete p; }
          int* p;
          std::string c;
};

TEST(SingleThreadMemoryPool, TestTrivial) {
          mempool::SingleMemoryPool pool(1024);
          auto trivial_ptr = pool.create<Trivial>();
          trivial_ptr->a = 42;
          trivial_ptr->p = nullptr;
          EXPECT_EQ(trivial_ptr->a, 42);
}

TEST(SingleThreadMemoryPool, TestNonTrivial) {
          mempool::SingleMemoryPool pool(1024);
          auto non_trivial_ptr = pool.create<NonTrivial>("hello");
          EXPECT_NE(non_trivial_ptr->p, nullptr);
          EXPECT_EQ(non_trivial_ptr->c, "hello");
}

TEST(SingleThreadMemoryPool, TestNonTrivialArray) {
          mempool::SingleMemoryPool pool(1024);
          auto non_trivial_ptr = pool.create_batch<NonTrivial>(3, "hello");

          EXPECT_NE(non_trivial_ptr[0]->p, nullptr);
          EXPECT_EQ(non_trivial_ptr[0]->c, "hello");
          EXPECT_NE(non_trivial_ptr[1]->p, nullptr);
          EXPECT_EQ(non_trivial_ptr[1]->c, "hello");
          EXPECT_NE(non_trivial_ptr[2]->p, nullptr);
          EXPECT_EQ(non_trivial_ptr[2]->c, "hello");
}

TEST(SingleThreadMemoryPool, TestBigChunkNonTrivial) {
          struct LargeNonTrivial {
                    LargeNonTrivial(const std::string& s) :p(new int), c(s) {}
                    ~LargeNonTrivial() { delete p; }
                    int* p;
                    std::string c;
                    char dummy[4096];
          };
          mempool::SingleMemoryPool pool(1024);
          auto non_trivial_ptr = pool.create<LargeNonTrivial>("hello");
          EXPECT_NE(non_trivial_ptr->p, nullptr);
          EXPECT_EQ(non_trivial_ptr->c, "hello");
}

TEST(SingleThreadMemoryPool, TestBigChunkTrivial) {
          struct LargeTrivial {
                    int a;
                    int* p;
                    char dummy[4096];
          };

          mempool::SingleMemoryPool pool(1024);
          auto trivial_ptr = pool.create<LargeTrivial>();
          trivial_ptr->a = 42;
          trivial_ptr->p = nullptr;
          EXPECT_EQ(trivial_ptr->a, 42);
}