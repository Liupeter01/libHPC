#include <stack>
#include <stdio.h>
#include <gtest/gtest.h>
#include <single_mempool.hpp>

struct Data {
          char* ptr;
};

void handler(void *var) {
          if (!var) return;
          struct Data* d = (struct Data*)var;
          if (d->ptr) {
                    free(d->ptr);
                    d->ptr = NULL;
          }
}

TEST(SingleThreadMemoryPool, TestCleanupASANPureC) {
          struct ngx_pool_s* mypool = ngx_create_pool(512); //size is 512
          EXPECT_NE((int)mypool, 0);

          for (int i = 0; i < 1000; ++i) {
                    struct ngx_pool_cleanup_s* handle = ngx_pool_cleanup_add(mypool, sizeof(struct Data));
                    EXPECT_NE((int)handle, 0);
                    handle->handler = handler;
                    Data* data = (Data*)handle->data;
                    size_t len = strlen("helloworld") + 1;
                    data->ptr = (char*)malloc(len);
                    memset(data->ptr, 0, len);
                    strncpy(data->ptr, "helloworld", len);
          }

          ngx_destroy_pool(mypool);
}

TEST(SingleThreadMemoryPool, TestLargePureC) {
          struct ngx_pool_s* mypool = ngx_create_pool(512); //size is 512
          EXPECT_NE((int)mypool, 0);

          std::stack<char*> s;
          for (int i = 0; i < 1000; ++i) {
                    //create a large block
                    void* p = ngx_palloc(mypool, 1024);
                    EXPECT_NE((int)p, 0);
                    s.push((char*)p);
          }

          while (!s.empty()){
                    auto& value = s.top();
                    s.pop();
                    EXPECT_EQ(ngx_pfree(mypool, value), 0);
          }

          ngx_destroy_pool(mypool);
}