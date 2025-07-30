#include <stdexcept>
#include <single_mempool.hpp>

void mempool::SingleMemoryPool::NgxPoolDeletor::operator()(ngx_pool_s* s) {
          if(s) ngx_destroy_pool(s);
}

mempool::SingleMemoryPool::SingleMemoryPool(const std::size_t pool_size,
                                                                                    const std::size_t alignment){

          init(pool_size, alignment);
}

void* mempool::SingleMemoryPool::allocate(const std::size_t size) {
          if (!size)  throw std::invalid_argument("Cannot allocate zero-size memory");
          void* raw = ngx_palloc(pool_.get(), size);
          if (!raw)  throw std::bad_alloc();
          return raw;
}

void mempool::SingleMemoryPool::reset() {
          if (pool_) ngx_reset_pool(pool_.get());
}

void mempool::SingleMemoryPool::destroy() {
          if (pool_)  pool_.reset(nullptr);
}

void mempool::SingleMemoryPool::init(const std::size_t pool_size,
          const std::size_t alignment) {

          ngx_pool_s* raw = ngx_create_pool(pool_size);
          if (!raw)  throw std::bad_alloc();
          size_ = raw->d.end - raw->d.last;
          pool_.reset(raw);
}

void mempool::SingleMemoryPool::resize(const std::size_t pool_size,
                                                                      const std::size_t alignment ) {

          destroy();
          init(pool_size, alignment);
}