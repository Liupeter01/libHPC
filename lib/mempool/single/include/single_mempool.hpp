#pragma once
#ifndef _SINGLE_MEMORY_POOL_HPP_
#define _SINGLE_MEMORY_POOL_HPP_

extern "C" {
#include <single_mempool_impl.h>
}
#include <memory>
#include <vector>

namespace mempool {

namespace details {

template <typename _Ty, typename... Args>
static _Ty *construct(_Ty *raw, Args... args) {
  return ::new (reinterpret_cast<void *>(raw)) _Ty(std::forward<Args>(args)...);
}

template <typename _Ty> static void destroy(void *raw) {
  if (!raw)
    return;

  if constexpr (!std::is_trivially_destructible_v<_Ty>) {
    reinterpret_cast<_Ty *>(raw)->~_Ty();
  }
}
} // namespace details

class SingleMemoryPool {
  struct NgxPoolDeletor {
    void operator()(ngx_pool_s *s);
  };

public:
  SingleMemoryPool() : size_{}, pool_(nullptr) {}
  SingleMemoryPool(const std::size_t pool_size,
                   const std::size_t alignment = NGX_POOL_ALIGNMENT);
  virtual ~SingleMemoryPool() { destroy(); }

public:
  template <typename _Ty, typename... Args>
  [[nodiscard]]
  std::shared_ptr<_Ty> create(Args &&...args) {
    return construct<_Ty>(std::forward<Args>(args)...);
  }

  template <typename _Ty, typename... Args>
  [[nodiscard]]
  std::vector<std::shared_ptr<_Ty>> create_batch(const std::size_t number,
                                                 Args &&...args) {
    std::vector<std::shared_ptr<_Ty>> ret;
    ret.reserve(number);
    for (std::size_t i = 0; i < number; ++i)
      ret.emplace_back(create<_Ty>(std::forward<Args>(args)...));
    return ret;
  }

  void reset();
  void destroy();
  void resize(const std::size_t pool_size,
              const std::size_t alignment = NGX_POOL_ALIGNMENT);

protected:
  void init(const std::size_t pool_size,
            const std::size_t alignment = NGX_POOL_ALIGNMENT);

  template <typename _Ty, typename... Args,
            std::enable_if_t<!std::is_trivially_destructible_v<_Ty>, int> = 0>
  [[nodiscard]]
  std::shared_ptr<_Ty> construct(Args &&...args) {
    ngx_pool_cleanup_s *handle = ngx_pool_cleanup_add(pool_.get(), sizeof(_Ty));
    if (!handle || !handle->data)
      throw std::bad_alloc();

    _Ty *raw = details::construct(reinterpret_cast<_Ty *>(handle->data),
                                  std::forward<Args>(args)...);
    handle->handler = [](void *ptr) { details::destroy<_Ty>(ptr); };
    return std::shared_ptr<_Ty>(raw, []([[maybe_unused]] _Ty *) noexcept {});
  }

  template <typename _Ty, typename... Args,
            std::enable_if_t<std::is_trivially_destructible_v<_Ty>, int> = 0>
  [[nodiscard]]
  std::shared_ptr<_Ty> construct(Args &&...args) {
    _Ty *raw = reinterpret_cast<_Ty *>(allocate(sizeof(_Ty)));
    if (!raw)
      throw std::bad_alloc();

    raw = details::construct(raw, std::forward<Args>(args)...);

    return std::shared_ptr<_Ty>(
        raw, [pool = pool_.get(), size = size_](_Ty *s) {
          if (s) {
            details::destroy<_Ty>(s);
            if (sizeof(_Ty) >= size) {
              [[maybe_unused]] bool status = ngx_pfree(pool, s);
            }
          }
        });
  }

private:
  /*allocate memory from memory pool or malloc directly!*/
  void *allocate(const std::size_t size);

private:
  std::size_t size_;
  std::unique_ptr<ngx_pool_s, NgxPoolDeletor> pool_{nullptr};
};
} // namespace mempool

#endif //_SINGLE_MEMEORY_POOL_HPP_
