#pragma once
#ifndef _MEMPOOL_HPP_
#define _MEMPOOL_HPP_
extern "C" {
#include <single_mempool_impl.h>
}
#include <Singleton.hpp>

namespace mempool {
class SingleMemoryPool : public Singleton<SingleMemoryPool> {
  friend class Singleton<SingleMemoryPool>;

public:
  ~SingleMemoryPool() {}

private:
  SingleMemoryPool() {}
};
} // namespace mempool

#endif //_MEMPOOL_HPP_
