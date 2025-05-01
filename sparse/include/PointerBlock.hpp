#pragma once
#ifndef _POINTERBLOCK_HPP_
#define _POINTERBLOCK_HPP_
#include <BaseBlock.hpp>
#include <memory>
#include <tbb/spin_mutex.h>

namespace sparse {

template <std::intptr_t PointerGridSize, typename OtherBlock>
struct PointerBlock : BaseBlock<PointerGridSize, false, OtherBlock> {
  static_assert((PointerGridSize & (PointerGridSize - 1)) == 0,
                "PointerGridSize must be a power of 2");

  using BaseType = ::sparse::BaseBlock<PointerGridSize, false, OtherBlock>;

  virtual OtherBlock &operator()(const std::intptr_t x,
                                 const std::intptr_t y) const override;
  virtual const OtherBlock &read(const std::intptr_t x,
                                 const std::intptr_t y) const override;
  virtual OtherBlock *fetch(const std::intptr_t x,
                            const std::intptr_t y) const override;
  virtual void write(const std::intptr_t x, const std::intptr_t y,
                     const OtherBlock &value) override;

  template <typename Func> void foreach (Func &&func) {}

  tbb::spin_mutex m_spinlock[PointerGridSize][PointerGridSize];
  std::unique_ptr<OtherBlock> m_data[PointerGridSize][PointerGridSize];
};

} // namespace sparse

#endif // _POINTERLOCK_HPP_
