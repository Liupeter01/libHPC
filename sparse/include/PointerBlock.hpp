#pragma once
#ifndef _POINTERBLOCK_HPP_
#define _POINTERBLOCK_HPP_
#include <BaseBlock.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <tbb/spin_mutex.h>

namespace sparse {
namespace details {
using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
} // namespace details

template <std::intptr_t PointerGridSize, typename OtherBlock>
struct PointerBlock : BlockInfo<PointerGridSize, false, OtherBlock> {
  static_assert((PointerGridSize & (PointerGridSize - 1)) == 0,
                "PointerGridSize must be a power of 2");

  /*related to subblock, we need its offset argument*/
  static constexpr std::intptr_t subblock_shift_bits =
      SubBlockInfo<OtherBlock>::offset_bits;

  using value_type = OtherBlock;
  using reference = OtherBlock &;
  using const_value = const  OtherBlock;

  struct WriteAccessor {
    WriteAccessor(PointerBlock &grid) : m_global(grid) {}
    void write(const std::intptr_t x, const std::intptr_t y,
               const OtherBlock &value) {
      auto key = m_global.getTransferredCoord(x, y);
      auto it = m_cache.find(key);

      if (it != m_cache.end()) {
        it->second.get().write(x, y, value);
        return;
      }

      auto &ref = m_global.touch_pointer(x, y);
      ref.get().write(x, y, value);
      m_cache.try_emplace(key, ref);
    }

    PointerBlock<PointerGridSize, OtherBlock> &m_global;
    std::map<details::Coord2D, std::reference_wrapper<value_type>> m_cache;
  };

  bool has(std::intptr_t x, std::intptr_t y) const {
            auto [new_x, new_y] = getTransferredCoord(x, y);
            return m_data[new_x][new_y] != nullptr;
  }

  virtual std::optional<std::reference_wrapper<value_type>> operator()(const std::intptr_t x,
                                              const std::intptr_t y) override {
    auto [new_x, new_y] = getTransferredCoord(x, y);
    auto &block = m_data[new_x][new_y];
    if (!block)
      return std::nullopt;
    return std::make_optional(std::ref(*block)); 
  }

  virtual std::optional<std::reference_wrapper<const_value>>
  operator()(const std::intptr_t x, const std::intptr_t y) const override {
    auto [new_x, new_y] = getTransferredCoord(x, y);
    auto &block = m_data[new_x][new_y];
    if (!block)
      return std::nullopt;
    return std::make_optional(std::cref(*block));
  }

  virtual std::optional<std::reference_wrapper<const_value>>
  read(const std::intptr_t x, const std::intptr_t y) const override {
    return operator()(x, y);
  }

  virtual void write(const std::intptr_t x, const std::intptr_t y,
                     const OtherBlock &value) override {
    touch_pointer(x, y).get() = value;
  }

  virtual std::optional<std::reference_wrapper<value_type>>
  fetch_pointer(const std::intptr_t x, const std::intptr_t y) override {
    auto [new_x, new_y] = getTransferredCoord(x, y);
    auto &block = m_data[new_x][new_y];
    if (!block)
      return std::nullopt;

    return std::make_optional(std::ref(*block));
  }

  virtual std::reference_wrapper<value_type>
  touch_pointer(const std::intptr_t x, const std::intptr_t y) override {
    auto [new_x, new_y] = getTransferredCoord(x, y);
    auto &block = m_data[new_x][new_y];

    /*Impl DCLP Lock Check Method!*/
    if (!block) {
      std::lock_guard<tbb::spin_mutex> _lck(m_spinlock[new_x][new_y]);
      if (!m_data[new_x][new_y])
        m_data[new_x][new_y] = std::make_unique<OtherBlock>();
    }
    return std::ref(*m_data[new_x][new_y]);
  }

  WriteAccessor access() { return {*this}; }

  template <typename Func> void foreach (Func &&func) {
#pragma omp parallel for collapse(2)
    for (std::size_t x = 0; x < PointerGridSize; ++x) {
      for (std::size_t y = 0; y < PointerGridSize; ++y) {
        if (auto opt = operator()(x, y); opt) {
                  func(x, y, opt->get());
        }
      }
    }
  }

  tbb::spin_mutex m_spinlock[PointerGridSize][PointerGridSize];
  std::unique_ptr<OtherBlock> m_data[PointerGridSize][PointerGridSize];

private:
  details::Coord2D getTransferredCoord(const std::intptr_t x,
                                       const std::intptr_t y) {
    return std::make_pair(x & this->BMask, y & this->BMask);
  }
};

} // namespace sparse

#endif // _POINTERLOCK_HPP_
