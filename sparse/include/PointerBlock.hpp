#pragma once
#ifndef _POINTERBLOCK_HPP_
#define _POINTERBLOCK_HPP_
#include <BaseBlock.hpp>
#include <map>
#include <atomic>

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

  PointerBlock() {
            for (std::size_t i = 0; i < PointerGridSize; ++i)
                      for (std::size_t j = 0; j < PointerGridSize; ++j)
                                m_data[i][j].store(nullptr, std::memory_order_relaxed);
  }

  virtual ~PointerBlock() {
            for (std::size_t x = 0; x < PointerGridSize; ++x)
                      for (std::size_t y = 0; y < PointerGridSize; ++y)
                                delete m_data[x][y].load(std::memory_order_relaxed);
  }

  using value_type = OtherBlock;
  using pointer = OtherBlock*;
  using reference = OtherBlock &;
  using const_value = const OtherBlock;

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
    return m_data[new_x][new_y].load(std::memory_order_acquire) != nullptr;
  }

  virtual std::optional<std::reference_wrapper<value_type>>
  operator()(const std::intptr_t x, const std::intptr_t y) override {
            auto [new_x, new_y] = getTransferredCoord(x, y);
            pointer block = m_data[new_x][new_y].load(std::memory_order_acquire);
            return block ? std::make_optional(std::ref(*block)) : std::nullopt;
  }

  virtual std::optional<std::reference_wrapper<const_value>>
  operator()(const std::intptr_t x, const std::intptr_t y) const override {
            auto [new_x, new_y] = getTransferredCoord(x, y);
            pointer block = m_data[new_x][new_y].load(std::memory_order_acquire);
            return block ? std::make_optional(std::cref(*block)) : std::nullopt;
  }

  virtual std::optional<std::reference_wrapper<const_value>>
  read(const std::intptr_t x, const std::intptr_t y) const override {
    return operator()(x, y);
  }

  virtual void write(const std::intptr_t x, const std::intptr_t y,
                     const OtherBlock &value) override {
    touch_pointer(x, y).get() = value;
  }

  virtual void write(const std::intptr_t x, const std::intptr_t y,
            OtherBlock&& value) override {
            touch_pointer(x, y).get() = std::move(value);
  }

  virtual std::optional<std::reference_wrapper<value_type>>
  fetch_pointer(const std::intptr_t x, const std::intptr_t y) override {
            return operator()(x, y);
  }

  virtual std::reference_wrapper<value_type>
  touch_pointer(const std::intptr_t x, const std::intptr_t y) override {
    auto [new_x, new_y] = getTransferredCoord(x, y);
    pointer block = m_data[new_x][new_y].load(std::memory_order_acquire);

    /*Impl DCLP Lock Check Method!*/

    if (!block) {
              pointer expected = nullptr;
              pointer desired = new OtherBlock;
              while (!m_data[new_x][new_y].compare_exchange_strong(
                        expected, desired,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {

                        if (expected != nullptr) {
                                  delete desired;
                                  block = expected;  
                                  break;
                        }
              }

              if (block == nullptr) {
                        block = desired;
              }
    }
    return std::ref(*block);
  }

  WriteAccessor access() { return {*this}; }

  template <typename Func> void foreach (Func &&func) {
#pragma omp parallel for collapse(2)
    for (std::size_t x = 0; x < PointerGridSize; ++x) {
      for (std::size_t y = 0; y < PointerGridSize; ++y) {
                pointer block = m_data[x][y].load(std::memory_order_acquire);
                if (block) {
                          func(x, y, *block);
                }
      }
    }
  }

  PointerBlock& operator=(const PointerBlock& other) {
            if (this == &other)
                      return *this;
            for (std::size_t x = 0; x < PointerGridSize; ++x) {
                      for (std::size_t y = 0; y < PointerGridSize; ++y) {
                                pointer src = other.m_data[x][y].load(std::memory_order_relaxed);
                                m_data[x][y].store(!src ? nullptr : new OtherBlock(*src), std::memory_order_relaxed);
                      }
            }
    return *this;
  }

  std::atomic< pointer> m_data[PointerGridSize][PointerGridSize]{ nullptr };

private:
  details::Coord2D getTransferredCoord(const std::intptr_t x,
                                       const std::intptr_t y) const {
    return std::make_pair(x & this->BMask, y & this->BMask);
  }
};

} // namespace sparse

#endif // _POINTERLOCK_HPP_
