#pragma once
#ifndef _HASHBLOCK_HPP_
#define _HASHBLOCK_HPP_
#include <BaseBlock.hpp>
#include <iostream>
#include <unordered_map>

namespace sparse {
namespace details {
using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
template <typename SizeT>
static void hash_combine_impl(SizeT &seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace details

template <typename OtherBlock>
struct HashBlock
    : public BlockInfo<1, false, OtherBlock> /*only related to current block,
                                                which is hashblock*/
{
  using value_type = OtherBlock;
  using reference = OtherBlock &;
  using const_reference = const OtherBlock &;
  using CurrBlockType = BlockInfo<1, false, OtherBlock>;

  /*related to subblock, we need its offset argument*/
  static constexpr std::intptr_t subblock_shift_bits =
      SubBlockInfo<OtherBlock>::offset_bits;

  virtual std::optional<reference> operator()(const std::intptr_t x,
                                              const std::intptr_t y) override {
    auto key = getKey(x, y);
    auto it = m_data.find(key);
    if (it == m_data.end()) {
      return std::nullopt;
    }
    return *it->second;
  }

  virtual std::optional<const_reference>
  operator()(const std::intptr_t x, const std::intptr_t y) const override {
    auto key = getKey(x, y);
    auto it = m_data.find(key);
    if (it == m_data.end()) {
      return std::nullopt;
    }
    return *it->second;
  }

  virtual std::optional<const_reference>
  read(const std::intptr_t x, const std::intptr_t y) const override {
    return operator()(x, y);
  }

  virtual void write(const std::intptr_t x, const std::intptr_t y,
                     const OtherBlock &value) override {
    touch_pointer(x, y).get() = value;
  }

  virtual std::optional<std::reference_wrapper<value_type>>
  fetch_pointer(const std::intptr_t x, const std::intptr_t y) const override {
    auto key = getKey(x, y);
    auto it = m_data.find(key);
    if (it == m_data.end()) {
      return std::nullopt;
    }
    return {*it->second};
  }

  virtual std::reference_wrapper<value_type>
  touch_pointer(const std::intptr_t x, const std::intptr_t y) override {
    auto key = getKey(x, y);
    auto it = m_data.find(key);
    if (it != m_data.end()) {
      return *it->second;
    }
    auto block = std::make_unique<OtherBlock>();
    auto ptr = block.get();
    m_data.emplace(key, std::move(block));
    return {*ptr};
  }

  template <typename Func> void foreach (Func &&func) {
    for (auto &[key, value] : m_data) {
      func(key.first, key.second, *value);
    }
  }

  std::unordered_map<details::Coord2D, std::unique_ptr<OtherBlock>,
                     std::hash<details::Coord2D>>
      m_data;

private:
  details::Coord2D getKey(const std::intptr_t x, const std::intptr_t y) const {
    return std::make_pair(x, y);
    // Currently, we do not need to use this
    // return std::make_pair(x >> this->subblock_shift_bits,
    //                       y >> this->subblock_shift_bits);
  }
};
} // namespace sparse

namespace std {
template <> struct std::hash<sparse::details::Coord2D> {
  std::intptr_t operator()(const sparse::details::Coord2D &vertex) const {
    std::intptr_t seed = 0;
    sparse::details::hash_combine_impl(seed, vertex.first);
    sparse::details::hash_combine_impl(seed, vertex.second);
    return seed;
  }
};
} // namespace std

#endif // _HASHBLOCK_HPP_
