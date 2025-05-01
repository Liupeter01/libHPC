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
struct HashBlock : BaseBlock<1, false, OtherBlock> {

  // Hides BaseBlock::BMask intentionally to mark a wildcard block mask
  static constexpr std::intptr_t BMask = ~0;

  using BaseType = ::sparse::BaseBlock<1, false, OtherBlock>;

  virtual OtherBlock &operator()(const std::intptr_t x,
                                 const std::intptr_t y) const override;
  virtual const OtherBlock &read(const std::intptr_t x,
                                 const std::intptr_t y) const override;
  virtual OtherBlock *fetch(const std::intptr_t x,
                            const std::intptr_t y) const override;
  virtual void write(const std::intptr_t x, const std::intptr_t y,
                     const OtherBlock &value) override;

  template <typename Func> void foreach (Func &&func) {}

  std::unordered_map<details::Coord2D, OtherBlock, std::hash<details::Coord2D>>
      m_data;
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
