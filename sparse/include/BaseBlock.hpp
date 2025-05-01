#pragma once
#ifndef _BASEBLOCK_HPP_
#define _BASEBLOCK_HPP_
#include <memory>
#include <iostream>
#include <optional>
#include <type_traits>

namespace sparse {
namespace details {
static constexpr inline std::intptr_t constexpr_log2(std::intptr_t n) {
  return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1);
}
} // namespace details

template <std::intptr_t BlockSize, bool _is_leaf, typename _Ty>
struct BaseBlock {

  using value_type = _Ty;
  using pointer = std::add_pointer_t<_Ty>;
  using reference = _Ty &;
  using const_reference = std::add_const_t<reference>;

  static constexpr std::intptr_t B = BlockSize;
  static constexpr std::intptr_t BShift = details::constexpr_log2(BlockSize);
  static constexpr std::intptr_t BMask = BlockSize - 1;
  static constexpr std::intptr_t is_leaf = _is_leaf;

  virtual std::optional<reference> operator()(const std::intptr_t x,
                                                                       const std::intptr_t y) = 0;
  virtual std::optional<const_reference> &read(const std::intptr_t x,
                                                                           const std::intptr_t y) const = 0;

  virtual void write(const std::intptr_t x, const std::intptr_t y,
                                const _Ty& value) = 0;

  virtual std::optional < std::shared_ptr<value_type>> fetch_pointer(const std::intptr_t x, const std::intptr_t y) const = 0;
  virtual std::shared_ptr<value_type> touch_pointer(const std::intptr_t x, const std::intptr_t y) = 0;
};
} // namespace sparse

#endif //_DENSEBLOCK_HPP_
