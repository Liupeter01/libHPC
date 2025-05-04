#pragma once
#ifndef _ROOTGRID_HPP_
#define _ROOTGRID_HPP_
#include <iostream>
#include <optional>

namespace sparse {
namespace details {
using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
} // namespace details

template <typename _Ty, typename _Layout> struct RootGrid {

  template <typename Node>
  std::optional<_Ty> _read(const Node &node, std::intptr_t x, std::intptr_t y) {
    // if it is denseblock, then it should be the last node point!
    if constexpr (Node::is_leaf) {
      return node.read(x, y);
    } else {
      const auto [new_x, new_y] = generateCoord<Node>(x, y);
      /*if it is hash/pointer block, then find the next block and do
       * recursion!*/
      auto next_node = node.read(new_x, new_y);

      if (!next_node.has_value())
        return std::nullopt;
      // you have to pass global coordinates to the function!
      return _read(next_node.value(), x, y);
    }
  }

  template <typename Node>
  void _write(Node &node, std::intptr_t x, std::intptr_t y, const _Ty &value) {
    // if it is denseblock, then it should be the last node point!
    if constexpr (Node::is_leaf) {
      node.write(x, y, value);
    } else {
      const auto [new_x, new_y] = generateCoord<Node>(x, y);
      /*if it is hash/pointer block, then find the next block and do
       * recursion!*/
      auto &next_node = node.touch_pointer(new_x, new_y);
      _write(next_node.get(), x, y, value);
    }
  }

  std::optional<_Ty> read(std::intptr_t x, std::intptr_t y) const {
    return _read(m_root, x, y);
  }
  void write(std::intptr_t x, std::intptr_t y, const _Ty &value) {
    _write(m_root, x, y, value);
  }

  template <typename Func> void foreach (Func &&func);

  _Layout m_root; // maybe start from hashmap

private:
  template <typename Node>
  details::Coord2D generateCoord(const std::intptr_t x, const std::intptr_t y) {
    return std::make_pair(x >> Node::subblock_shift_bits,
                          y >> Node::subblock_shift_bits);
  }
};
} // namespace sparse

#endif //_ROOTGRID_HPP_
