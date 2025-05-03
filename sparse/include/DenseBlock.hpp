#pragma once
#ifndef _DENSEBLOCK_HPP_
#define _DENSEBLOCK_HPP_
#include <BaseBlock.hpp>

namespace sparse {
                    namespace details {
                    using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
          } // namespace details
template <std::intptr_t BlockSize, typename _Ty>
struct DenseBlock : BlockInfo<BlockSize, true, _Ty> {

  static_assert((BlockSize & (BlockSize - 1)) == 0,
                "BlockSize must be a power of 2");

  using value_type = _Ty;
  using pointer = std::add_pointer_t<_Ty>;
  using reference = _Ty&;
  using const_reference = const reference;

  virtual std::optional<reference> operator()(const std::intptr_t x,
                                                                  const std::intptr_t y) override;

  virtual  std::optional<const  _Ty&>read(const std::intptr_t x,
                                                                  const std::intptr_t y) const override;

  virtual void write(const std::intptr_t x, const std::intptr_t y,
                     const _Ty &value) override;

  virtual std::optional < std::reference_wrapper<value_type>>
            fetch_pointer(const std::intptr_t x, const std::intptr_t y) const override;

  virtual  std::reference_wrapper<value_type>
            touch_pointer(const std::intptr_t x, const std::intptr_t y) override;

  template <typename Func> void foreach (Func &&func) {
    for (std::size_t x = 0; x < BlockSize; ++x) {
      for (std::size_t y = 0; y < BlockSize; ++y) {
        func(x, y, m_block(x, y));
      }
    }
  }

  _Ty m_block[BlockSize][BlockSize];

private:
          details::Coord2D getTransferredCoord(const std::intptr_t x, const std::intptr_t y);
};
} // namespace sparse

#endif //_DENSEBLOCK_HPP_
