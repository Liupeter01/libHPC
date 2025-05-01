#pragma once
#include <iterator>
#include <optional>
#ifndef _DENSEBLOCK_HPP_
#define _DENSEBLOCK_HPP_
#include <BaseBlock.hpp>

namespace sparse {

template <std::intptr_t BlockSize, typename _Ty>
struct DenseBlock : BaseBlock<BlockSize, true, _Ty> {
  static_assert((BlockSize & (BlockSize - 1)) == 0,
                "BlockSize must be a power of 2");

  using BaseType = ::sparse::BaseBlock<BlockSize, true, _Ty>;

  virtual std::optional<_Ty> operator()(const std::intptr_t x,
                                        const std::intptr_t y) const override;

  virtual const _Ty &read(const std::intptr_t x,
                          const std::intptr_t y) const override;
  virtual _Ty *fetch(const std::intptr_t x,
                     const std::intptr_t y) const override;
  virtual void write(const std::intptr_t x, const std::intptr_t y,
                     const _Ty &value) override;

  template <typename Func> void foreach (Func &&func) {
    for (std::size_t x = 0; x < BlockSize; ++x) {
      for (std::size_t y = 0; y < BlockSize; ++y) {
        func(x, y, m_block(x, y));
      }
    }
  }

  _Ty m_block[BlockSize][BlockSize];
};
} // namespace sparse

#endif //_DENSEBLOCK_HPP_
