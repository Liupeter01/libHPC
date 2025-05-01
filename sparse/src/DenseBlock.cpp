#include "BaseBlock.hpp"
#include <DenseBlock.hpp>

template <std::intptr_t BlockSize, typename _Ty>
std::optional<_Ty>
sparse::DenseBlock<BlockSize, _Ty>::operator()(const std::intptr_t x,
                                               const std::intptr_t y) const {
  return m_block[x & BaseType::BMask][y & BaseType::BMask];
}

template <std::intptr_t BlockSize, typename _Ty>
_Ty *sparse::DenseBlock<BlockSize, _Ty>::fetch(const std::intptr_t x,
                                               const std::intptr_t y) const {

  return &m_block[x & BaseType::BMask][y & BaseType::BMask];
}

template <std::intptr_t BlockSize, typename _Ty>
const _Ty &
sparse::DenseBlock<BlockSize, _Ty>::read(const std::intptr_t x,
                                         const std::intptr_t y) const {

  return m_block[x & BaseType::BMask][y & BaseType::BMask];
}

template <std::intptr_t BlockSize, typename _Ty>
void sparse::DenseBlock<BlockSize, _Ty>::write(const std::intptr_t x,
                                               const std::intptr_t y,
                                               const _Ty &value) {

  m_block[x & BaseType::BMask][y & BaseType::BMask] = value;
}
