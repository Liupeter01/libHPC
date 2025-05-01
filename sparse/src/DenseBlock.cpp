#include <DenseBlock.hpp>

constexpr std::intptr_t sparse::details::constexpr_log2(std::intptr_t n) {
  return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1);
}

template <std::intptr_t BlockSize, typename _Ty>
_Ty &sparse::DenseBlock<BlockSize, _Ty>::operator()(
    const std::intptr_t x, const std::intptr_t y) const {

  return m_block[x & BMask][y & BMask];
}

template <std::intptr_t BlockSize, typename _Ty>
_Ty *sparse::DenseBlock<BlockSize, _Ty>::fetch(const std::intptr_t x,
                                               const std::intptr_t y) const {

  return &m_block[x & BMask][y & BMask];
}

template <std::intptr_t BlockSize, typename _Ty>
const _Ty &
sparse::DenseBlock<BlockSize, _Ty>::read(const std::intptr_t x,
                                         const std::intptr_t y) const {

  return m_block[x & BMask][y & BMask];
}

template <std::intptr_t BlockSize, typename _Ty>
void sparse::DenseBlock<BlockSize, _Ty>::write(const std::intptr_t x,
                                               const std::intptr_t y,
                                               const _Ty &value) {

  m_block[x & BMask][y & BMask] = value;
}
