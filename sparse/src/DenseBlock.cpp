#include "BaseBlock.hpp"
#include <DenseBlock.hpp>

template <std::intptr_t BlockSize, typename _Ty>
std::optional<_Ty&>
sparse::DenseBlock<BlockSize, _Ty>::operator()(const std::intptr_t x,
                                               const std::intptr_t y) {

          auto [new_x, new_y] = getTransferredCoord(x, y);
          return m_block[new_x][new_y];
}

template <std::intptr_t BlockSize, typename _Ty>
std::optional<const _Ty&>
sparse::DenseBlock<BlockSize, _Ty>::read(const std::intptr_t x,
          const std::intptr_t y) const {

          return operator()(x, y);
}

template <std::intptr_t BlockSize, typename _Ty>
std::optional <  std::reference_wrapper<_Ty>>
sparse::DenseBlock<BlockSize, _Ty>::fetch_pointer(const std::intptr_t x, const std::intptr_t y) const {
          auto [new_x, new_y] = getTransferredCoord(x, y);
          return { &m_block[new_x][new_y] };
}

template <std::intptr_t BlockSize, typename _Ty>
std::reference_wrapper<_Ty>
sparse::DenseBlock<BlockSize, _Ty>::touch_pointer(const std::intptr_t x, const std::intptr_t y) {
          return  fetch_pointer(x, y);
}

template <std::intptr_t BlockSize, typename _Ty>
void sparse::DenseBlock<BlockSize, _Ty>::write(const std::intptr_t x,
                                               const std::intptr_t y,
                                               const _Ty &value) {

          auto ref_opt = operator()(x, y);
          if (ref_opt) {
                    *ref_opt = value;
          }
}

template <std::intptr_t BlockSize, typename _Ty>
sparse::details::Coord2D
sparse::DenseBlock<BlockSize, _Ty>::getTransferredCoord(const std::intptr_t x, const std::intptr_t y) {
          return std::make_pair(x & this->BMask, y & this->BMask);
}