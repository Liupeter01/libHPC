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
sparse::DenseBlock<BlockSize, _Ty>::operator()(const std::intptr_t x,
                                                                                const std::intptr_t y) const {
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
          return { m_block[new_x][new_y] };
}

template <std::intptr_t BlockSize, typename _Ty>
std::reference_wrapper<_Ty>
sparse::DenseBlock<BlockSize, _Ty>::touch_pointer(const std::intptr_t x, const std::intptr_t y) {
          auto opt = fetch_pointer(x, y);
          assert(opt.has_value() && "touch_pointer() called on invalid coordinate");
          return *opt;
}

template <std::intptr_t BlockSize, typename _Ty>
void sparse::DenseBlock<BlockSize, _Ty>::write(const std::intptr_t x,
                                               const std::intptr_t y,
                                               const _Ty &value) {

          touch_pointer(x, y).get() = value;
}

template <std::intptr_t BlockSize, typename _Ty>
sparse::details::Coord2D
sparse::DenseBlock<BlockSize, _Ty>::getTransferredCoord(const std::intptr_t x, const std::intptr_t y) const {
          return std::make_pair(x & this->BMask, y & this->BMask);
}