#include <PointerBlock.hpp>

template <std::intptr_t PointerGridSize, typename OtherBlock>
std::optional<OtherBlock&>
sparse::PointerBlock<PointerGridSize, OtherBlock>::operator()(
    const std::intptr_t x, const std::intptr_t y){

          auto [new_x, new_y] = getTransferredCoord(x, y);
          auto& block = m_data[new_x][new_y];
          if (!block)
                    return std::nullopt;
          return { *block };
}

template <std::intptr_t PointerGridSize, typename OtherBlock>
std::optional< const  OtherBlock&> 
sparse::PointerBlock<PointerGridSize, OtherBlock>::operator()(const std::intptr_t x,
          const std::intptr_t y) const{
          auto [new_x, new_y] = getTransferredCoord(x, y);
          auto& block = m_data[new_x][new_y];
          if (!block)
                    return std::nullopt;
          return { *block }; 
}

template <std::intptr_t PointerGridSize, typename OtherBlock>
std::optional < const OtherBlock&>
sparse::PointerBlock<PointerGridSize, OtherBlock>::read(
    const std::intptr_t x, const std::intptr_t y) const {

  return operator()(x, y);
}

template <std::intptr_t PointerGridSize, typename OtherBlock>
std::optional <std::reference_wrapper<OtherBlock>>
sparse::PointerBlock<PointerGridSize, OtherBlock>::fetch_pointer(const std::intptr_t x, const std::intptr_t y) const {

          auto [new_x, new_y] = getTransferredCoord(x, y);
          auto& block = m_data[new_x][new_y];
          if (!block)
                    return std::nullopt;

          return { *block };
}

template <std::intptr_t PointerGridSize, typename OtherBlock>
std::reference_wrapper<OtherBlock>
sparse::PointerBlock<PointerGridSize, OtherBlock>::touch_pointer(const std::intptr_t x, const std::intptr_t y) {

          auto [new_x, new_y] = getTransferredCoord(x, y);
          auto& block = m_data[new_x][new_y];

          /*Impl DCLP Lock Check Method!*/
          if (!block) {
                    std::lock_guard<tbb::spin_mutex> _lck(m_spinlock[new_x][new_y]);
                    if (!m_data[new_x][new_y])
                              m_data[new_x][new_y] = std::make_unique < OtherBlock>();
          }
          return { *m_data[new_x][new_y] };
}

template <std::intptr_t PointerGridSize, typename OtherBlock>
void sparse::PointerBlock<PointerGridSize, OtherBlock>::write(
    const std::intptr_t x, const std::intptr_t y, const OtherBlock &value) {

          auto block_ptr = touch_pointer(x, y);
          block_ptr.get().write(x, y, value);
}

template <std::intptr_t PointerGridSize, typename OtherBlock>
sparse::details::Coord2D 
sparse::PointerBlock<PointerGridSize, OtherBlock>::getTransferredCoord(const std::intptr_t x, const std::intptr_t y) {
          return std::make_pair(x & this->BMask, y & this->BMask);
}

template <std::intptr_t PointerGridSize, typename OtherBlock>
void 
sparse::PointerBlock<PointerGridSize, OtherBlock>::WriteAccessor::write(const std::intptr_t x, 
                                                                                                                        const std::intptr_t y, 
                                                                                                                        const OtherBlock& value) {
          auto key = m_global.getTransferredCoord(x, y);
          auto it = m_cache.find(key);

          if (it != m_cache.end()) {
                    it->second.get().write(x, y, value); 
                    return;
          }

          auto& ref = m_global.touch_pointer(x, y); 
          ref.get().write(x, y, value);
          m_cache.try_emplace(key, ref);
}