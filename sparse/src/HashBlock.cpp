#include <HashBlock.hpp>

template <typename OtherBlock>
 std::optional< OtherBlock&> 
sparse::HashBlock<OtherBlock>::operator()(const std::intptr_t x,
                                                                         const std::intptr_t y) {
           auto key = getKey(x, y);
           auto it = m_data.find(key);
           if (it == m_data.end()) {
                     return std::nullopt;
           }
           return *it->second;
}

 template <typename OtherBlock>
 std::optional< const OtherBlock&>
sparse::HashBlock<OtherBlock>::operator()(const std::intptr_t x,
                                                                         const std::intptr_t y)const {
           auto key = getKey(x, y);
           auto it = m_data.find(key);
           if (it == m_data.end()) {
                     return std::nullopt;
           }
           return *it->second;
 }

template <typename OtherBlock>
std::optional<const OtherBlock&>
sparse::HashBlock<OtherBlock>::read(const std::intptr_t x,
                                                                 const std::intptr_t y) const {
          return operator()(x, y);
}

template <typename OtherBlock>
void 
sparse::HashBlock<OtherBlock>::write(const std::intptr_t x, 
                                                                 const std::intptr_t y,
                                                                 const OtherBlock& value) {

          auto& shared = touch_pointer(x, y);
          *shared = value;
}

template <typename OtherBlock>
std::optional < std::reference_wrapper<OtherBlock>>
sparse::HashBlock<OtherBlock>::fetch_pointer(const std::intptr_t x, const std::intptr_t y) const {

          auto key = getKey(x, y);
          auto it = m_data.find(key);
          if (it == m_data.end()) {
                    return std::nullopt;
          }
          return { *it->second };
}

template <typename OtherBlock>
std::reference_wrapper<OtherBlock>
sparse::HashBlock<OtherBlock>::touch_pointer(const std::intptr_t x, const std::intptr_t y) {

          auto key = getKey(x, y);
          auto it = m_data.find(key);
          if (it != m_data.end()) {
                    return *it->second;
          }
          auto block = std::make_unique<OtherBlock>();
          auto ptr = block.get();
          m_data.emplace(key, std::move(block));
          return { *ptr };
}

template <typename OtherBlock>
sparse::details::Coord2D 
sparse::HashBlock<OtherBlock>::getKey(const std::intptr_t x, const std::intptr_t y)  const {
          return std::make_pair<std::intptr_t, std::intptr_t>(x, y);
          //Currently, we do not need to use this
          //return std::make_pair(x >> this->subblock_shift_bits, 
          //                                    y >> this->subblock_shift_bits);
}