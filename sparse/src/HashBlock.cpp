#include <HashBlock.hpp>

template <typename OtherBlock>
OtherBlock &
sparse::HashBlock<OtherBlock>::operator()(const std::intptr_t x,
                                          const std::intptr_t y) const {}

template <typename OtherBlock>
const OtherBlock &
sparse::HashBlock<OtherBlock>::read(const std::intptr_t x,
                                    const std::intptr_t y) const {}

template <typename OtherBlock>
OtherBlock *sparse::HashBlock<OtherBlock>::fetch(const std::intptr_t x,
                                                 const std::intptr_t y) const {}

template <typename OtherBlock>
void sparse::HashBlock<OtherBlock>::write(const std::intptr_t x,
                                          const std::intptr_t y,
                                          const OtherBlock &value) {}
