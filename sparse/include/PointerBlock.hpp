#pragma once
#ifndef _POINTERBLOCK_HPP_
#define  _POINTERLOCK_HPP_
#include <memory>
#include <tbb/spin_mutex.h>

namespace sparse {
          template <std::intptr_t PointerGridSize, typename OtherBlock>
          struct PointerBlock {
                    static_assert((PointerGridSize& (PointerGridSize - 1)) == 0,
                              "PointerGridSize must be a power of 2");

                    static constexpr std::intptr_t B = PointerGridSize;
                    static constexpr std::intptr_t BShift = details::constexpr_log2(PointerGridSize);
                    static constexpr std::intptr_t BMask = PointerGridSize - 1;
                    static constexpr bool is_leaf = false;   //this is the child of the structure

                    OtherBlock& operator()(const std::intptr_t x, const std::intptr_t y) const;
                    const OtherBlock& read(const std::intptr_t x, const std::intptr_t y) const;
                    OtherBlock* fetch(const std::intptr_t x, const std::intptr_t y) const;
                    void write(const std::intptr_t x, const std::intptr_t y, const OtherBlock& value);

                    template<typename Func>
                    void foreach(Func&& func) {}

                    tbb::spin_mutex m_spinlock[PointerGridSize][PointerGridSize];
                    std::unique_ptr< OtherBlock> m_data[PointerGridSize][PointerGridSize];
          };

}

#endif // _POINTERLOCK_HPP_