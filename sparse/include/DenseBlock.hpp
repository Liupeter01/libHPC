#pragma once
#ifndef _DENSEBLOCK_HPP_
#define _DENSEBLOCK_HPP_
#include <iostream>

namespace sparse {
          namespace details {
                    static constexpr inline std::intptr_t constexpr_log2(std::intptr_t n);
          }

          template <std::intptr_t BlockSize, typename _Ty >
          struct DenseBlock {
                    static_assert((BlockSize& (BlockSize - 1)) == 0, "BlockSize must be a power of 2");

                    static constexpr std::intptr_t B = BlockSize;
                    static constexpr std::intptr_t BShift = details::constexpr_log2(BlockSize);
                    static constexpr std::intptr_t BMask = BlockSize - 1;
                    static constexpr bool is_leaf = true;   //this is the last node of the structure

                    _Ty& operator()(const std::intptr_t x, const std::intptr_t y) const;
                    const _Ty& read(const std::intptr_t x, const std::intptr_t y) const;
                    _Ty* fetch(const std::intptr_t x, const std::intptr_t y) const;
                    void write(const std::intptr_t x, const std::intptr_t y, const _Ty& value);

                    template<typename Func>
                    void foreach(Func&& func) {
                              for (std::size_t x = 0; x < BlockSize; ++x) {
                                        for (std::size_t y = 0; y < BlockSize; ++y) {
                                                  func(x, y, m_block(x, y));
                                        }
                              }
                    }

                    _Ty m_block[BlockSize][BlockSize];
          };
}

#endif //_DENSEBLOCK_HPP_