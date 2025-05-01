#pragma once
#ifndef _HASHBLOCK_HPP_
#define  _HASHBLOCK_HPP_
#include <iostream>
#include <unordered_map>

namespace sparse {
          namespace details {
                    using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
                    template <typename SizeT>
                    static void hash_combine_impl(SizeT& seed, SizeT value) {
                              seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    }
          }

          template <typename OtherBlock>
          struct HashBlock {
                    static constexpr std::intptr_t B = 1;
                    static constexpr std::intptr_t BShift = 0;
                    static constexpr std::intptr_t BMask = ~0;
                    static constexpr bool is_leaf = false;   //this is the child of the structure

                    OtherBlock& operator()(const std::intptr_t x, const std::intptr_t y) const;
                    const OtherBlock& read(const std::intptr_t x, const std::intptr_t y) const;
                    OtherBlock* fetch(const std::intptr_t x, const std::intptr_t y) const;
                    void write(const std::intptr_t x, const std::intptr_t y, const OtherBlock& value);

                    template<typename Func>
                    void foreach(Func&& func) {}

                    std::unordered_map<details::Coord2D, OtherBlock, std::hash<details::Coord2D>> m_data;
          };
}

namespace std {
          template <> struct std::hash<sparse::details::Coord2D> {
                    std::intptr_t  operator()(const sparse::details::Coord2D& vertex) const {
                              std::intptr_t seed = 0;
                              sparse::details::hash_combine_impl(seed, vertex.first);
                              sparse::details::hash_combine_impl(seed, vertex.second);
                              return seed;
                    }
          };
}

#endif // _HASHBLOCK_HPP_