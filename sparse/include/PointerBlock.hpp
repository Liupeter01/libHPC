#pragma once
#ifndef _POINTERBLOCK_HPP_
#define _POINTERBLOCK_HPP_
#include <map>
#include <mutex>
#include <memory>
#include <BaseBlock.hpp>
#include <tbb/spin_mutex.h>

namespace sparse {
          namespace details {
                    using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
          } // namespace details
          
template <std::intptr_t PointerGridSize, typename OtherBlock>
struct PointerBlock 
          : BlockInfo<PointerGridSize, false, OtherBlock> 
{
            static_assert((PointerGridSize & (PointerGridSize - 1)) == 0,
                          "PointerGridSize must be a power of 2");

            /*related to subblock, we need its offset argument*/
            static constexpr std::intptr_t subblock_shift_bits = SubBlockInfo<OtherBlock>::offset_bits;

            using value_type = OtherBlock;
            using pointer = std::add_pointer_t<OtherBlock>;
            using reference = OtherBlock&;
            using const_reference = const reference;

           struct WriteAccessor {
                     WriteAccessor(PointerBlock& grid) : m_global(grid) {}
                     void write(const std::intptr_t x, const std::intptr_t y, const OtherBlock& value);
                     PointerBlock<PointerGridSize, OtherBlock>& m_global;
                     std::map<details::Coord2D, std::reference_wrapper<value_type>> m_cache;
           };

  virtual std::optional< reference> operator()(const std::intptr_t x,
            const std::intptr_t y) override;

  virtual std::optional< const  OtherBlock&> operator()(const std::intptr_t x,
            const std::intptr_t y) const override;

  virtual  std::optional<const  OtherBlock&>read(const std::intptr_t x,
            const std::intptr_t y) const override;

  virtual void write(const std::intptr_t x, const std::intptr_t y,
            const OtherBlock& value) override;

  virtual std::optional <std::reference_wrapper<value_type>>
            fetch_pointer(const std::intptr_t x, const std::intptr_t y) const override;

  virtual std::reference_wrapper<value_type>
            touch_pointer(const std::intptr_t x, const std::intptr_t y) override;

  WriteAccessor access() { return { *this }; }

  template <typename Func> void foreach (Func &&func) {
#pragma omp parallel for collapse(2)
            for (std::size_t x = 0; x < PointerGridSize; ++x) {
                      for (std::size_t y = 0; y < PointerGridSize; ++y) {
                                if (auto opt = operator()(x, y); opt) {
                                          func(x, y, *opt);
                                }
                      }
            }
  }

  tbb::spin_mutex m_spinlock[PointerGridSize][PointerGridSize];
  std::unique_ptr<OtherBlock> m_data[PointerGridSize][PointerGridSize];

private:
          details::Coord2D getTransferredCoord(const std::intptr_t x, const std::intptr_t y);
};

} // namespace sparse

#endif // _POINTERLOCK_HPP_
