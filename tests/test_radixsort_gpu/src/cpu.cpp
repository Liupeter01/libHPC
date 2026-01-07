#include <cpu.hpp>

void cpu_global_base_ref(const std::vector<uint32_t>& in, std::vector<uint32_t>& base) {
          constexpr size_t BinSize = 256;
          std::vector<uint32_t> hist(BinSize, 0);

          for (uint32_t v : in)
                    hist[v & 255]++;

          base.resize(BinSize);
          uint32_t sum = 0;
          for (size_t i = 0; i < BinSize; ++i) {
                    base[i] = sum;
                    sum += hist[i];
          }
}

void cpu_local_count_ref(
          const std::vector<uint32_t>& in,
          size_t numBlocks,
          std::vector<uint32_t>& out_local_count) {
          constexpr size_t BinSize = 256;
          out_local_count.assign(numBlocks * BinSize, 0);

          for (size_t b = 0; b < numBlocks; ++b) {
                    size_t begin = b * BinSize;
                    size_t end = std::min(begin + BinSize, in.size());

                    for (size_t i = begin; i < end; ++i) {
                              uint32_t v = in[i];
                              size_t bin = v & 255;
                              out_local_count[b * BinSize + bin]++;
                    }
          }
}

void cpu_local_offset_ref(
          std::vector<uint32_t>& local, // in-place
          size_t numBlocks) {
          constexpr size_t BinSize = 256;

          for (size_t bin = 0; bin < BinSize; ++bin) {
                    uint32_t sum = 0;
                    for (size_t b = 0; b < numBlocks; ++b) {
                              uint32_t tmp = local[b * BinSize + bin];
                              local[b * BinSize + bin] = sum;
                              sum += tmp;
                    }
          }
}