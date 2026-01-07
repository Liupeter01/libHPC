#pragma once
#ifndef _CPU_HPP_
#define _CPU_HPP_
#include <vector>

void cpu_local_count_ref( const std::vector<uint32_t>& in, size_t numBlocks, std::vector<uint32_t>& out_local_count);
void cpu_local_offset_ref(std::vector<uint32_t>& local,  size_t numBlocks);
void cpu_global_base_ref(const std::vector<uint32_t>& in, std::vector<uint32_t>& base);

#endif //_CPU_HPP_