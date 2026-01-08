#pragma once
#ifndef _GPU_CUH_
#define _GPU_CUH_
#include <cuda.h>
#include <cudaUtils.cuh>
#include <cuda_radix_sort.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <vector>

void gpu_local_count_func(uint32_t *d_in, size_t N,
                          std::vector<uint32_t> &out_local_count);
void gpu_global_base(uint32_t *d_in, size_t N, std::vector<uint32_t> &out_base);

template <size_t BlockSize = 32>
void gpu_local_offset_func(uint32_t *d_in, size_t N,
                           std::vector<uint32_t> &out_local_offset) {

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  // ---------- local_count ----------
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize, 0);

  ::sort::gpu::radix::details::v1::kernel_local_histogram_v1<256>
      <<<numBlocks, 256>>>(d_in, N, d_local.data().get(), numBlocks * BinSize,
                           /*rshift=*/0);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // ---------- transpose ----------
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  dim3 block(BlockSize, BlockSize);
  dim3 grid((BinSize + BlockSize - 1) / BlockSize,
            (numBlocks + BlockSize - 1) / BlockSize);

  ::gpu::util::kernel_transpose<uint32_t, BlockSize><<<grid, block>>>(
      d_localT.data().get(), d_local.data().get(), BinSize, numBlocks);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // ---------- per-bin scan ----------
  for (int bin = 0; bin < (int)BinSize; ++bin) {
    auto first = d_localT.begin() + bin * numBlocks;
    auto last = first + numBlocks;
    thrust::exclusive_scan(thrust::cuda::par.on(0), first, last, first);
  }

  // ---------- transpose back ----------
  dim3 grid_bt((numBlocks + BlockSize - 1) / BlockSize,
               (BinSize + BlockSize - 1) / BlockSize);

  ::gpu::util::kernel_transpose<uint32_t, BlockSize><<<grid_bt, block>>>(
      d_local.data().get(), d_localT.data().get(), numBlocks, BinSize);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // ---------- copy back ----------
  out_local_offset.resize(numBlocks * BinSize);
  thrust::copy(d_local.begin(), d_local.end(), out_local_offset.begin());
}

#endif //_GPU_CUH_
