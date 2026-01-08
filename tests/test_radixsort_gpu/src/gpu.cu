#include <cudaAllocator.hpp>
#include <gpu.cuh>
#include <gtest/gtest.h>

void gpu_local_count_func(uint32_t *d_in, size_t N,
                          std::vector<uint32_t> &out_local_count) {

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v1::kernel_local_histogram_v1<256>
      <<<numBlocks, 256>>>(d_in, N, d_local.data().get(), numBlocks * BinSize,
                           /*rshift=*/0);

  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  out_local_count.resize(numBlocks * BinSize);
  thrust::copy(d_local.begin(), d_local.end(), out_local_count.begin());
}

void gpu_global_base(uint32_t *d_in, size_t N,
                     std::vector<uint32_t> &out_base) {
  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v1::kernel_local_histogram_v1<256>
      <<<numBlocks, 256>>>(d_in, N, d_local.data().get(), numBlocks * 256,
                           /*rshift=*/0);

  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  auto sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  ::sort::gpu::radix::details::v1::kernel_global_reduce_from_local_v1<256>
      <<<numBlocks, 256>>>(d_local.data().get(), numBlocks * 256,
                           d_global.data().get());

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  thrust::exclusive_scan(thrust::cuda::par.on(0), d_global.begin(),
                         d_global.end(), d_global.begin());

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  out_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), out_base.begin());
}
