#include <algorithm>
#include <cpu.hpp>
#include <cudaAllocator.hpp>
#include <cuda_hierarchical_exclusive_scan_localT_1024.cuh>
#include <gpu.cuh>
#include <gtest/gtest.h>
#include <radix_sort_gpu.h>
#include <thrust/device_vector.h>
#include <vector>

class RadixLocalOffsetTest : public ::testing::TestWithParam<size_t> {};

TEST(RadixLocalOffsetTest, KernelStartupOnly) {
  ::sort::gpu::radix::details::__kernel_startup();
}

TEST_P(RadixLocalOffsetTest, GlobalLocalThrustScan) {

  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  constexpr size_t BlockSize = 32;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  // input
  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  // ---------- CPU reference ----------
  std::vector<uint32_t> cpu_local;
  cpu_local_count_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      numBlocks, cpu_local);
  cpu_local_offset_ref(cpu_local, numBlocks);

  // ---------- GPU ----------
  std::vector<uint32_t> gpu_local;
  gpu_local_offset_func<BlockSize>(gpu_array.data(), N, gpu_local);

  // ---------- compare ----------
  ASSERT_EQ(cpu_local.size(), gpu_local.size());
  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local[b * BinSize + bin], gpu_local[b * BinSize + bin])
          << "Mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }
}

TEST_P(RadixLocalOffsetTest, GlobalLocalThrustScanIgnoresPadding) {

  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  constexpr size_t BlockSize = 32;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  // input
  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  // ---------- CPU reference ----------
  std::vector<uint32_t> cpu_local;
  cpu_local_count_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      numBlocks, cpu_local);
  cpu_local_offset_ref(cpu_local, numBlocks);

  // ---------- GPU ----------
  std::vector<uint32_t> gpu_local;
  gpu_local_offset_func<BlockSize>(gpu_array.data(), N, gpu_local);

  // ---------- compare ----------
  ASSERT_EQ(cpu_local.size(), gpu_local.size());
  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local[b * BinSize + bin], gpu_local[b * BinSize + bin])
          << "Mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }
}

TEST_P(RadixLocalOffsetTest, GlobalLocalHierarchicalScanV1) {

  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  constexpr size_t BlockSize = 32;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  constexpr std::size_t fan_in = 1024;

  // input
  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  // ---------- CPU reference ----------
  std::vector<uint32_t> cpu_local;
  cpu_local_count_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      numBlocks, cpu_local);
  cpu_local_offset_ref(cpu_local, numBlocks);

  // ---------- GPU ----------
  std::vector<uint32_t> gpu_local;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::device_vector<uint32_t> big_buffer;
  std::vector<::sort::gpu::radix::details::LevelDesc> levels =
      ::sort::gpu::radix::details::build_level_layout_and_allocate(
          big_buffer, BinSize, numBlocks, 32);

  cudahelper::
      kernel_clear_u32<<<(d_global.size() + fan_in - 1) / fan_in, fan_in>>>(
          d_global.data().get(), d_global.size());

  cudahelper::
      kernel_clear_u32<<<(d_local.size() + fan_in - 1) / fan_in, fan_in>>>(
          d_local.data().get(), d_local.size());

  cudahelper::
      kernel_clear_u32<<<(d_localT.size() + fan_in - 1) / fan_in, fan_in>>>(
          d_localT.data().get(), d_localT.size());

  cudahelper::
      kernel_clear_u32<<<(reduce_a.size() + fan_in - 1) / fan_in, fan_in>>>(
          reduce_a.data().get(), reduce_a.size());

  cudahelper::
      kernel_clear_u32<<<(reduce_b.size() + fan_in - 1) / fan_in, fan_in>>>(
          reduce_b.data().get(), reduce_b.size());

  cudahelper::
      kernel_clear_u32<<<(big_buffer.size() + fan_in - 1) / fan_in, fan_in>>>(
          big_buffer.data().get(), big_buffer.size());

  ::sort::gpu::radix::details::v1::kernel_local_histogram_v1<256>
      <<<numBlocks, 256>>>(gpu_array.data(), N, d_local.data().get(),
                           numBlocks * 256,
                           /*rshift=*/0);

  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  auto sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  // Transpose localT[bin][block] = local[block][bin]
  dim3 block(32, 32, 1);
  dim3 grid((BinSize + 32 - 1) / 32, (numBlocks + 32 - 1) / 32, 1);

  ::gpu::util::kernel_transpose<uint32_t, 32><<<grid, block>>>(
      /*out = */ d_localT.data().get(),
      /*in = */ d_local.data().get(), BinSize, numBlocks);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  uint32_t *local_ptr = d_local.data().get();
  uint32_t *localT_ptr = d_localT.data().get();
  uint32_t *read_ptr = reduce_a.data().get();
  uint32_t *write_ptr = reduce_b.data().get();

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  ::sort::gpu::radix::details::hierarchical_exclusive_scan_localT_1024(
      localT_ptr, big_buffer, levels, BinSize, numBlocks, numBlocks);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  dim3 grid_bt((numBlocks + BlockSize - 1) / BlockSize, // width = numBlocks
               (BinSize + BlockSize - 1) / BlockSize,   // height = BinSize
               1);

  nvtxRangePushA("transposeT");
  ::gpu::util::kernel_transpose<uint32_t, BlockSize><<<grid_bt, block>>>(
      /*out = */ d_local.data().get(),
      /*in = */ d_localT.data().get(), numBlocks, BinSize);
  nvtxRangePop();

  cudahelper::checkCuda(cudaGetLastError());
  cudahelper::checkCuda(cudaDeviceSynchronize());

  gpu_local.resize(numBlocks * BinSize);
  thrust::copy(d_local.begin(), d_local.end(), gpu_local.begin());

  // ---------- compare ----------
  ASSERT_EQ(cpu_local.size(), gpu_local.size());
  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local[b * BinSize + bin], gpu_local[b * BinSize + bin])
          << "Mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }

  d_global.clear();
  d_local.clear();
  d_localT.clear();
  reduce_a.clear();
  reduce_b.clear();

  big_buffer.clear();
}

TEST_P(RadixLocalOffsetTest, GlobalLocalHierarchicalScanV1IgnoresPadding) {

  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  constexpr size_t BlockSize = 32;

  constexpr std::size_t fan_in = 1024;

  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);
  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  // CPU reference: only original N
  std::vector<uint32_t> cpu_local;
  cpu_local_count_ref(
      std::vector<uint32_t>(gpu_array.begin(), gpu_array.begin() + N),
      numBlocks, cpu_local);
  cpu_local_offset_ref(cpu_local, numBlocks);

  // ---------- GPU ----------
  std::vector<uint32_t> gpu_local;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::device_vector<uint32_t> big_buffer;
  std::vector<::sort::gpu::radix::details::LevelDesc> levels =
      ::sort::gpu::radix::details::build_level_layout_and_allocate(
          big_buffer, BinSize, numBlocks, 32);

  cudahelper::
      kernel_clear_u32<<<(d_global.size() + fan_in - 1) / fan_in, fan_in>>>(
          d_global.data().get(), d_global.size());

  cudahelper::
      kernel_clear_u32<<<(d_local.size() + fan_in - 1) / fan_in, fan_in>>>(
          d_local.data().get(), d_local.size());

  cudahelper::
      kernel_clear_u32<<<(d_localT.size() + fan_in - 1) / fan_in, fan_in>>>(
          d_localT.data().get(), d_localT.size());

  cudahelper::
      kernel_clear_u32<<<(reduce_a.size() + fan_in - 1) / fan_in, fan_in>>>(
          reduce_a.data().get(), reduce_a.size());

  cudahelper::
      kernel_clear_u32<<<(reduce_b.size() + fan_in - 1) / fan_in, fan_in>>>(
          reduce_b.data().get(), reduce_b.size());

  cudahelper::
      kernel_clear_u32<<<(big_buffer.size() + fan_in - 1) / fan_in, fan_in>>>(
          big_buffer.data().get(), big_buffer.size());

  ::sort::gpu::radix::details::v1::kernel_local_histogram_v1<256>
      <<<numBlocks, 256>>>(gpu_array.data(), N, d_local.data().get(),
                           numBlocks * 256,
                           /*rshift=*/0);

  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  auto sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  // Transpose localT[bin][block] = local[block][bin]
  dim3 block(32, 32, 1);
  dim3 grid((BinSize + 32 - 1) / 32, (numBlocks + 32 - 1) / 32, 1);

  ::gpu::util::kernel_transpose<uint32_t, 32><<<grid, block>>>(
      /*out = */ d_localT.data().get(),
      /*in = */ d_local.data().get(), BinSize, numBlocks);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  uint32_t *local_ptr = d_local.data().get();
  uint32_t *localT_ptr = d_localT.data().get();
  uint32_t *read_ptr = reduce_a.data().get();
  uint32_t *write_ptr = reduce_b.data().get();

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  ::sort::gpu::radix::details::hierarchical_exclusive_scan_localT_1024(
      localT_ptr, big_buffer, levels, BinSize, numBlocks, numBlocks);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  dim3 grid_bt((numBlocks + BlockSize - 1) / BlockSize, // width = numBlocks
               (BinSize + BlockSize - 1) / BlockSize,   // height = BinSize
               1);

  nvtxRangePushA("transposeT");
  ::gpu::util::kernel_transpose<uint32_t, BlockSize><<<grid_bt, block>>>(
      /*out = */ d_local.data().get(),
      /*in = */ d_localT.data().get(), numBlocks, BinSize);
  nvtxRangePop();

  cudahelper::checkCuda(cudaGetLastError());
  cudahelper::checkCuda(cudaDeviceSynchronize());

  gpu_local.resize(numBlocks * BinSize);
  thrust::copy(d_local.begin(), d_local.end(), gpu_local.begin());

  for (size_t b = 0; b < numBlocks; ++b) {
    for (size_t bin = 0; bin < BinSize; ++bin) {
      ASSERT_EQ(cpu_local[b * BinSize + bin], gpu_local[b * BinSize + bin])
          << "Padding mismatch at N=" << N << " block=" << b << " bin=" << bin;
    }
  }

  d_global.clear();
  d_local.clear();
  d_localT.clear();
  reduce_a.clear();
  reduce_b.clear();

  big_buffer.clear();
}

INSTANTIATE_TEST_SUITE_P(RadixEdgeCases, RadixLocalOffsetTest,
                         ::testing::Values(1, 111, 256, 297, 500, 512, 3987,
                                           1024 * 256 + 57, 290'000'000));
