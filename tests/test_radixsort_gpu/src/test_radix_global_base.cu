#include <algorithm>
#include <cpu.hpp>
#include <cudaAllocator.hpp>
#include <cuda_global_reduce_from_local.cuh>
#include <cuda_local_histogram.cuh>
#include <gpu.cuh>
#include <gtest/gtest.h>
#include <radix_sort_gpu.h>
#include <vector>

class RadixGlobalBaseTest : public ::testing::TestWithParam<size_t> {};

TEST_P(RadixGlobalBaseTest, Histogramv1GlobalReducev1) {
  size_t N = GetParam();

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v1::kernel_local_histogram_v1<256>
      <<<numBlocks, 256>>>(gpu_array.data(), N, d_local.data().get(),
                           numBlocks * 256,
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

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv1GlobalReducev2) {
  size_t N = GetParam();
  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  // GPU Part!
  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);
  thrust::fill(d_localT.begin(), d_localT.end(), 0);

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

  ::sort::gpu::radix::details::v2::kernel_global_reduce_from_local_v2<256>
      <<<BinSize, 32>>>(d_localT.data().get(), numBlocks * BinSize,
                        d_global.data().get(), numBlocks);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  thrust::exclusive_scan(thrust::cuda::par.on(0), d_global.begin(),
                         d_global.end(), d_global.begin());

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv1GlobalReducev3) {
  size_t N = GetParam();

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

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

  std::size_t in_stride = numBlocks;
  std::size_t in_real = numBlocks;

  std::size_t out_stride = pitch;
  std::size_t out_real = (in_real + 1024 - 1) / 1024;

  block = dim3(32, 32, 1);
  grid = dim3(out_real, BinSize, 1);

  // ---- round 0: localT -> write_ptr ----
  ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
      <<<grid, block>>>(localT_ptr, write_ptr, in_stride, in_real, out_stride,
                        out_real);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  // update next iteration
  in_stride = pitch;
  in_real = out_real;
  std::swap(read_ptr, write_ptr);

  while (in_real > 1) {

    out_real = (in_real + 1024 - 1) / 1024;
    grid = dim3(out_real, BinSize, 1);

    ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
        <<<grid, block>>>(read_ptr, write_ptr,
                          /*in_stride=*/pitch, /*in_real=*/in_real,
                          /*out_stride=*/pitch, /*out_real=*/out_real);

    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    sync_err = cudaDeviceSynchronize();
    ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

    in_real = out_real;
    std::swap(read_ptr, write_ptr);
  }

  uint32_t *final_result = read_ptr;

  // Scanning
  thrust::exclusive_scan(thrust::cuda::par.on(0), final_result,
                         final_result + BinSize, final_result);

  // Copy 2 Global
  cudaMemcpy(d_global.data().get(), final_result, BinSize * sizeof(uint32_t),
             cudaMemcpyDeviceToDevice // DeviceToHost
  );

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv1GlobalReducev1IgnoresPadding) {
  size_t N = GetParam();

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v1::kernel_local_histogram_v1<256>
      <<<numBlocks, 256>>>(gpu_array.data(), N, d_local.data().get(),
                           numBlocks * 256,
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

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv1GlobalReducev2IgnoresPadding) {
  size_t N = GetParam();

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);
  thrust::fill(d_localT.begin(), d_localT.end(), 0);

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

  ::sort::gpu::radix::details::v2::kernel_global_reduce_from_local_v2<256>
      <<<BinSize, 32>>>(d_localT.data().get(), numBlocks * BinSize,
                        d_global.data().get(), numBlocks);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  thrust::exclusive_scan(thrust::cuda::par.on(0), d_global.begin(),
                         d_global.end(), d_global.begin());

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv1GlobalReducev3IgnoresPadding) {
  size_t N = GetParam();

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

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

  std::size_t in_stride = numBlocks;
  std::size_t in_real = numBlocks;

  std::size_t out_stride = pitch;
  std::size_t out_real = (in_real + 1024 - 1) / 1024;

  block = dim3(32, 32, 1);
  grid = dim3(out_real, BinSize, 1);

  // ---- round 0: localT -> write_ptr ----
  ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
      <<<grid, block>>>(localT_ptr, write_ptr, in_stride, in_real, out_stride,
                        out_real);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  // update next iteration
  in_stride = pitch;
  in_real = out_real;
  std::swap(read_ptr, write_ptr);

  while (in_real > 1) {

    out_real = (in_real + 1024 - 1) / 1024;
    grid = dim3(out_real, BinSize, 1);

    ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
        <<<grid, block>>>(read_ptr, write_ptr,
                          /*in_stride=*/pitch, /*in_real=*/in_real,
                          /*out_stride=*/pitch, /*out_real=*/out_real);

    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    sync_err = cudaDeviceSynchronize();
    ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

    in_real = out_real;
    std::swap(read_ptr, write_ptr);
  }

  uint32_t *final_result = read_ptr;

  // Scanning
  thrust::exclusive_scan(thrust::cuda::par.on(0), final_result,
                         final_result + BinSize, final_result);

  // Copy 2 Global
  cudaMemcpy(d_global.data().get(), final_result, BinSize * sizeof(uint32_t),
             cudaMemcpyDeviceToDevice // DeviceToHost
  );

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv2GlobalReducev1) {
  size_t N = GetParam();

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v2::kernel_local_histogram_v2<BinSize>
      <<<numBlocks, BinSize>>>(gpu_array.data(), N, d_local.data().get(),
                               numBlocks * BinSize, 0);

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

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv2GlobalReducev2) {
  size_t N = GetParam();

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v2::kernel_local_histogram_v2<BinSize>
      <<<numBlocks, BinSize>>>(gpu_array.data(), N, d_local.data().get(),
                               numBlocks * BinSize, 0);

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

  ::sort::gpu::radix::details::v2::kernel_global_reduce_from_local_v2<256>
      <<<BinSize, 32>>>(d_localT.data().get(), numBlocks * BinSize,
                        d_global.data().get(), numBlocks);

  thrust::exclusive_scan(thrust::cuda::par.on(0), d_global.begin(),
                         d_global.end(), d_global.begin());

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv2GlobalReducev3) {
  size_t N = GetParam();

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v2::kernel_local_histogram_v2<BinSize>
      <<<numBlocks, BinSize>>>(gpu_array.data(), N, d_local.data().get(),
                               numBlocks * BinSize, 0);

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

  std::size_t in_stride = numBlocks;
  std::size_t in_real = numBlocks;

  std::size_t out_stride = pitch;
  std::size_t out_real = (in_real + 1024 - 1) / 1024;

  block = dim3(32, 32, 1);
  grid = dim3(out_real, BinSize, 1);

  // ---- round 0: localT -> write_ptr ----
  ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
      <<<grid, block>>>(localT_ptr, write_ptr, in_stride, in_real, out_stride,
                        out_real);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  // update next iteration
  in_stride = pitch;
  in_real = out_real;
  std::swap(read_ptr, write_ptr);

  while (in_real > 1) {

    out_real = (in_real + 1024 - 1) / 1024;
    grid = dim3(out_real, BinSize, 1);

    ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
        <<<grid, block>>>(read_ptr, write_ptr,
                          /*in_stride=*/pitch, /*in_real=*/in_real,
                          /*out_stride=*/pitch, /*out_real=*/out_real);

    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    sync_err = cudaDeviceSynchronize();
    ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

    in_real = out_real;
    std::swap(read_ptr, write_ptr);
  }

  uint32_t *final_result = read_ptr;

  // Scanning
  thrust::exclusive_scan(thrust::cuda::par.on(0), final_result,
                         final_result + BinSize, final_result);

  // Copy 2 Global
  cudaMemcpy(d_global.data().get(), final_result, BinSize * sizeof(uint32_t),
             cudaMemcpyDeviceToDevice // DeviceToHost
  );

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv2GlobalReducev1IgnoresPadding) {
  size_t N = GetParam();

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v2::kernel_local_histogram_v2<BinSize>
      <<<numBlocks, BinSize>>>(gpu_array.data(), N, d_local.data().get(),
                               numBlocks * BinSize, 0);

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

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv2GlobalReducev2IgnoresPadding) {
  size_t N = GetParam();

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v2::kernel_local_histogram_v2<BinSize>
      <<<numBlocks, BinSize>>>(gpu_array.data(), N, d_local.data().get(),
                               numBlocks * BinSize, 0);

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

  ::sort::gpu::radix::details::v2::kernel_global_reduce_from_local_v2<256>
      <<<BinSize, 32>>>(d_localT.data().get(), numBlocks * BinSize,
                        d_global.data().get(), numBlocks);

  thrust::exclusive_scan(thrust::cuda::par.on(0), d_global.begin(),
                         d_global.end(), d_global.begin());

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

TEST_P(RadixGlobalBaseTest, Histogramv2GlobalReducev3IgnoresPadding) {
  size_t N = GetParam();

  constexpr size_t BinSize = 256;
  size_t numBlocks = (N + BinSize - 1) / BinSize;
  size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> gpu_array;
  ::sort::gpu::radix::details::helper::generate_random(gpu_array, N);

  gpu_array.resize(padded_n, std::numeric_limits<uint32_t>::max());

  std::vector<uint32_t> cpu_base, gpu_base;

  cpu_global_base_ref(std::vector<uint32_t>(gpu_array.begin(), gpu_array.end()),
                      cpu_base);

  thrust::device_vector<uint32_t> d_global(BinSize);
  thrust::device_vector<uint32_t> d_local(numBlocks * BinSize);
  thrust::device_vector<uint32_t> d_localT(numBlocks * BinSize);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::fill(d_global.begin(), d_global.end(), 0);
  thrust::fill(d_local.begin(), d_local.end(), 0);

  ::sort::gpu::radix::details::v2::kernel_local_histogram_v2<BinSize>
      <<<numBlocks, BinSize>>>(gpu_array.data(), N, d_local.data().get(),
                               numBlocks * BinSize, 0);

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

  std::size_t in_stride = numBlocks;
  std::size_t in_real = numBlocks;

  std::size_t out_stride = pitch;
  std::size_t out_real = (in_real + 1024 - 1) / 1024;

  block = dim3(32, 32, 1);
  grid = dim3(out_real, BinSize, 1);

  // ---- round 0: localT -> write_ptr ----
  ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
      <<<grid, block>>>(localT_ptr, write_ptr, in_stride, in_real, out_stride,
                        out_real);

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  sync_err = cudaDeviceSynchronize();
  ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

  // update next iteration
  in_stride = pitch;
  in_real = out_real;
  std::swap(read_ptr, write_ptr);

  while (in_real > 1) {

    out_real = (in_real + 1024 - 1) / 1024;
    grid = dim3(out_real, BinSize, 1);

    ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
        <<<grid, block>>>(read_ptr, write_ptr,
                          /*in_stride=*/pitch, /*in_real=*/in_real,
                          /*out_stride=*/pitch, /*out_real=*/out_real);

    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    sync_err = cudaDeviceSynchronize();
    ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

    in_real = out_real;
    std::swap(read_ptr, write_ptr);
  }

  uint32_t *final_result = read_ptr;

  // Scanning
  thrust::exclusive_scan(thrust::cuda::par.on(0), final_result,
                         final_result + BinSize, final_result);

  // Copy 2 Global
  cudaMemcpy(d_global.data().get(), final_result, BinSize * sizeof(uint32_t),
             cudaMemcpyDeviceToDevice // DeviceToHost
  );

  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  gpu_base.resize(BinSize);
  thrust::copy(d_global.begin(), d_global.end(), gpu_base.begin());

  ASSERT_EQ(cpu_base.size(), gpu_base.size());
  for (size_t i = 0; i < cpu_base.size(); ++i) {
    ASSERT_EQ(cpu_base[i], gpu_base[i])
        << "Mismatch at N=" << N << " bin=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(RadixEdgeCases, RadixGlobalBaseTest,
                         ::testing::Values(1, 111, 256, 297, 500, 512, 3987,
                                           1024 * 256 + 57));
