#include <array>
#include <cudaHelper.cuh>
#include <cudaUtils.cuh>
#include <cuda_global_reduce_from_local.cuh>
#include <cuda_hierarchical_exclusive_scan_localT_1024.cuh>
#include <cuda_local_histogram.cuh>
#include <cuda_radix_scatter.cuh>
#include <cuda_radix_sort_v4.cuh>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h >

namespace sort {
namespace gpu {
namespace radix {
namespace details {
namespace v4 {
template <std::size_t BinSize, std::size_t BlockSize,
          std::enable_if_t<(BinSize == 256u), int> = 0>
static inline void radix_sort_v4(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input) {
  static constexpr std::size_t bits_per_round = constexpr_log2(BinSize);
  static constexpr std::size_t round_number =
      std::numeric_limits<uint32_t>::digits / bits_per_round;
  static constexpr std::size_t fan_in = BlockSize * BlockSize; // 1024

  static_assert((BinSize & (BinSize - 1)) == 0, "BinSize must be power of two");
  static_assert((std::numeric_limits<uint32_t>::digits % bits_per_round) == 0,
                "BinSize must evenly divide key bit width");

  static_assert(BinSize == 256u, "BinSize must be 256 in v1");
  static_assert(std::is_unsigned_v<uint32_t>,
                "radix_sort expects unsigned integers");

  if (input.empty())
    return;

  const std::size_t original_n = input.size();
  const std::size_t numBlocks = (original_n + BinSize - 1) / BinSize;
  const std::size_t padded_n = numBlocks * BinSize;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> backup;
  input.resize(padded_n, std::numeric_limits<uint32_t>::max());
  backup.resize(padded_n, std::numeric_limits<uint32_t>::max());

  uint32_t *ping = input.data();
  uint32_t *pong = backup.data();

  // The global array base is primarily used to record the total size of each
  // bin in the range[0, BinSize).
  //   This global array is obtained by merging the per - block local
  //   histograms, and by applying a prefix sum,
  //             we can compute the base offset of each bin in the global output
  //             array(i.e., the global base).

  thrust::device_vector<uint32_t> global(BinSize);

  //[numBlocks][BinSize] & [BinSize][numBlocks]
  thrust::device_vector<uint32_t> local(padded_n), localT(padded_n);

  const std::size_t pitch =
      std::max<std::size_t>(1, (numBlocks + 1024 - 1) / 1024);

  // fixed pitch buffer2D: [BinSize][pitch]
  thrust::device_vector<uint32_t> reduce_a(BinSize * pitch);
  thrust::device_vector<uint32_t> reduce_b(BinSize * pitch);

  thrust::device_vector<uint32_t> big_buffer;
  std::vector<::sort::gpu::radix::details::LevelDesc> levels =
      ::sort::gpu::radix::details::build_level_layout_and_allocate(
          big_buffer, BinSize, numBlocks, 32);

  for (std::size_t round = 0; round < round_number; ++round) {

    cudahelper::
        kernel_clear_u32<<<(global.size() + fan_in - 1) / fan_in, fan_in>>>(
            global.data().get(), global.size());

    cudahelper::
        kernel_clear_u32<<<(local.size() + fan_in - 1) / fan_in, fan_in>>>(
            local.data().get(), local.size());

    cudahelper::
        kernel_clear_u32<<<(localT.size() + fan_in - 1) / fan_in, fan_in>>>(
            localT.data().get(), localT.size());

    cudahelper::
        kernel_clear_u32<<<(reduce_a.size() + fan_in - 1) / fan_in, fan_in>>>(
            reduce_a.data().get(), reduce_a.size());

    cudahelper::
        kernel_clear_u32<<<(reduce_b.size() + fan_in - 1) / fan_in, fan_in>>>(
            reduce_b.data().get(), reduce_b.size());

    cudahelper::
        kernel_clear_u32<<<(big_buffer.size() + fan_in - 1) / fan_in, fan_in>>>(
            big_buffer.data().get(), big_buffer.size());

    // generate local
    nvtxRangePushA("histogram");
    v2::kernel_local_histogram_v2<BinSize><<<numBlocks, BinSize>>>(
        ping, original_n, local.data().get(), padded_n, bits_per_round * round);
    nvtxRangePop();

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

    // Transpose localT[bin][block] = local[block][bin]
    dim3 block = dim3(BlockSize, BlockSize, 1);
    dim3 grid = dim3((BinSize + BlockSize - 1) / BlockSize,
                     (numBlocks + BlockSize - 1) / BlockSize, 1);

    nvtxRangePushA("transpose");
    ::gpu::util::kernel_transpose<uint32_t, BlockSize><<<grid, block>>>(
        /*out = */ localT.data().get(),
        /*in = */ local.data().get(), BinSize, numBlocks);
    nvtxRangePop();

    /*
     * reduce local array to global(WITHOUT SCANNING GLOBAL NOW!)
     * for kernel_global_reduce_from_local_v3 only
     */
    uint32_t *local_ptr = local.data().get();
    uint32_t *localT_ptr = localT.data().get();
    uint32_t *read_ptr = reduce_a.data().get();
    uint32_t *write_ptr = reduce_b.data().get();

    std::size_t in_stride = numBlocks;
    std::size_t in_real = numBlocks;

    std::size_t out_stride = pitch;
    std::size_t out_real = (in_real + 1024 - 1) / 1024;

    block = dim3(32, 32, 1);
    grid = dim3(out_real, BinSize, 1);

    /*
     * generate a reduced 2D array for next step reduce
     * and this array will be used for calculating reduce offset operation
     */
    ::sort::gpu::radix::details::v3::kernel_global_reduce_from_local_v3<32>
        <<<grid, block>>>(
            /*in = */ localT_ptr,
            /*out = */ write_ptr, in_stride, in_real, out_stride, out_real);

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

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

#ifndef NDEBUG
      cudahelper::checkCuda(cudaGetLastError());
      cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

      in_real = out_real;
      std::swap(read_ptr, write_ptr);
    }

    uint32_t *final_result = read_ptr;

    // Scanning
    thrust::exclusive_scan(thrust::cuda::par.on(0), final_result,
                           final_result + BinSize, final_result);

    // Copy 2 Global
    cudaMemcpy(global.data().get(), final_result,
               global.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToDevice // DeviceToHost
    );

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

    ::sort::gpu::radix::details::hierarchical_exclusive_scan_localT_1024(
        localT_ptr, big_buffer, levels, BinSize, numBlocks, numBlocks);

    dim3 grid_bt((numBlocks + BlockSize - 1) / BlockSize, // width = numBlocks
                 (BinSize + BlockSize - 1) / BlockSize,   // height = BinSize
                 1);

    nvtxRangePushA("transposeT");
    ::gpu::util::kernel_transpose<uint32_t, BlockSize><<<grid_bt, block>>>(
        /*out = */ local.data().get(),
        /*in = */ localT.data().get(), numBlocks, BinSize);
    nvtxRangePop();

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

    nvtxRangePushA("scatter");
    v2::kernel_radix_scatter_v2<BinSize><<<numBlocks, BinSize>>>(
        ping, pong, padded_n, local.data().get(), padded_n, global.data().get(),
        BinSize, bits_per_round * round);

    nvtxRangePop();

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

    std::swap(ping, pong);
  }

  cudahelper::checkCuda(cudaDeviceSynchronize());
  if (ping != input.data()) {
    cudahelper::checkCuda(cudaMemcpy(input.data(), ping,
                                     original_n * sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost));
  }

  input.resize(original_n);

  global.clear();
  local.clear();
  localT.clear();
  reduce_a.clear();
  reduce_b.clear();

  big_buffer.clear();
}

void __radix_sort_v4(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input) {
  nvtxRangePushA("radix_sort_v4");
  radix_sort_v4<256, 32>(input);
  nvtxRangePop();
}

} // namespace v4
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort
