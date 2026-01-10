#include <cudaUtils.cuh>
#include <cuda_global_reduce_from_local.cuh>
#include <cuda_local_histogram.cuh>
#include <cuda_radix_scatter.cuh>
#include <cuda_radix_sort_v1.cuh>
#include <thrust/device_vector.h>

namespace sort {
namespace gpu {
namespace radix {
namespace details {
namespace v1 {
template <std::size_t BinSize, std::size_t BlockSize,
          std::enable_if_t<(BinSize == 256u), int> = 0>
static inline void radix_sort_v1(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input) {
  static constexpr std::size_t bits_per_round = constexpr_log2(BinSize);
  static constexpr std::size_t round_number =
      std::numeric_limits<uint32_t>::digits / bits_per_round;

  static_assert((BinSize & (BinSize - 1)) == 0, "BinSize must be power of two");
  static_assert((std::numeric_limits<uint32_t>::digits % bits_per_round) == 0,
                "BinSize must evenly divide key bit width");

  static_assert(BinSize == 256u, "BinSize must be 256 in v1");
  static_assert(std::is_unsigned_v<uint32_t>,
                "radix_sort expects unsigned integers");

  if (input.empty())
    return;

  std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> backup;
  const std::size_t original_n = input.size();
  const std::size_t numBlocks = (original_n + BinSize - 1) / BinSize;
  const std::size_t padded_n = numBlocks * BinSize;

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

  //[numBlocks][BinSize]
  thrust::device_vector<uint32_t> local(padded_n);

  //[BinSize][numBlocks]
  thrust::device_vector<uint32_t> localT(padded_n);

  for (std::size_t round = 0; round < round_number; ++round) {
    thrust::fill_n(global.begin(), BinSize, 0);
    thrust::fill_n(local.begin(), padded_n, 0);
    thrust::fill_n(localT.begin(), padded_n, 0);

    // generate local

    nvtxRangePushA("histogram");
    kernel_local_histogram_v1<BinSize><<<numBlocks, BinSize>>>(
        ping, original_n, local.data().get(), padded_n, bits_per_round * round);
    nvtxRangePop();

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

    nvtxRangePushA("global-reduce");
    // reduce local array to global(WITHOUT SCANNING GLOBAL NOW!)
    kernel_global_reduce_from_local_v1<BinSize><<<numBlocks, BinSize>>>(
        local.data().get(), padded_n, global.data().get());
    nvtxRangePop();

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

    // Scanning Global
    thrust::exclusive_scan(thrust::cuda::par.on(0), global.begin(),
                           global.end(), global.begin());

    // Transpose localT[bin][block] = local[block][bin]
    dim3 block(BlockSize, BlockSize, 1);
    dim3 grid((BinSize + BlockSize - 1) / BlockSize,
              (numBlocks + BlockSize - 1) / BlockSize, 1);

    nvtxRangePushA("transpose");
    ::gpu::util::kernel_transpose<uint32_t, BlockSize><<<grid, block>>>(
        /*out = */ localT.data().get(),
        /*in = */ local.data().get(), BinSize, numBlocks);
    nvtxRangePop();

#ifndef NDEBUG
    cudahelper::checkCuda(cudaGetLastError());
    cudahelper::checkCuda(cudaDeviceSynchronize());
#endif

    for (int bin = 0; bin < (int)BinSize; ++bin) {
      auto first = localT.begin() + bin * numBlocks;
      auto last = first + numBlocks;
      thrust::exclusive_scan(thrust::cuda::par.on(0), first, last, first);
    }

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
    kernel_radix_scatter_v1<BinSize><<<numBlocks, BinSize>>>(
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
}

void __radix_sort_v1(
    std::vector<uint32_t, CudaAllocator<uint32_t, CudaMemManaged>> &input) {
  radix_sort_v1<256, 32>(input);
}
} // namespace v1
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort
