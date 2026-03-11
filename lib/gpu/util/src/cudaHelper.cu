#include <cudaHelper.cuh>

namespace cudahelper {

__global__ void kernel_clear_u32(uint32_t *__restrict ptr, std::size_t n) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = idx; i < n; i += stride) {
    ptr[i] = 0;
  }
}
} // namespace cudahelper
