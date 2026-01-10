#include <cudaUtils.cuh>
#include <cuda_radix_sort.cuh>
#include <thrust/device_vector.h>

namespace sort {
namespace gpu {
namespace radix {
namespace details {

__global__ void kernel_startup() { (void)threadIdx.x; }

void __kernel_startup() { kernel_startup<<<1, 1>>>(); }
} // namespace details
} // namespace radix
} // namespace gpu
} // namespace sort
