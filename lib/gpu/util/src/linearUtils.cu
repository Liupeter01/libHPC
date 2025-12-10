#include <linearUtils.cuh>

namespace cudahelper {
          namespace util {
                    __device__ std::size_t global_linear_tid() {
                              std::size_t b =
                                        blockIdx.x +
                                        gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

                              std::size_t t =
                                        threadIdx.x +
                                        blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

                              return b * (blockDim.x * blockDim.y * blockDim.z) + t;
                    }

                    __device__  std::size_t global_linear_stride() {
                              return (std::size_t)gridDim.x * gridDim.y * gridDim.z *
                                        (std::size_t)blockDim.x * blockDim.y * blockDim.z;
                    }
          }
}