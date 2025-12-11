#include <cudaAllocator.hpp>
#include <test_stall_lg.h>
#include <vector>

void test_cuda_tut_stall_lg() {

  constexpr int nx = 1 << 14, ny = 1 << 14;
  std::vector<int8_t, CudaAllocator<int8_t, CudaMemManaged>> stall_in(nx * ny);
  std::vector<int8_t, CudaAllocator<int8_t, CudaMemManaged>> stall_out(nx * ny);
  for (std::size_t i = 0; i < nx * ny; ++i) {
    stall_in[i] = static_cast<int8_t>(1);
  }

  run_kernel("stall_lg_worse", stall_lg_worse, stall_in.data(),
             stall_out.data());
  stall_out.clear();
  run_kernel("stall_lg_coalesced_32", stall_lg_coalesced_32, stall_in.data(),
             stall_out.data());
  stall_out.clear();
  run_kernel("stall_lg_coalesced_128", stall_lg_coalesced_128, stall_in.data(),
             stall_out.data());
  stall_out.clear();
  run_kernel("stall_lg_coalesced_256_best", stall_lg_coalesced_256_best,
             stall_in.data(), stall_out.data());
}
