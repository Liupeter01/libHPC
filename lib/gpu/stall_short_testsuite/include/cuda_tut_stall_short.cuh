#pragma once
#ifndef _CUDA_TUT_STALL_SHORT_CUH_
#define _CUDA_TUT_STALL_SHORT_CUH_
#include <cuda_runtime.h>

template <typename _Ty>
__global__ void parallel_transpose(_Ty *out, const _Ty *in, int nx, int ny) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int x = linear % nx;
  int y = linear / nx;
  if (x >= nx || y >= ny)
    return;
  out[nx * y + x] = in[nx * x + y];
}

template <typename _Ty>
__global__ void parallel_transpose_dim3(_Ty *out, const _Ty *in, int nx,
                                        int ny) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny)
    return;
  out[nx * y + x] = in[nx * x + y];
}

template <typename _Ty, std::size_t BlockSize>
__global__ void parallel_transpose_shared(_Ty *out, const _Ty *in, int nx,
                                          int ny) {
  int x = blockIdx.x * BlockSize + threadIdx.x;
  int y = blockIdx.y * BlockSize + threadIdx.y;
  if (x >= nx || y >= ny)
    return;
  int rx = blockIdx.y * BlockSize + threadIdx.x;
  int ry = blockIdx.x * BlockSize + threadIdx.y;
  __shared__ volatile _Ty buffer[BlockSize * BlockSize];
  buffer[threadIdx.y * BlockSize + threadIdx.x] = in[nx * ry + rx];
  __syncthreads(); // barrier
  out[y * nx + x] = buffer[threadIdx.x * BlockSize + threadIdx.y];
}

template <typename _Ty, std::size_t BlockSize>
__global__ void parallel_transpose_solv_conflict(_Ty *out, const _Ty *in,
                                                 int nx, int ny) {
  int x = blockIdx.x * BlockSize + threadIdx.x;
  int y = blockIdx.y * BlockSize + threadIdx.y;
  if (x >= nx || y >= ny)
    return;
  int rx = blockIdx.y * BlockSize + threadIdx.x;
  int ry = blockIdx.x * BlockSize + threadIdx.y;
  __shared__ volatile _Ty tile[BlockSize][BlockSize + 1];
  tile[threadIdx.y][threadIdx.x] = in[nx * ry + rx];
  __syncthreads(); // barrier
  out[y * nx + x] = tile[threadIdx.x][threadIdx.y];
}

void __benchmark_baseline();
void __benchmark_dim3();
void __benchmark_dim3_shared();
void __benchmark_dim3_shared_solv_conflict();
void __benchmark_all();

#endif //_CUDA_TUT_STALL_SHORT_CUH_
