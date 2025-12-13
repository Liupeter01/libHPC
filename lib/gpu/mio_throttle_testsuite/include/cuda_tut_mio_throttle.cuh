#pragma once
#ifndef _CUDA_TUT_MIO_THROTTLE_CUH_
#define _CUDA_TUT_MIO_THROTTLE_CUH_
#include <cuda_runtime.h>

__global__ void stall_mio_worse();
__global__ void stall_mio_better();
__global__ void stall_mio_good();

void __benchmark_worse();
void __benchmark_better();
void __benchmark_good();
void __benchmark_all();

#endif //_CUDA_TUT_MIO_THROTTLE_CUH_