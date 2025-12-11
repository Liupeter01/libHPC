#pragma once
#ifndef _CUDA_TUT_TRANSFER_OVERLAP_HPP_
#define _CUDA_TUT_TRANSFER_OVERLAP_HPP_
#include <cudaHelper.cuh>
#include <cuda_runtime.h>
#include <iostream>

__global__ void simple_kernel(float *a, int offset);
__global__ void compute_kernel_unroll4(float *a, int offset);
__global__ void compute_kernel_unroll8(float *a, int offset);

void __benchmark_overlap_pipeline_sweep(const int nStreams);
void __benchmark_compute_intensity_latency_sweep(const int nStreams);

#endif //_CUDA_TUT_TRANSFER_OVERLAP_HPP_
