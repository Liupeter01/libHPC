#pragma once
#ifndef _CUDA_TUT_STALL_LG_HPP_
#define _CUDA_TUT_STALL_LG_HPP_
#include <cuda_runtime.h>
#include <iostream>

void run_kernel(const char *name,
                void (*kernel)(int8_t *__restrict, int8_t *__restrict),
                int8_t *__restrict d_in, int8_t *__restrict d_out);

__global__ void stall_lg_worse(int8_t *__restrict, int8_t *__restrict);
__global__ void stall_lg_coalesced_32(int8_t *__restrict, int8_t *__restrict);
__global__ void stall_lg_coalesced_128(int8_t *__restrict, int8_t *__restrict);
__global__ void stall_lg_coalesced_256_best(int8_t *__restrict,
                                            int8_t *__restrict);

#endif //_CUDA_TUT_STALL_LG_HPP_
