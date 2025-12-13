#pragma once
#ifndef _TEST_STALL_SHORT_H_
#define _TEST_STALL_SHORT_H_
#include <cuda_tut_stall_short.cuh>

void benchmark_baseline();
void benchmark_dim3();
void benchmark_dim3_shared();
void benchmark_dim3_shared_solv_conflict();
void benchmark_all();

#endif //_TEST_STALL_SHORT_H_
