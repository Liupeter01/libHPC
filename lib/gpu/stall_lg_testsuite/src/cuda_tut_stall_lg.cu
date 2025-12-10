#include <cuda_tut_stall_lg.cuh>
#include <nvtx3/nvToolsExt.h>
#include <cudaHelper.cuh>

void run_kernel(const char* name, void (*kernel)(int8_t* __restrict, int8_t* __restrict), int8_t* __restrict d_in, int8_t* __restrict d_out) {

		  if (!name || !kernel) return;

		  printf("Current Test is: %s\n", name);
		  // create events and streams
		  cudaEvent_t startEvent, stopEvent;

		  cudahelper::checkCuda(cudaEventCreate(&startEvent));
		  cudahelper::checkCuda(cudaEventCreate(&stopEvent));

		  cudahelper::checkCuda(cudaEventRecord(startEvent, 0));

		  nvtxRangePushA(name);
		  kernel << <64, 256 >> > (d_in, d_out);
		  nvtxRangePop();

		  cudahelper::checkCuda(cudaEventRecord(stopEvent, 0));
		  cudahelper::checkCuda(cudaEventSynchronize(stopEvent));

		  float ms; // elapsed time in milliseconds
		  cudahelper::checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		  printf("Time for %s execute (ms): %f\n", name, ms);
}

__global__ void stall_lg_worse(int8_t* __restrict ptr1, int8_t* __restrict ptr2) {
		  int id = threadIdx.x + blockIdx.x * blockDim.x;
		  int offset = id * 1000;
		  for (int i = 0; i < 2000; ++i) {
					ptr2[offset + i] = ptr1[offset + i];
		  }
}

__global__ void stall_lg_coalesced_32(int8_t* __restrict ptr1, int8_t* __restrict ptr2) {
		  int id = threadIdx.x + blockIdx.x * blockDim.x;
		  for (int i = 0; i < 2000; ++i) {
					ptr2[id + i * blockDim.x] = ptr1[id + i * blockDim.x];
		  }
}

__global__ void stall_lg_coalesced_128(int8_t* __restrict ptr1, int8_t* __restrict ptr2) {
		  int* p1 = reinterpret_cast<int*>(ptr1);
		  int* p2 = reinterpret_cast<int*>(ptr2);
		  int id = threadIdx.x + blockIdx.x * blockDim.x;
		  for (int i = 0; i < 2000 / 4; ++i) {
					p2[blockDim.x * i + id] = p1[blockDim.x * i + id];
		  }
}

__global__ void stall_lg_coalesced_256_best(int8_t* __restrict ptr1, int8_t* __restrict ptr2) {
		  int4* p1 = reinterpret_cast<int4*>(ptr1);
		  int4* p2 = reinterpret_cast<int4*>(ptr2);
		  int id = threadIdx.x + blockIdx.x * blockDim.x;
		  for (int i = 0; i < 2000 / 16; ++i) {
					p2[blockDim.x * i + id] = p1[blockDim.x * i + id];
		  }
}