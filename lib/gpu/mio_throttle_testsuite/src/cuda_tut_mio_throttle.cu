#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <cuda_tut_mio_throttle.cuh>

__global__ void stall_mio_worse() {
		  int id = (threadIdx.x + blockIdx.x * blockDim.x) & 31;	// % 32
		  __shared__  volatile int block1[32][32];
		  __shared__  volatile int block2[32][32];
		  for (int i = 0; i < 32; ++i) {
					block2[id][i] = block1[id][i];
		  }
		  __syncthreads();
}

__global__ void stall_mio_better() {
		  int id = (threadIdx.x + blockIdx.x * blockDim.x) & 31;	// % 32
		  __shared__ volatile int block1[32][32];
		  __shared__  volatile  int block2[32][32];
		  for (int i = 0; i < 32; ++i) {
					block2[i][id] = block1[id][i];
		  }
		  __syncthreads();
}

__global__ void stall_mio_good() {
		  int id = (threadIdx.x + blockIdx.x * blockDim.x) & 31;	// % 32
		  __shared__ volatile int block1[32][33];
		  __shared__ volatile  int block2[32][32];
		  for (int i = 0; i < 32; ++i) {
					block2[i][id] = block1[id][i + 1];
		  }
		  __syncthreads();
}

void __benchmark_worse() {
		  
		  float ms;
		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);

					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "stall_mio_worse ");
					nvtxRangePushA("stall_mio_worse ");

					stall_mio_worse << <1024, 1024 >> > ();
					nvtxRangePop();

					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);

					printf("Time for %s execute (ms): %f\n", "stall_mio_worse ", ms);
					cudaEventDestroy(startEvent);
					cudaEventDestroy(stopEvent);
}

void __benchmark_better() {

		  float ms;
		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);
		  
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "stall_mio_better");
					nvtxRangePushA("stall_mio_better ");
					stall_mio_better << <1024, 1024 >> > ();
					nvtxRangePop();
					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);
					printf("Time for %s execute (ms): %f\n", "stall_mio_better", ms);

					cudaEventDestroy(startEvent);
					cudaEventDestroy(stopEvent);
}

void __benchmark_good() {
		  float ms;
		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);

		  
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "stall_mio_good");
					nvtxRangePushA("stall_mio_good");
					stall_mio_good << <1024, 1024 >> > ();
					nvtxRangePop();
					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);
					printf("Time for %s execute (ms): %f\n", "stall_mio_good", ms);
		  

		  cudaEventDestroy(startEvent);
		  cudaEventDestroy(stopEvent);
}

void __benchmark_all() {
		  float ms;
		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);

		  {
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "stall_mio_worse ");
					nvtxRangePushA("stall_mio_worse ");

					stall_mio_worse << <1024, 1024 >> > ();
					nvtxRangePop();

					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);

					printf("Time for %s execute (ms): %f\n", "stall_mio_worse ", ms);
		  }

		  {
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "stall_mio_better");
					nvtxRangePushA("stall_mio_better ");
					stall_mio_better << <1024, 1024 >> > ();
					nvtxRangePop();
					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);
					printf("Time for %s execute (ms): %f\n", "stall_mio_better", ms);
		  }

		  {
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "stall_mio_good");
					nvtxRangePushA("stall_mio_good");
					stall_mio_good << <1024, 1024 >> > ();
					nvtxRangePop();
					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);
					printf("Time for %s execute (ms): %f\n", "stall_mio_good", ms);
		  }

		  cudaEventDestroy(startEvent);
		  cudaEventDestroy(stopEvent);
}