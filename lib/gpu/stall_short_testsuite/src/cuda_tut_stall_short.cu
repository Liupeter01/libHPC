#include <vector>
#include <nvtx3/nvToolsExt.h>
#include <cudaAllocator.hpp>
#include <cuda_tut_stall_short.cuh>

void __benchmark_baseline() {
		  float ms; // elapsed time in milliseconds
		  constexpr int nx = 1 << 14, ny = 1 << 14;
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> in(nx * ny);
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> out(nx * ny);
		  for (std::size_t i = 0; i < nx * ny; ++i) {
					in[i] = i;
		  }

		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);
		  cudaEventRecord(startEvent, 0);

		  printf("Current Test is: %s\n", "parallel_transpose_baseline");
		  nvtxRangePushA("parallel_transpose_baseline");
		  parallel_transpose << <(nx * ny) / 1024, 1024 >> > (out.data(), in.data(), nx, ny);
		  nvtxRangePop();

		  cudaEventRecord(stopEvent, 0);
		  cudaEventSynchronize(stopEvent);
		  cudaEventElapsedTime(&ms, startEvent, stopEvent);

		  printf("Time for %s execute (ms): %f\n", "parallel_transpose_baseline", ms);

		  cudaEventDestroy(startEvent);
		  cudaEventDestroy(stopEvent);
}

void __benchmark_dim3() {
		  float ms; // elapsed time in milliseconds
		  constexpr int nx = 1 << 14, ny = 1 << 14;
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> in(nx * ny);
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> out(nx * ny);
		  for (std::size_t i = 0; i < nx * ny; ++i) {
					in[i] = i;
		  }

		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);

		  cudaEventRecord(startEvent, 0);
		  printf("Current Test is: %s\n", "parallel_transpose_dim3");
		  nvtxRangePushA("parallel_transpose_dim3");
		  parallel_transpose_dim3 << <dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1) >> > (out.data(), in.data(), nx, ny);
		  nvtxRangePop();

		  cudaEventRecord(stopEvent, 0);
		  cudaEventSynchronize(stopEvent);
		  cudaEventElapsedTime(&ms, startEvent, stopEvent);

		  printf("Time for %s execute (ms): %f\n", "parallel_transpose_dim3", ms);

		  cudaEventDestroy(startEvent);
		  cudaEventDestroy(stopEvent);
}

void __benchmark_dim3_shared() {
		  float ms; // elapsed time in milliseconds
		  constexpr int nx = 1 << 14, ny = 1 << 14;
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> in(nx * ny);
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> out(nx * ny);
		  for (std::size_t i = 0; i < nx * ny; ++i) {
					in[i] = i;
		  }

		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);

		  cudaEventRecord(startEvent, 0);
		  printf("Current Test is: %s\n", "parallel_transpose_shared");
		  nvtxRangePushA("parallel_transpose_shared");
		  parallel_transpose_shared<int, 32> << <dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1) >> > (out.data(), in.data(), nx, ny);
		  nvtxRangePop();

		  cudaEventRecord(stopEvent, 0);
		  cudaEventSynchronize(stopEvent);
		  cudaEventElapsedTime(&ms, startEvent, stopEvent);

		  printf("Time for %s execute (ms): %f\n", "parallel_transpose_shared", ms);

		  cudaEventDestroy(startEvent);
		  cudaEventDestroy(stopEvent);
}

void __benchmark_dim3_shared_solv_conflict() {
		  float ms; // elapsed time in milliseconds
		  constexpr int nx = 1 << 14, ny = 1 << 14;
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> in(nx * ny);
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> out(nx * ny);
		  for (std::size_t i = 0; i < nx * ny; ++i) {
					in[i] = i;
		  }

		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);

		  cudaEventRecord(startEvent, 0);
		  printf("Current Test is: %s\n", "parallel_transpose_solv_conflict");
		  nvtxRangePushA("parallel_transpose_solv_conflict");
		  parallel_transpose_solv_conflict<int, 32> << <dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1) >> > (out.data(), in.data(), nx, ny);
		  nvtxRangePop();

		  cudaEventRecord(stopEvent, 0);
		  cudaEventSynchronize(stopEvent);
		  cudaEventElapsedTime(&ms, startEvent, stopEvent);

		  printf("Time for %s execute (ms): %f\n", "parallel_transpose_solv_conflict", ms);

		  cudaEventDestroy(startEvent);
		  cudaEventDestroy(stopEvent);
}

void __benchmark_all() {
		  float ms; // elapsed time in milliseconds
		  constexpr int nx = 1 << 14, ny = 1 << 14;
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> in(nx * ny);
		  std::vector<int, CudaAllocator<int, CudaMemManaged>> out(nx * ny);
		  for (std::size_t i = 0; i < nx * ny; ++i) {
					in[i] = i;
		  }

		  cudaEvent_t startEvent, stopEvent;
		  cudaEventCreate(&startEvent);
		  cudaEventCreate(&stopEvent);

		  {
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "parallel_transpose_baseline");
					nvtxRangePushA("parallel_transpose_baseline");
					parallel_transpose << <(nx * ny) / 1024, 1024 >> > (out.data(), in.data(), nx, ny);
					nvtxRangePop();

					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);

					printf("Time for %s execute (ms): %f\n", "parallel_transpose_baseline", ms);
		  }

		  out.clear();

		  {
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "parallel_transpose_dim3");
					nvtxRangePushA("parallel_transpose_dim3");
					parallel_transpose_dim3 << <dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1) >> > (out.data(), in.data(), nx, ny);
					nvtxRangePop();

					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);

					printf("Time for %s execute (ms): %f\n", "parallel_transpose_dim3", ms);
		  }

		  out.clear();

		  {
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "parallel_transpose_shared");
					nvtxRangePushA("parallel_transpose_shared");
					parallel_transpose_shared<int, 32> << <dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1) >> > (out.data(), in.data(), nx, ny);
					nvtxRangePop();

					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);

					printf("Time for %s execute (ms): %f\n", "parallel_transpose_shared", ms);
		  }

		  out.clear();

		  {
					cudaEventRecord(startEvent, 0);
					printf("Current Test is: %s\n", "parallel_transpose_solv_conflict");
					nvtxRangePushA("parallel_transpose_solv_conflict");
					parallel_transpose_solv_conflict<int, 32> << <dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1) >> > (out.data(), in.data(), nx, ny);
					nvtxRangePop();

					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&ms, startEvent, stopEvent);

					printf("Time for %s execute (ms): %f\n", "parallel_transpose_solv_conflict", ms);
		  }

		  cudaEventDestroy(startEvent);
		  cudaEventDestroy(stopEvent);
}