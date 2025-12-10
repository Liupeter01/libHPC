#include <vector>
#include <nvtx3/nvToolsExt.h>
#include <cuda_tut_transfer_overlap.cuh>

__global__ void simple_kernel(float* a, int offset) {
          int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
          float temp = a[i];
          float x = (float)i;
          float s = sinf(x);
          float c = cosf(x);
          temp += sqrtf(s * s + c * c);
          a[i] = temp;
}

__global__ void compute_kernel_unroll4(float* a, int offset)
{
          int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
          float temp = a[i];
          float x = (float)i;
          float s = sinf(x);
          float c = cosf(x);
          temp += sqrtf(s * s + c * c);
#pragma unroll 4
          for (int k = 0; k < 64; k++)
                    temp += sinf(x) * cosf(x);
          a[i] = temp;
}

__global__ void compute_kernel_unroll8(float* a, int offset)
{
          int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
          float temp = a[i];
          float x = (float)i;
          float s = sinf(x);
          float c = cosf(x);
          temp += sqrtf(s * s + c * c);
#pragma unroll 8
          for (int k = 0; k < 64; k++)
                    temp += sinf(x) * cosf(x);
          a[i] = temp;
}

void __benchmark_overlap_pipeline_sweep(const int nStreams) {
          /* ------------------------- Config ------------------------- */
          const int blockSize = 256;
          const int n = 16 * 1024 * blockSize * nStreams;
          const int chunk = n / nStreams;
          const int bytes = n * sizeof(float);
          const int chunkBytes = chunk * sizeof(float);

          /* ------------------------- Memory ------------------------- */
          float* h; float* d;
          cudaMallocHost(&h, bytes);
          cudaMalloc(&d, bytes);
          memset(h, 0, bytes);

          /* ------------------------- Timing ------------------------- */
          cudaEvent_t start, stop;
          cudaEventCreate(&start);
          cudaEventCreate(&stop);

          std::vector<cudaStream_t> streams(nStreams);
          for (auto& s : streams) cudaStreamCreate(&s);

          float ms = 0.0f;


          /* ============================================================
                              Baseline (No Overlap)
             ============================================================ */
          nvtxRangePushA("Baseline: memcpy + kernel + memcpy");
          cudaEventRecord(start);

          cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
          simple_kernel<< <n / blockSize, blockSize >> > (d, 0);
          cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);

          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&ms, start, stop);
          nvtxRangePop();

          printf("[Baseline] latency (ms): %f\n", ms);

          memset(h, 0, bytes);


          /* ============================================================
                  Pattern A: Pipelines { copy ¡ú kernel ¡ú copy }
             ============================================================ */
          nvtxRangePushA("Overlap Pattern A: {copy ¡ú kernel ¡ú copy}");
          cudaEventRecord(start);

          for (int i = 0; i < nStreams; ++i) {
                    int off = i * chunk;
                    cudaMemcpyAsync(&d[off], &h[off], chunkBytes, cudaMemcpyHostToDevice, streams[i]);
                    simple_kernel << <chunk / blockSize, blockSize, 0, streams[i] >> > (d, off);
                    cudaMemcpyAsync(&h[off], &d[off], chunkBytes, cudaMemcpyDeviceToHost, streams[i]);
          }

          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&ms, start, stop);
          nvtxRangePop();

          printf("[Overlap A] latency (ms): %f\n", ms);

          memset(h, 0, bytes);


          /* ============================================================
           Pattern B: batch copy ¡ú batch kernel ¡ú batch copy
             ============================================================ */
          nvtxRangePushA("Overlap Pattern B: batch(copy ¡ú kernel ¡ú copy)");
          cudaEventRecord(start);

          for (int i = 0; i < nStreams; ++i)
                    cudaMemcpyAsync(&d[i * chunk], &h[i * chunk], chunkBytes, cudaMemcpyHostToDevice, streams[i]);

          for (int i = 0; i < nStreams; ++i)
                    simple_kernel << <chunk / blockSize, blockSize, 0, streams[i] >> > (d, i * chunk);

          for (int i = 0; i < nStreams; ++i)
                    cudaMemcpyAsync(&h[i * chunk], &d[i * chunk], chunkBytes, cudaMemcpyDeviceToHost, streams[i]);

          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&ms, start, stop);
          nvtxRangePop();

          printf("[Overlap B] latency (ms): %f\n", ms);


          /* ------------------------- Cleanup ------------------------- */
          for (auto& s : streams) cudaStreamDestroy(s);
          cudaFree(d);
          cudaFreeHost(h);
          cudaEventDestroy(start);
          cudaEventDestroy(stop);
}

void __benchmark_compute_intensity_latency_sweep(const int nStreams) {
          /* ------------------------- Config ------------------------- */
          const int blockSize = 256;
          const int n = 16 * 1024 * blockSize * nStreams;
          const int chunk = n / nStreams;
          const int bytes = n * sizeof(float);
          const int chunkBytes = chunk * sizeof(float);

          /* ------------------------- Memory ------------------------- */
          float* h;
          float* d;
          cudaMallocHost(&h, bytes);
          cudaMalloc(&d, bytes);
          memset(h, 0, bytes);

          /* ------------------------- Timing ------------------------- */
          cudaEvent_t start, stop;
          cudaEventCreate(&start);
          cudaEventCreate(&stop);

          std::vector<cudaStream_t> streams(nStreams);
          for (auto& s : streams) cudaStreamCreate(&s);

          float ms = 0.0f;
          auto run_case = [&](auto kernel, const char* tag) {

                    memset(h, 0, bytes);
                    nvtxRangePushA(tag);
                    cudaEventRecord(start);

                    /* --- Copy --- */
                    for (int i = 0; i < nStreams; ++i)
                              cudaMemcpyAsync(&d[i * chunk], &h[i * chunk],
                                        chunkBytes, cudaMemcpyHostToDevice, streams[i]);

                    /* --- Kernel --- */
                    for (int i = 0; i < nStreams; ++i)
                              kernel << <chunk / blockSize, blockSize, 0, streams[i] >> > (d, i * chunk);

                    /* --- Copy Back --- */
                    for (int i = 0; i < nStreams; ++i)
                              cudaMemcpyAsync(&h[i * chunk], &d[i * chunk],
                                        chunkBytes, cudaMemcpyDeviceToHost, streams[i]);

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&ms, start, stop);
                    nvtxRangePop();

                    printf("[%-30s] latency (ms): %f\n", tag, ms);
                    };


          /* ============================================================
               1. Baseline Compute: simple_kernel
             ============================================================ */
          run_case(simple_kernel, "simple kernel");


          /* ============================================================
               2. Medium compute: unroll 4
             ============================================================ */
          run_case(compute_kernel_unroll4, "unroll4: medium compute");


          /* ============================================================
               3. Heavy compute: unroll 8
             ============================================================ */
          run_case(compute_kernel_unroll8, "unroll8: heavy compute");


          /* ------------------------- Cleanup ------------------------- */
          for (auto& s : streams) cudaStreamDestroy(s);
          cudaFree(d);
          cudaFreeHost(h);
          cudaEventDestroy(start);
          cudaEventDestroy(stop);
}