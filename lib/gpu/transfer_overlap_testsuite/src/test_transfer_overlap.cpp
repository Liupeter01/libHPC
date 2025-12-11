#include <cuda_tut_transfer_overlap.cuh>
#include <test_transfer_overlap.h>
#include <vector>

void benchmark_overlap_pipeline_sweep(const int nStreams) {
  __benchmark_overlap_pipeline_sweep(nStreams);
}

void benchmark_compute_intensity_latency_sweep(const int nStreams) {
  __benchmark_compute_intensity_latency_sweep(nStreams);
}
