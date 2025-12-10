#include <vector>
#include <test_transfer_overlap.h>
#include <cuda_tut_transfer_overlap.cuh>

void benchmark_overlap_pipeline_sweep(const int nStreams ) {
          __benchmark_overlap_pipeline_sweep(nStreams);
}

void benchmark_compute_intensity_latency_sweep(const int nStreams) {
          __benchmark_compute_intensity_latency_sweep(nStreams);
}
