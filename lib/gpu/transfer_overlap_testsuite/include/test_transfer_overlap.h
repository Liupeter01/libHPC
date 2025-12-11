#pragma once
#ifndef _TEST_TRANSFER_OVERLAP_H_
#define _TEST_TRANSFER_OVERLAP_H_
#include <cuda_tut_transfer_overlap.cuh>

// Hypothesis:
// Stream-level transfer/compute overlapping efficiency is determined primarily
// by pipeline scheduling order rather than stream count (nStreams ¡Ö 4 fixed).
void benchmark_overlap_pipeline_sweep(const int nStreams = 4);

// Hypothesis:
// Increasing compute intensity improves H2D&D2H latency hiding up to
// saturation, after which further unrolling shows diminishing returns.
void benchmark_compute_intensity_latency_sweep(const int nStreams = 4);

#endif //_TEST_TRANSFER_OVERLAP_H_
