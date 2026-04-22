# libHPC

libHPC is a high-performance computing library focused on Linux and Windows
environments. It provides SIMD-optimized kernels, concurrent data structures,
GPU utilities, and HPC-oriented memory management components.

## Project Status

This public archive preserves the state of libHPC at the point where its core
HPC primitives — GPU radix sort, ABA-safe lock-free queue, SIMD kernels,
and cache-hierarchy benchmarks — reached a stable, validated milestone.

Active development continues privately. The archive is retained for study,
reference, and portfolio purposes; commercial use, redistribution, or
derivative proprietary work without explicit permission is not permitted.

Platform support status, known limitations, and benchmarks are documented
in the sections below.

## 0x00 Platform Support

| Platform                          | Status               |
| --------------------------------- | -------------------- |
| **Linux (x86_64 / CUDA)**         | ✓ Supported          |
| **Windows (MSVC / CUDA)**         | ✓ Supported          |
| **macOS (Intel)**                 | ✓ Supported, limited |
| **macOS (Apple Silicon / ARM64)** | ✗ **Not supported**  |

### macOS ARM Technical Post-Mortem

libHPC previously supported macOS ARM. However, recent Xcode toolchains explicitly mark several libc++ ABI symbols as **FORBIDDEN** (Xcode displays a “prohibited symbol” icon).

Specifically, `std::__1::__hash_memory`, a critical dependency for oneTBB, has been removed/hidden at the SDK level.

Since this is a breaking change in the Apple SDK/Toolchain itself, it cannot be resolved within libHPC. As a result, macOS ARM support has been formally dropped to maintain the integrity of the HPC pipeline.


## 0x01 macOS Apple Silicon Notice

libHPC does not support macOS ARM (Apple Silicon).

The reason is simple:

Apple’s recent macOS / Xcode toolchain updates introduced ABI changes in libc++, causing oneTBB and other HPC components to fail at link-time.

> **Apple’s recent macOS / Xcode toolchain updates introduced ABI changes in `libc++`, causing oneTBB and other HPC components to fail at link-time.**  
>
> These issues do not occur on Linux or Windows, and they did not occur on older macOS versions.

Since the goal of libHPC is stable, reproducible high-performance computing,
macOS ARM is excluded to avoid degraded reliability or performance.

---

## 0x02 GPU Performance Optimization Highlights

libHPC includes GPU-accelerated kernels optimized for high-throughput computation on NVIDIA CUDA-compatible devices:

- **Radix-Sort Kernel:** Processes 500M elements in ~360ms on an RTX 3080 Ti(laptop), sustaining ~1.39B elements/sec throughput.  
- **Warp-Synchronous & Tiled Memory Layouts:** Maximizes shared memory utilization and minimizes global memory latency.  
- **Concurrent GPU Pipelines:** Supports asynchronous kernel launches and stream-based scheduling for overlapping compute and memory operations.  
- **Profiling & Validation:** Includes tools for warp efficiency, memory access analysis, and synchronization correctness across GPU architectures.  
- **Realistic HPC Throughput:** Designed for bulk-parallel computation and scientific workloads, **not** real-time ultra-low-latency trading systems.
